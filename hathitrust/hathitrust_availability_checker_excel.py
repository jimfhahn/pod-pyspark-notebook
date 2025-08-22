import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Set, Tuple
import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hathitrust_scan.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HathiTrustFullScanner:
    def __init__(self, max_workers: int = 5, rate_limit_delay: float = 0.2):
        """Initialize scanner with configurable parameters"""
        self.hathitrust_base_url = "https://catalog.hathitrust.org/api/volumes"
        self.session = requests.Session()
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.results = []
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print("\n\nInterrupted! Saving results...")
        self.interrupted = True
        self._save_intermediate_results()
        sys.exit(0)

    def _save_intermediate_results(self):
        if self.results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            reports_dir = os.path.join('hathitrust', 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            filename = os.path.join(reports_dir, f"hathitrust_results_{timestamp}.csv")
            df = pd.DataFrame(self.results)
            df.to_csv(filename, index=False)
            print(f"\nResults saved to: {filename}")
            print(f"Processed {len(self.results)} records")

    def query_hathitrust(self, identifier_type: str, identifier: str) -> Dict:
        url = f"{self.hathitrust_base_url}/brief/{identifier_type}/{identifier}.json"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, dict) and 'items' in data and data['items']:
                        return {
                            'found': True,
                            'num_items': len(data['items']),
                            'items': data['items'],
                            'records': data.get('records', {})
                        }
                except:
                    pass
            return {'found': False, 'num_items': 0, 'items': [], 'records': {}}
        except Exception as e:
            logger.error(f"Error querying {identifier_type}:{identifier} - {str(e)}")
            return {'found': False, 'num_items': 0, 'items': [], 'records': {}, 'error': str(e)}

    def process_record(self, row: pd.Series, row_index: int) -> Dict:
        result = {
            'index': row_index,
            'mms_id': row.get('MMS_ID', 'N/A'),
            'title': str(row.get('F245', 'N/A'))[:100],
            'borrowdir_id': row.get('borrowdir_id', 'N/A'),
            'found_in_hathitrust': False,
            'hathitrust_items': 0,
            'full_view_available': False,
            'tested_identifiers': [],
            'matching_identifier': None,
            'matching_type': None,
            'error': None
        }
        if pd.notna(row.get('F260_str')):
            year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', str(row['F260_str']))
            if year_matches:
                result['pub_year'] = int(year_matches[0])
        if pd.notna(row.get('F020_str')):
            isbn_str = str(row['F020_str'])
            isbns = re.findall(r'\b(\d{9}[\dXx]|\d{13})\b', isbn_str)
            for isbn in isbns[:2]:
                isbn_clean = re.sub(r'[^0-9Xx]', '', isbn).upper()
                if isbn_clean:
                    result['tested_identifiers'].append(f"ISBN:{isbn_clean}")
                    response = self.query_hathitrust('isbn', isbn_clean)
                    if response['found']:
                        result['found_in_hathitrust'] = True
                        result['hathitrust_items'] = response['num_items']
                        result['matching_identifier'] = isbn_clean
                        result['matching_type'] = 'ISBN'
                        for item in response['items']:
                            if 'Full' in item.get('usRightsString', ''):
                                result['full_view_available'] = True
                                break
                        break
                    time.sleep(self.rate_limit_delay)
        if not result['found_in_hathitrust'] and pd.notna(row.get('F010_str')):
            lccn_str = str(row['F010_str'])
            clean_lccn = re.sub(r'[^0-9a-zA-Z]', '', lccn_str).strip()
            if clean_lccn:
                result['tested_identifiers'].append(f"LCCN:{clean_lccn}")
                response = self.query_hathitrust('lccn', clean_lccn)
                if response['found']:
                    result['found_in_hathitrust'] = True
                    result['hathitrust_items'] = response['num_items']
                    result['matching_identifier'] = clean_lccn
                    result['matching_type'] = 'LCCN'
                    for item in response['items']:
                        if 'Full' in item.get('usRightsString', ''):
                            result['full_view_available'] = True
                            break
                time.sleep(self.rate_limit_delay)
        if not result['found_in_hathitrust'] and pd.notna(row.get('id_list_str')):
            oclc_matches = re.findall(r'(?:OCoLC|oclc|OCLC)[:\s]*(\d+)', str(row['id_list_str']))
            for oclc in oclc_matches[:1]:
                oclc_clean = oclc.lstrip('0')
                if oclc_clean:
                    result['tested_identifiers'].append(f"OCLC:{oclc_clean}")
                    response = self.query_hathitrust('oclc', oclc_clean)
                    if response['found']:
                        result['found_in_hathitrust'] = True
                        result['hathitrust_items'] = response['num_items']
                        result['matching_identifier'] = oclc_clean
                        result['matching_type'] = 'OCLC'
                        for item in response['items']:
                            if 'Full' in item.get('usRightsString', ''):
                                result['full_view_available'] = True
                                break
                        break
                    time.sleep(self.rate_limit_delay)
        return result

    def scan_full_file(self, excel_path: str, start_from: int = 0, batch_size: int = 100):
        print("\n" + "="*60)
        print("HATHITRUST FULL FILE SCAN")
        print("="*60)
        print("\nLoading Excel file...")
        df = pd.read_excel(excel_path)
        total_records = len(df)
        print(f"Total records: {total_records:,}")
        print(f"Starting from record: {start_from}")
        print(f"Batch size: {batch_size}")
        print(f"Max workers: {self.max_workers}")
        print(f"Rate limit delay: {self.rate_limit_delay}s")
        print("\nFiltering records with identifiers...")
        mask = df['F020_str'].notna() | df['F010_str'].notna()
        if 'id_list_str' in df.columns:
            oclc_mask = df['id_list_str'].astype(str).str.contains(
                r'(?:OCoLC|oclc|OCLC)[:\s]*\d+',
                regex=True,
                na=False
            )
            mask = mask | oclc_mask
        filtered_df = df[mask].iloc[start_from:]
        records_to_process = len(filtered_df)
        print(f"Records with identifiers: {records_to_process:,}")
        if records_to_process == 0:
            print("\nNo records with identifiers to process!")
            return
        print(f"\nProcessing will take approximately {records_to_process * self.rate_limit_delay / 60:.1f} minutes")
        print("\nStarting scan... (Press Ctrl+C to interrupt and save progress)\n")
        start_time = datetime.now()
        matches_found = 0
        full_view_found = 0
        errors = 0
        with tqdm(total=records_to_process, desc="Scanning records", unit="rec") as pbar:
            for batch_start in range(0, records_to_process, batch_size):
                if self.interrupted:
                    break
                batch_end = min(batch_start + batch_size, records_to_process)
                batch_df = filtered_df.iloc[batch_start:batch_end]
                for idx, row in batch_df.iterrows():
                    if self.interrupted:
                        break
                    result = self.process_record(row, idx)
                    self.results.append(result)
                    if result['found_in_hathitrust']:
                        matches_found += 1
                        if result['full_view_available']:
                            full_view_found += 1
                    if result.get('error'):
                        errors += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        'Matches': matches_found,
                        'Full View': full_view_found,
                        'Match Rate': f"{matches_found/(len(self.results))*100:.1f}%"
                    })
                if len(self.results) % (batch_size * 5) == 0:
                    self._save_intermediate_results()
        end_time = datetime.now()
        duration = end_time - start_time
        print("\n" + "="*60)
        print("SCAN COMPLETE")
        print("="*60)
        print(f"\nProcessing Summary:")
        print(f"  Total time: {duration}")
        print(f"  Records processed: {len(self.results):,}")
        print(f"  Records/minute: {len(self.results) / (duration.total_seconds() / 60):.1f}")
        print(f"  HathiTrust matches: {matches_found:,} ({matches_found/len(self.results)*100:.1f}%)")
        print(f"  Full view available: {full_view_found:,} ({full_view_found/len(self.results)*100:.1f}%)")
        print(f"  Errors: {errors}")
        self._save_final_results()

    def _save_final_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = os.path.join('hathitrust', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        results_df = pd.DataFrame(self.results)
        results_filename = os.path.join(reports_dir, f"hathitrust_scan_results_{timestamp}.csv")
        results_df.to_csv(results_filename, index=False)
        print(f"\nDetailed results saved to: {results_filename}")
        self._generate_summary_report(results_df, timestamp)

    def _generate_summary_report(self, results_df: pd.DataFrame, timestamp: str):
        report_filename = os.path.join('hathitrust', 'reports', f"hathitrust_scan_summary_{timestamp}.txt")
        with open(report_filename, 'w') as f:
            f.write("HATHITRUST SCAN SUMMARY REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total records processed: {len(results_df):,}\n\n")
            matches = results_df['found_in_hathitrust'].sum()
            full_view = results_df['full_view_available'].sum()
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"HathiTrust matches: {matches:,} ({matches/len(results_df)*100:.2f}%)\n")
            f.write(f"Full view available: {full_view:,} ({full_view/len(results_df)*100:.2f}%)\n")
            f.write(f"Limited view only: {matches - full_view:,}\n")
            f.write(f"Not in HathiTrust: {len(results_df) - matches:,}\n\n")
            f.write("MATCHES BY IDENTIFIER TYPE\n")
            f.write("-" * 30 + "\n")
            for id_type in ['ISBN', 'LCCN', 'OCLC']:
                type_matches = results_df[results_df['matching_type'] == id_type].shape[0]
                f.write(f"{id_type}: {type_matches:,} matches\n")
            if 'pub_year' in results_df.columns:
                f.write("\nPUBLICATION YEAR ANALYSIS\n")
                f.write("-" * 30 + "\n")
                year_df = results_df[results_df['pub_year'].notna()].copy()
                if not year_df.empty:
                    year_df['decade'] = (year_df['pub_year'] // 10) * 10
                    decade_matches = year_df.groupby('decade')['found_in_hathitrust'].agg(['sum', 'count'])
                    for decade, row in decade_matches.iterrows():
                        match_rate = row['sum'] / row['count'] * 100
                        f.write(f"{int(decade)}s: {int(row['sum'])}/{int(row['count'])} ({match_rate:.1f}%)\n")
            f.write("\nSAMPLE OF MATCHED RECORDS\n")
            f.write("-" * 30 + "\n")
            matched_sample = results_df[results_df['found_in_hathitrust']].head(10)
            for _, record in matched_sample.iterrows():
                f.write(f"\nMMS ID: {record['mms_id']}\n")
                f.write(f"Title: {record['title']}\n")
                f.write(f"Match Type: {record['matching_type']}\n")
                f.write(f"Identifier: {record['matching_identifier']}\n")
                f.write(f"Full View: {'Yes' if record['full_view_available'] else 'No'}\n")
        print(f"Summary report saved to: {report_filename}")

    def resume_from_checkpoint(self, checkpoint_file: str):
        print(f"\nResuming from checkpoint: {checkpoint_file}")
        checkpoint_df = pd.read_csv(checkpoint_file)
        self.results = checkpoint_df.to_dict('records')
        print(f"Loaded {len(self.results)} previous results")
        return len(self.results)

def main():
    parser = argparse.ArgumentParser(description='Full HathiTrust scan with progress tracking')
    parser.add_argument('--input-file', 
                       default='penn_only_holdings_text_format.xlsx',
                       help='Path to the Excel file')
    parser.add_argument('--start-from',
                       type=int,
                       default=0,
                       help='Record number to start from (for resuming)')
    parser.add_argument('--batch-size',
                       type=int,
                       default=100,
                       help='Number of records per batch')
    parser.add_argument('--max-workers',
                       type=int,
                       default=5,
                       help='Maximum concurrent workers')
    parser.add_argument('--rate-limit',
                       type=float,
                       default=0.2,
                       help='Delay between API calls in seconds')
    parser.add_argument('--resume',
                       help='Resume from checkpoint CSV file')
    args = parser.parse_args()
    if not os.path.exists(args.input_file):
        print(f"❌ File not found: {args.input_file}")
        return
    scanner = HathiTrustFullScanner(
        max_workers=args.max_workers,
        rate_limit_delay=args.rate_limit
    )
    start_from = args.start_from
    if args.resume:
        start_from = scanner.resume_from_checkpoint(args.resume)
    scanner.scan_full_file(
        args.input_file,
        start_from=start_from,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
