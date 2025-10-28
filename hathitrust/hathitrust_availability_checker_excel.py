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
    def __init__(self, max_workers: int = 5, rate_limit_delay: float = 0.2, output_dir: str = None):
        """Initialize scanner with configurable parameters"""
        self.hathitrust_base_url = "https://catalog.hathitrust.org/api/volumes"
        self.session = requests.Session()
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.results = []
        self.interrupted = False
        self.output_dir = output_dir  # Store output directory for results
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print("\n\nInterrupted! Saving results...")
        self.interrupted = True
        self._save_intermediate_results()
        sys.exit(0)

    def _save_intermediate_results(self, is_checkpoint: bool = True):
        """Save intermediate results with configurable location"""
        if not self.results:
            logger.warning("No results to save")
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use configured output directory or default
        if self.output_dir:
            reports_dir = self.output_dir
        else:
            reports_dir = os.path.join('hathitrust', 'reports')
        
        os.makedirs(reports_dir, exist_ok=True)
        
        # Use descriptive filename
        prefix = "checkpoint" if is_checkpoint else "results"
        filename = os.path.join(
            reports_dir, 
            f"hathitrust_{prefix}_{timestamp}.csv"
        )
        
        try:
            df = pd.DataFrame(self.results)
            df.to_csv(filename, index=False)
            logger.info(f"Results saved to: {filename}")
            logger.info(f"Processed {len(self.results)} records")
            return filename
        except Exception as e:
            logger.error(f"Error saving results to {filename}: {e}")
            return None

    def query_hathitrust(self, identifier_type: str, identifier: str) -> Dict:
        url = f"{self.hathitrust_base_url}/brief/{identifier_type}/{identifier}.json"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, dict) and 'items' in data and \
                            data['items']:
                        return {
                            'found': True,
                            'num_items': len(data['items']),
                            'items': data['items'],
                            'records': data.get('records', {})
                        }
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON parse error: {json_err}")
            return {'found': False, 'num_items': 0, 'items': [], 'records': {}}
        except Exception as e:
            logger.error(
                f"Error querying {identifier_type}:{identifier} - {str(e)}"
            )
            return {
                'found': False, 'num_items': 0, 'items': [],
                'records': {}, 'error': str(e)
            }

    def process_record(self, row: pd.Series, row_index: int) -> Dict:
        # Standardized result schema expected by downstream notebooks
        result = {
            'index': row_index,
            # Use uppercase MMS_ID to align with merge step
            'MMS_ID': row.get('MMS_ID', row.get('mms_id', 'N/A')),
            'title': str(row.get('F245', 'N/A'))[:100],
            'borrowdir_id': row.get('borrowdir_id', 'N/A'),
            # Normalized flags used by reporting/merging
            'found': False,                   # normalized
            'num_items': 0,                   # normalized
            # access_type: 'full' | 'limited' | 'unknown' | None
            'access_type': None,
            'tested_identifiers': [],
            'matching_identifier': None,
            'match_type': None,               # normalized
            'error': None,
            # Backward-compatible fields kept to avoid breaking older summaries
            'found_in_hathitrust': False,
            'hathitrust_items': 0,
            'full_view_available': False,
        }
        if pd.notna(row.get('F260_str')):
            year_matches = re.findall(
                r'\b(19\d{2}|20\d{2})\b', str(row['F260_str'])
            )
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
                        result['found'] = True
                        result['found_in_hathitrust'] = True
                        result['num_items'] = response['num_items']
                        result['hathitrust_items'] = response['num_items']
                        result['matching_identifier'] = isbn_clean
                        result['match_type'] = 'ISBN'
                        # Derive access_type and full_view flag
                        access = 'unknown'
                        for item in response['items']:
                            rights = item.get('usRightsString', '')
                            if 'Full' in rights:
                                access = 'full'
                                result['full_view_available'] = True
                                break
                            elif 'Limited' in rights and access != 'full':
                                access = 'limited'
                        result['access_type'] = access
                        break
                    time.sleep(self.rate_limit_delay)
        if not result['found'] and pd.notna(row.get('F010_str')):
            lccn_str = str(row['F010_str'])
            clean_lccn = re.sub(r'[^0-9a-zA-Z]', '', lccn_str).strip()
            if clean_lccn:
                result['tested_identifiers'].append(f"LCCN:{clean_lccn}")
                response = self.query_hathitrust('lccn', clean_lccn)
                if response['found']:
                    result['found'] = True
                    result['found_in_hathitrust'] = True
                    result['num_items'] = response['num_items']
                    result['hathitrust_items'] = response['num_items']
                    result['matching_identifier'] = clean_lccn
                    result['match_type'] = 'LCCN'
                    access = 'unknown'
                    for item in response['items']:
                        rights = item.get('usRightsString', '')
                        if 'Full' in rights:
                            access = 'full'
                            result['full_view_available'] = True
                            break
                        elif 'Limited' in rights and access != 'full':
                            access = 'limited'
                    result['access_type'] = access
                time.sleep(self.rate_limit_delay)
        # Prefer direct OCLC column if present
        if not result['found'] and pd.notna(row.get('OCLC')):
            oclc_str = str(row.get('OCLC'))
            # Clean OCLC: keep digits and strip leading zeros
            oclc_clean = re.sub(r'[^0-9]', '', oclc_str).lstrip('0')
            if oclc_clean:
                result['tested_identifiers'].append(f"OCLC:{oclc_clean}")
                response = self.query_hathitrust('oclc', oclc_clean)
                if response['found']:
                    result['found'] = True
                    result['found_in_hathitrust'] = True
                    result['num_items'] = response['num_items']
                    result['hathitrust_items'] = response['num_items']
                    result['matching_identifier'] = oclc_clean
                    result['match_type'] = 'OCLC'
                    access = 'unknown'
                    for item in response['items']:
                        rights = item.get('usRightsString', '')
                        if 'Full' in rights:
                            access = 'full'
                            result['full_view_available'] = True
                            break
                        elif 'Limited' in rights and access != 'full':
                            access = 'limited'
                    result['access_type'] = access
                time.sleep(self.rate_limit_delay)
        # Legacy fallback: try parsing OCLC from id_list_str if provided
        if not result['found'] and pd.notna(row.get('id_list_str')):
            oclc_matches = re.findall(
                r'(?:OCoLC|oclc|OCLC)[:\s]*(\d+)',
                str(row['id_list_str'])
            )
            for oclc in oclc_matches[:1]:
                oclc_clean = oclc.lstrip('0')
                if oclc_clean:
                    result['tested_identifiers'].append(f"OCLC:{oclc_clean}")
                    response = self.query_hathitrust('oclc', oclc_clean)
                    if response['found']:
                        result['found'] = True
                        result['found_in_hathitrust'] = True
                        result['num_items'] = response['num_items']
                        result['hathitrust_items'] = response['num_items']
                        result['matching_identifier'] = oclc_clean
                        result['match_type'] = 'OCLC'
                        access = 'unknown'
                        for item in response['items']:
                            rights = item.get('usRightsString', '')
                            if 'Full' in rights:
                                access = 'full'
                                result['full_view_available'] = True
                                break
                            elif 'Limited' in rights and access != 'full':
                                access = 'limited'
                        result['access_type'] = access
                        break
                    time.sleep(self.rate_limit_delay)
        return result

    def scan_full_file(
        self,
        excel_path: str,
        start_from: int = 0,
        batch_size: int = 100,
    ):
        print("\n" + "="*60)
        print("HATHITRUST FULL FILE SCAN")
        print("="*60)
        print("\nLoading Excel file...")
        # Read Excel defensively to avoid dtype surprises
        df = pd.read_excel(excel_path, dtype=str)
        total_records = len(df)
        print(f"Total records: {total_records:,}")
        print(f"Starting from record: {start_from}")
        print(f"Batch size: {batch_size}")
        print(f"Max workers: {self.max_workers}")
        print(f"Rate limit delay: {self.rate_limit_delay}s")
        print("\nFiltering records with identifiers...")
        # Build mask only with columns that exist
        mask = pd.Series(False, index=df.index)
        if 'F020_str' in df.columns:
            mask = mask | df['F020_str'].notna()
        if 'F010_str' in df.columns:
            mask = mask | df['F010_str'].notna()
        if 'OCLC' in df.columns:
            mask = mask | (
                df['OCLC'].notna() &
                (df['OCLC'].astype(str).str.strip() != '')
            )
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
    # If no explicit output_dir was provided,
    # derive a dataset-specific directory
        if not self.output_dir:
            base_dir = os.path.dirname(excel_path)
            fname = os.path.basename(excel_path)
            dataset_name = os.path.splitext(fname)[0]
            if dataset_name.startswith('temp_hathitrust_'):
                dataset_name = dataset_name[len('temp_hathitrust_'):]
            reports_dir = os.path.join(
                base_dir,
                'hathitrust_reports',
                dataset_name
            )
            os.makedirs(reports_dir, exist_ok=True)
            self.output_dir = reports_dir
        else:
            os.makedirs(self.output_dir, exist_ok=True)
        est_minutes = (
            records_to_process * self.rate_limit_delay / 60
        )
        print(
            f"\nProcessing will take approximately {est_minutes:.1f} minutes"
        )
        print(
            "\nStarting scan... (Press Ctrl+C to interrupt "
            "and save progress)\n"
        )
        start_time = datetime.now()
        matches_found = 0
        full_view_found = 0
        errors = 0
        with tqdm(
            total=records_to_process,
            desc="Scanning records",
            unit="rec",
        ) as pbar:
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
                    match_rate_str = (
                        f"{matches_found/(len(self.results))*100:.1f}%"
                    )
                    pbar.set_postfix({
                        'Matches': matches_found,
                        'Full View': full_view_found,
                        'Match Rate': match_rate_str,
                    })
                if len(self.results) % (batch_size * 5) == 0:
                    self._save_intermediate_results()
        end_time = datetime.now()
        duration = end_time - start_time
        print("\n" + "="*60)
        print("SCAN COMPLETE")
        print("="*60)
        print("\nProcessing Summary:")
        print(f"  Total time: {duration}")
        print(f"  Records processed: {len(self.results):,}")
        recs_per_min = len(self.results) / (duration.total_seconds() / 60)
        matches_pct = (
            (matches_found / len(self.results) * 100)
            if self.results else 0
        )
        full_view_pct = (
            (full_view_found / len(self.results) * 100)
            if self.results else 0
        )
        print(f"  Records/minute: {recs_per_min:.1f}")
        print(
            f"  HathiTrust matches: {matches_found:,} ({matches_pct:.1f}%)"
        )
        print(
            "  Full view available: "
            f"{full_view_found:,} ({full_view_pct:.1f}%)"
        )
        print(f"  Errors: {errors}")
        self._save_final_results()

    def _save_final_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = (
            self.output_dir
            if self.output_dir
            else os.path.join('hathitrust', 'reports')
        )
        os.makedirs(reports_dir, exist_ok=True)
        results_df = pd.DataFrame(self.results)
        results_filename = os.path.join(
            reports_dir,
            f"hathitrust_scan_results_{timestamp}.csv",
        )
        results_df.to_csv(results_filename, index=False)
        print(f"\nDetailed results saved to: {results_filename}")
        self._generate_summary_report(results_df, timestamp)

    def _generate_summary_report(
        self,
        results_df: pd.DataFrame,
        timestamp: str,
    ):
        reports_dir = (
            self.output_dir
            if self.output_dir
            else os.path.join('hathitrust', 'reports')
        )
        report_filename = os.path.join(
            reports_dir,
            f"hathitrust_scan_summary_{timestamp}.txt",
        )
        with open(report_filename, 'w') as f:
            f.write("HATHITRUST SCAN SUMMARY REPORT\n")
            f.write("=" * 60 + "\n")
            ts_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"Generated: {ts_str}\n")
            f.write(f"Total records processed: {len(results_df):,}\n\n")
            # Prefer normalized columns
            if 'found' in results_df.columns:
                matches = results_df['found'].sum()
            else:
                matches = results_df.get(
                    'found_in_hathitrust',
                    pd.Series(dtype=int),
                ).sum()
            full_view = results_df.get(
                'full_view_available',
                pd.Series(dtype=int),
            ).sum()
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 30 + "\n")
            match_pct = (
                (matches / len(results_df) * 100)
                if len(results_df) else 0
            )
            full_pct = (
                (full_view / len(results_df) * 100)
                if len(results_df) else 0
            )
            f.write(
                f"HathiTrust matches: {matches:,} ({match_pct:.2f}%)\n"
            )
            f.write(
                f"Full view available: {full_view:,} ({full_pct:.2f}%)\n"
            )
            f.write(f"Limited view only: {matches - full_view:,}\n")
            f.write(f"Not in HathiTrust: {len(results_df) - matches:,}\n\n")
            f.write("MATCHES BY IDENTIFIER TYPE\n")
            f.write("-" * 30 + "\n")
            for id_type in ['ISBN', 'LCCN', 'OCLC']:
                # Use normalized 'match_type' if available
                key_col = (
                    'match_type'
                    if 'match_type' in results_df.columns
                    else 'matching_type'
                )
                type_matches = results_df[
                    results_df[key_col] == id_type
                ].shape[0]
                f.write(f"{id_type}: {type_matches:,} matches\n")
            if 'pub_year' in results_df.columns:
                f.write("\nPUBLICATION YEAR ANALYSIS\n")
                f.write("-" * 30 + "\n")
                year_df = results_df[results_df['pub_year'].notna()].copy()
                if not year_df.empty:
                    year_df['decade'] = (year_df['pub_year'] // 10) * 10
                    # Prefer normalized 'found'
                    found_col = (
                        'found'
                        if 'found' in year_df.columns
                        else 'found_in_hathitrust'
                    )
                    decade_matches = year_df.groupby('decade')[found_col].agg(
                        ['sum', 'count']
                    )
                    for decade, row in decade_matches.iterrows():
                        match_rate = row['sum'] / row['count'] * 100
                        f.write(
                            f"{int(decade)}s: {int(row['sum'])}/"
                            f"{int(row['count'])} ({match_rate:.1f}%)\n"
                        )
            f.write("\nSAMPLE OF MATCHED RECORDS\n")
            f.write("-" * 30 + "\n")
            matched_sample = (
                results_df[results_df['found']].head(10)
                if 'found' in results_df.columns
                else results_df[results_df['found_in_hathitrust']].head(10)
            )
            for _, record in matched_sample.iterrows():
                mms_val = (
                    record['MMS_ID']
                    if 'MMS_ID' in record
                    else record.get('mms_id', 'N/A')
                )
                f.write(f"\nMMS ID: {mms_val}\n")
                f.write(f"Title: {record['title']}\n")
                key_col = (
                    'match_type'
                    if 'match_type' in results_df.columns
                    else 'matching_type'
                )
                f.write(f"Match Type: {record[key_col]}\n")
                f.write(f"Identifier: {record['matching_identifier']}\n")
                full_str = (
                    'Yes' if record['full_view_available'] else 'No'
                )
                f.write(f"Full View: {full_str}\n")
        print(f"Summary report saved to: {report_filename}")

    def resume_from_checkpoint(self, checkpoint_file: str):
        print(f"\nResuming from checkpoint: {checkpoint_file}")
        checkpoint_df = pd.read_csv(checkpoint_file)
        self.results = checkpoint_df.to_dict('records')
        print(f"Loaded {len(self.results)} previous results")
        return len(self.results)


def main():
    parser = argparse.ArgumentParser(
        description='Full HathiTrust scan with progress tracking'
    )
    parser.add_argument(
        '--input-file',
        default='penn_only_holdings_text_format.xlsx',
        help='Path to the Excel file',
    )
    parser.add_argument(
        '--start-from',
        type=int,
        default=0,
        help='Record number to start from (for resuming)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of records per batch',
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='Maximum concurrent workers',
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=0.2,
        help='Delay between API calls in seconds',
    )
    parser.add_argument(
        '--resume',
        help='Resume from checkpoint CSV file',
    )
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
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
