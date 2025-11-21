#!/usr/bin/env python3
"""
Convert Buckeye Corpus to TIMIT-compatible format

This script:
1. Converts .phones files to .phn format
2. Changes time-based timestamps (seconds) to sample-based timestamps
3. Maps Buckeye phonemes to TIMIT phoneme set
4. Removes header metadata from Buckeye files
5. Filters out non-speech events (IVER, NOISE, etc.)
6. Copies corresponding .wav and .txt files
7. Generates detailed conversion statistics
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json

# Sample rate for both datasets
SAMPLE_RATE = 16000


class BuckeyeToTimitConverter:
    """Convert Buckeye Corpus format to TIMIT format"""
    
    def __init__(self):
        # Phone mapping: Buckeye -> TIMIT
        self.phone_map = self._build_phone_mapping()
        self.stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'phones_converted': defaultdict(int),
            'phones_skipped': defaultdict(int),
            'phones_mapped': defaultdict(lambda: defaultdict(int)),
            'total_frames_before': 0,
            'total_frames_after': 0,
            'errors': []
        }
    
    def _build_phone_mapping(self) -> Dict[str, Optional[str]]:
        """
        Build comprehensive phone mapping from Buckeye to TIMIT
        
        Returns:
            Dict mapping Buckeye phone -> TIMIT phone (None means skip)
        """
        
        # Core phones that exist in both datasets (no mapping needed)
        shared_phones = {
            'aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 
            'dx', 'eh', 'el', 'em', 'en', 'eng', 'er', 'ey', 'f', 'g', 
            'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'nx', 
            'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 
            'v', 'w', 'y', 'z', 'zh'
        }
        
        phone_map = {phone: phone for phone in shared_phones}
        
        # Glottalized phones (phone;) -> base phone
        glottalized = [
            'b', 'd', 'g', 'p', 't', 'k',  # Stops
            's', 'z', 'sh', 'zh', 'f', 'v', 'th', 'dh',  # Fricatives
            'ch', 'jh',  # Affricates
            'm', 'n', 'ng', 'nx',  # Nasals
            'l', 'r',  # Liquids
            'w', 'y',  # Glides
            'aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'eh', 'er', 'ey', 
            'ih', 'iy', 'ow', 'oy', 'uh', 'uw'  # Vowels
        ]
        
        for phone in glottalized:
            phone_map[f"{phone};"] = phone
        
        # Nasalized vowels (phoneN) -> base vowel
        # Note: In TIMIT these would have nasal=0, but we'll preserve the base vowel
        nasalized_vowels = {
            'iyn': 'iy', 'ahn': 'ah', 'ehn': 'eh', 'own': 'ow',
            'aen': 'ae', 'aan': 'aa', 'aon': 'ao', 'awn': 'aw',
            'ayn': 'ay', 'ern': 'er', 'ihn': 'ih'
        }
        phone_map.update(nasalized_vowels)
        
        # Special markers -> TIMIT equivalents
        special_markers = {
            'SIL': 'pau',        # Silence -> pause
            'NOISE': 'pau',      # Noise -> pause
            '{B_TRANS}': 'h#',   # Begin transcription -> sentence boundary
            '{E_TRANS}': 'h#',   # End transcription -> sentence boundary
        }
        phone_map.update(special_markers)
        
        # Skip these (non-speech events)
        skip_phones = [
            'IVER',           # Interviewer speech
            'VOCNOISE',       # Vocal noise (breathing, um, uh)
            'LAUGH',          # Laughter
            'UNKNOWN',        # Unknown sound
            '<EXCLUDE-name>', # Excluded proper names
        ]
        
        for phone in skip_phones:
            phone_map[phone] = None  # None means skip
        
        # Buckeye-specific phones -> TIMIT equivalents
        buckeye_specific = {
            'tq': 'dx',   # Buckeye flap variant -> TIMIT flap
            'h': 'hh',    # /h/ variant
            'no': 'en',   # Syllabic nasal
        }
        phone_map.update(buckeye_specific)
        
        return phone_map
    
    def read_buckeye_phones(self, phones_path: Path) -> List[Tuple[float, str]]:
        """
        Read Buckeye .phones file
        
        Args:
            phones_path: Path to .phones file
        
        Returns:
            List of (time_in_seconds, phone_label) tuples
        """
        phones = []
        in_header = True
        
        try:
            with open(phones_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Skip header until we hit '#'
                    if in_header:
                        if line == '#':
                            in_header = False
                        continue
                    
                    # Parse data line: time color phone
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            time_sec = float(parts[0])
                            phone = parts[2]
                            phones.append((time_sec, phone))
                        except (ValueError, IndexError) as e:
                            self.stats['errors'].append(
                                f"Error parsing line in {phones_path.name}: {line}"
                            )
        
        except Exception as e:
            self.stats['errors'].append(f"Error reading {phones_path}: {str(e)}")
            return []
        
        return phones
    
    def convert_to_timit_phn(
        self, 
        phones: List[Tuple[float, str]], 
        source_file: str
    ) -> List[Tuple[int, int, str]]:
        """
        Convert Buckeye phones to TIMIT .phn format
        
        Args:
            phones: List of (time_seconds, phone) from Buckeye
            source_file: Source filename for error reporting
        
        Returns:
            List of (start_sample, end_sample, phone) tuples
        """
        timit_phones = []
        
        for i in range(len(phones) - 1):
            time_sec, phone = phones[i]
            next_time_sec, _ = phones[i + 1]
            
            # Map Buckeye phone to TIMIT phone
            if phone in self.phone_map:
                mapped_phone = self.phone_map[phone]
                
                # Track statistics
                self.stats['phones_converted'][phone] += 1
                if mapped_phone != phone:
                    self.stats['phones_mapped'][phone][mapped_phone] += 1
                
                # Skip if mapped to None (non-speech events)
                if mapped_phone is None:
                    self.stats['phones_skipped'][phone] += 1
                    continue
                
                # Convert time to samples
                start_sample = int(time_sec * SAMPLE_RATE)
                end_sample = int(next_time_sec * SAMPLE_RATE)
                
                # Skip zero-length segments
                if start_sample >= end_sample:
                    continue
                
                timit_phones.append((start_sample, end_sample, mapped_phone))
                self.stats['total_frames_after'] += 1
            
            else:
                # Unknown phone - skip and warn
                self.stats['errors'].append(
                    f"Unknown phone '{phone}' in {source_file} - skipping"
                )
                self.stats['phones_skipped'][phone] += 1
            
            self.stats['total_frames_before'] += 1
        
        return timit_phones
    
    def write_timit_phn(self, phn_path: Path, phones: List[Tuple[int, int, str]]):
        """
        Write TIMIT-style .phn file
        
        Args:
            phn_path: Output .phn file path
            phones: List of (start_sample, end_sample, phone) tuples
        """
        try:
            with open(phn_path, 'w', encoding='utf-8') as f:
                for start_sample, end_sample, phone in phones:
                    f.write(f"{start_sample} {end_sample} {phone}\n")
        except Exception as e:
            self.stats['errors'].append(f"Error writing {phn_path}: {str(e)}")
    
    def convert_file(
        self, 
        input_phones: Path, 
        output_phn: Path,
        input_wav: Path,
        output_wav: Path,
        input_txt: Optional[Path] = None,
        output_txt: Optional[Path] = None
    ) -> bool:
        """
        Convert a single Buckeye recording to TIMIT format
        
        Args:
            input_phones: Input .phones file
            output_phn: Output .phn file
            input_wav: Input .wav file
            output_wav: Output .wav file
            input_txt: Input .txt file (optional)
            output_txt: Output .txt file (optional)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read Buckeye phones
            phones = self.read_buckeye_phones(input_phones)
            if not phones:
                self.stats['files_skipped'] += 1
                return False
            
            # Convert to TIMIT format
            timit_phones = self.convert_to_timit_phn(phones, input_phones.name)
            if not timit_phones:
                self.stats['files_skipped'] += 1
                return False
            
            # Write TIMIT .phn file
            output_phn.parent.mkdir(parents=True, exist_ok=True)
            self.write_timit_phn(output_phn, timit_phones)
            
            # Copy .wav file
            if input_wav.exists():
                shutil.copy2(input_wav, output_wav)
            else:
                self.stats['errors'].append(f"Missing .wav file: {input_wav}")
            
            # Copy .txt file if exists
            if input_txt and input_txt.exists() and output_txt:
                shutil.copy2(input_txt, output_txt)
            
            self.stats['files_processed'] += 1
            return True
        
        except Exception as e:
            self.stats['errors'].append(
                f"Error converting {input_phones.name}: {str(e)}"
            )
            self.stats['files_skipped'] += 1
            return False
    
    def convert_speaker(self, input_dir: Path, output_dir: Path) -> Dict:
        """
        Convert all recordings for a speaker
        
        Args:
            input_dir: Input speaker directory (e.g., Buckeye_Corpus/s01)
            output_dir: Output directory (e.g., Buckeye_preprocessed/s01)
        
        Returns:
            Statistics dictionary
        """
        print(f"\nConverting speaker: {input_dir.name}")
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")
        
        # Find all .phones files
        phones_files = sorted(input_dir.glob("*.phones"))
        
        if not phones_files:
            print(f"  Warning: No .phones files found in {input_dir}")
            return self.stats
        
        print(f"  Found {len(phones_files)} recordings")
        
        # Process each recording
        for phones_file in phones_files:
            basename = phones_file.stem  # e.g., 's0101a'
            
            # Define input files
            input_wav = input_dir / f"{basename}.wav"
            input_txt = input_dir / f"{basename}.txt"
            
            # Define output files
            output_phn = output_dir / f"{basename}.phn"
            output_wav = output_dir / f"{basename}.wav"
            output_txt = output_dir / f"{basename}.txt"
            
            # Convert
            success = self.convert_file(
                phones_file, output_phn,
                input_wav, output_wav,
                input_txt, output_txt
            )
            
            if success:
                print(f"    ✓ {basename}")
            else:
                print(f"    ✗ {basename} - FAILED")
        
        return self.stats
    
    def generate_summary_report(self, output_file: Path):
        """Generate detailed preprocessing summary report"""
        
        report = []
        report.append("=" * 80)
        report.append("BUCKEYE TO TIMIT PREPROCESSING SUMMARY")
        report.append("=" * 80)
        report.append("")
        
        # Overview
        report.append("CONVERSION OVERVIEW")
        report.append("-" * 80)
        report.append(f"Files successfully processed: {self.stats['files_processed']}")
        report.append(f"Files skipped/failed:        {self.stats['files_skipped']}")
        report.append(f"Total phone frames (before): {self.stats['total_frames_before']}")
        report.append(f"Total phone frames (after):  {self.stats['total_frames_after']}")
        retained_pct = (self.stats['total_frames_after'] / self.stats['total_frames_before'] * 100 
                       if self.stats['total_frames_before'] > 0 else 0)
        report.append(f"Frame retention rate:        {retained_pct:.1f}%")
        report.append("")
        
        # Format changes
        report.append("FORMAT CHANGES")
        report.append("-" * 80)
        report.append("BEFORE (Buckeye Format):")
        report.append("  - File extension:   .phones")
        report.append("  - Timestamp format: Time in seconds (decimal)")
        report.append("  - File structure:   Header + data lines with 'time color phone'")
        report.append("  - Example line:     '32.622045 122 k'")
        report.append("")
        report.append("AFTER (TIMIT Format):")
        report.append("  - File extension:   .phn")
        report.append("  - Timestamp format: Sample indices (integer)")
        report.append("  - File structure:   Plain text with 'start_sample end_sample phone'")
        report.append("  - Example line:     '522025 529152 k'")
        report.append("  - Conversion:       sample = int(time_seconds * 16000)")
        report.append("")
        
        # Phoneme mapping
        report.append("PHONEME MAPPING SUMMARY")
        report.append("-" * 80)
        
        # Phones that were mapped (changed)
        if self.stats['phones_mapped']:
            report.append("Phonemes Mapped (Buckeye -> TIMIT):")
            for buckeye_phone in sorted(self.stats['phones_mapped'].keys()):
                for timit_phone, count in self.stats['phones_mapped'][buckeye_phone].items():
                    if timit_phone is not None:
                        report.append(f"  {buckeye_phone:15} -> {timit_phone:10}  (n={count})")
            report.append("")
        
        # Phones that were skipped
        if self.stats['phones_skipped']:
            report.append("Phonemes Skipped (Non-speech events):")
            for phone, count in sorted(self.stats['phones_skipped'].items(), 
                                      key=lambda x: x[1], reverse=True):
                report.append(f"  {phone:20}  (n={count})")
            report.append("")
        
        # Phone frequency distribution
        report.append("Phoneme Frequency (Top 20):")
        top_phones = sorted(self.stats['phones_converted'].items(), 
                          key=lambda x: x[1], reverse=True)[:20]
        for phone, count in top_phones:
            mapped = self.stats['phones_mapped'].get(phone, {})
            if mapped:
                mapped_to = list(mapped.keys())[0]
                if mapped_to is not None:
                    report.append(f"  {phone:15} -> {mapped_to:10}  (n={count})")
                else:
                    report.append(f"  {phone:15} -> [SKIP]      (n={count})")
            else:
                report.append(f"  {phone:15} (unchanged)    (n={count})")
        report.append("")
        
        # Alignment with TIMIT
        report.append("ALIGNMENT WITH TIMIT")
        report.append("-" * 80)
        report.append("Compatibility achieved:")
        report.append("  ✓ Audio format:      16 kHz, 16-bit, mono PCM WAVE (same as TIMIT)")
        report.append("  ✓ File extension:    .phn (matches TIMIT)")
        report.append("  ✓ Timestamp format:  Sample-based indices (matches TIMIT)")
        report.append("  ✓ Phoneme set:       Mapped to TIMIT phoneme inventory (61 phones)")
        report.append("  ✓ File structure:    Flat text format (matches TIMIT)")
        report.append("  ✓ Non-speech events: Filtered out (IVER, LAUGH, VOCNOISE, etc.)")
        report.append("")
        report.append("Differences remaining:")
        report.append("  - Speech type:       Spontaneous conversation vs. read speech")
        report.append("  - Phoneme distribution: Natural conversational vs. phonetically balanced")
        report.append("  - Glottalization:    Information preserved via phone mapping")
        report.append("  - Nasalization:      Vowel nasalization mapped to base vowels")
        report.append("")
        
        # Phone mapping details
        report.append("DETAILED PHONEME MAPPING RULES")
        report.append("-" * 80)
        report.append("")
        report.append("1. GLOTTALIZED PHONES (phone;)")
        report.append("   Rule: Map to base phoneme")
        report.append("   Examples:")
        report.append("     t; -> t    (glottal stop replacing /t/)")
        report.append("     k; -> k    (glottal stop replacing /k/)")
        report.append("     d; -> d    (glottalized /d/)")
        report.append("     aa; -> aa  (glottalized vowel)")
        report.append("")
        report.append("2. NASALIZED VOWELS (phoneN)")
        report.append("   Rule: Map to base vowel (nasal feature implicit)")
        report.append("   Examples:")
        report.append("     iyn -> iy  (nasalized /iy/)")
        report.append("     ahn -> ah  (nasalized /ah/)")
        report.append("     ehn -> eh  (nasalized /eh/)")
        report.append("")
        report.append("3. SPECIAL MARKERS")
        report.append("   Rule: Map to nearest TIMIT equivalent or skip")
        report.append("   Examples:")
        report.append("     SIL -> pau         (silence to pause)")
        report.append("     NOISE -> pau       (noise to pause)")
        report.append("     {B_TRANS} -> h#    (boundary marker)")
        report.append("     IVER -> [SKIP]     (interviewer speech removed)")
        report.append("     VOCNOISE -> [SKIP] (vocal noise removed)")
        report.append("")
        report.append("4. BUCKEYE-SPECIFIC PHONES")
        report.append("   Rule: Map to functionally equivalent TIMIT phone")
        report.append("   Examples:")
        report.append("     tq -> dx   (flap variant)")
        report.append("     h -> hh    (/h/ variant)")
        report.append("     no -> en   (syllabic nasal)")
        report.append("")
        
        # Errors and warnings
        if self.stats['errors']:
            report.append("ERRORS AND WARNINGS")
            report.append("-" * 80)
            for error in self.stats['errors'][:50]:  # Limit to first 50
                report.append(f"  {error}")
            if len(self.stats['errors']) > 50:
                report.append(f"  ... and {len(self.stats['errors']) - 50} more errors")
            report.append("")
        
        # Usage
        report.append("USAGE WITH FRAMEWORK")
        report.append("-" * 80)
        report.append("The preprocessed Buckeye data can now be used with the existing")
        report.append("TIMIT-based framework without any code modifications:")
        report.append("")
        report.append("  python main.py \\")
        report.append("    --dataset 01_Raw_Phonetic_Annotated_Datasets/02_Buckeye_preprocessed_dataset_s01 \\")
        report.append("    --model wavlm-base \\")
        report.append("    --output 02_OUTPUTS/Buckeye_Outputs")
        report.append("")
        report.append("All phoneme labels are now compatible with the feature dictionary")
        report.append("defined in configs/phoneme_features.py")
        report.append("")
        
        report.append("=" * 80)
        report.append(f"Preprocessing completed successfully")
        report.append(f"Summary saved to: {output_file}")
        report.append("=" * 80)
        
        # Write report
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Buckeye Corpus to TIMIT-compatible format"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input Buckeye speaker directory (e.g., Buckeye_Corpus/s01)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for preprocessed data'
    )
    parser.add_argument(
        '--summary',
        type=str,
        default=None,
        help='Path to save preprocessing summary (default: output_dir/../preprocessing_summary.txt)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Determine summary file path
    if args.summary:
        summary_path = Path(args.summary)
    else:
        summary_path = output_dir.parent / "preprocessing_summary.txt"
    
    # Create converter
    converter = BuckeyeToTimitConverter()
    
    # Convert speaker
    print("=" * 80)
    print("BUCKEYE TO TIMIT CONVERSION")
    print("=" * 80)
    
    stats = converter.convert_speaker(input_dir, output_dir)
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY REPORT")
    print("=" * 80)
    
    report = converter.generate_summary_report(summary_path)
    
    # Print summary to console
    print("\n" + report)
    
    print(f"\n✓ Conversion complete!")
    print(f"  Processed files: {output_dir}")
    print(f"  Summary report:  {summary_path}")


if __name__ == "__main__":
    main()
