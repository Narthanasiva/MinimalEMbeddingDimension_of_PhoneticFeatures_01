#!/usr/bin/env python3
"""
Batch convert all Buckeye Corpus speakers to TIMIT format
Processes all 40 speakers (s01 to s40)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import subprocess

def main():
    base_dir = Path("/home/narthana/MinimalEMbeddingDimension_of_PhoneticFeatures_01")
    input_base = base_dir / "01_Raw_Phonetic_Annotated_Datasets/Buckeye_Corpus_RAW"
    output_base = base_dir / "01_Raw_Phonetic_Annotated_Datasets/02_Buckeye_preprocessed_dataset"
    script_path = base_dir / "scripts/convert_buckeye_to_timit.py"
    
    # Create output directory
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Find all speaker directories
    speaker_dirs = sorted([d for d in input_base.iterdir() if d.is_dir() and d.name.startswith('s')])
    
    print("=" * 80)
    print(f"BATCH CONVERSION: ALL BUCKEYE SPEAKERS TO TIMIT FORMAT")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total speakers to process: {len(speaker_dirs)}")
    print(f"Input directory: {input_base}")
    print(f"Output directory: {output_base}")
    print("=" * 80)
    print()
    
    # Track statistics
    successful = []
    failed = []
    
    # Process each speaker
    for i, speaker_dir in enumerate(speaker_dirs, 1):
        speaker_num = speaker_dir.name[1:]  # Extract number from 's01' -> '01'
        output_dir = output_base / f"speaker{speaker_num}"
        summary_file = output_base / f"preprocessing_summary_speaker{speaker_num}.txt"
        
        print(f"[{i}/{len(speaker_dirs)}] Processing {speaker_dir.name}...")
        print(f"  Input:  {speaker_dir}")
        print(f"  Output: {output_dir}")
        
        try:
            # Run conversion script
            cmd = [
                sys.executable,
                str(script_path),
                "--input", str(speaker_dir),
                "--output", str(output_dir),
                "--summary", str(summary_file)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per speaker
            )
            
            if result.returncode == 0:
                print(f"  ✓ SUCCESS")
                successful.append(speaker_dir.name)
            else:
                print(f"  ✗ FAILED")
                print(f"  Error: {result.stderr[:200]}")
                failed.append(speaker_dir.name)
        
        except subprocess.TimeoutExpired:
            print(f"  ✗ TIMEOUT (exceeded 5 minutes)")
            failed.append(speaker_dir.name)
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            failed.append(speaker_dir.name)
        
        print()
    
    # Generate master summary
    print("=" * 80)
    print("BATCH CONVERSION COMPLETE")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total speakers processed: {len(speaker_dirs)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print()
    
    if successful:
        print("✓ Successfully converted speakers:")
        for speaker in successful:
            print(f"  - {speaker}")
        print()
    
    if failed:
        print("✗ Failed speakers:")
        for speaker in failed:
            print(f"  - {speaker}")
        print()
    
    # Create master summary file
    master_summary = output_base / "MASTER_SUMMARY.txt"
    with open(master_summary, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BUCKEYE CORPUS - BATCH PREPROCESSING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Speakers: {len(speaker_dirs)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n")
        f.write(f"Success Rate: {len(successful)/len(speaker_dirs)*100:.1f}%\n\n")
        
        f.write("INPUT DIRECTORY:\n")
        f.write(f"  {input_base}\n\n")
        
        f.write("OUTPUT STRUCTURE:\n")
        f.write(f"  {output_base}/\n")
        f.write(f"  ├── speaker01/  (s01 converted)\n")
        f.write(f"  ├── speaker02/  (s02 converted)\n")
        f.write(f"  ├── ...\n")
        f.write(f"  └── speaker40/  (s40 converted)\n\n")
        
        if successful:
            f.write("SUCCESSFULLY CONVERTED SPEAKERS:\n")
            for speaker in successful:
                f.write(f"  ✓ {speaker}\n")
            f.write("\n")
        
        if failed:
            f.write("FAILED SPEAKERS:\n")
            for speaker in failed:
                f.write(f"  ✗ {speaker}\n")
            f.write("\n")
        
        f.write("INDIVIDUAL SUMMARIES:\n")
        f.write("  Each speaker has a detailed summary file:\n")
        f.write("  - preprocessing_summary_speaker01.txt\n")
        f.write("  - preprocessing_summary_speaker02.txt\n")
        f.write("  - ...\n")
        f.write("  - preprocessing_summary_speaker40.txt\n\n")
        
        f.write("USAGE:\n")
        f.write("  To use all preprocessed speakers together:\n\n")
        f.write("  python main.py \\\n")
        f.write("    --dataset 01_Raw_Phonetic_Annotated_Datasets/02_Buckeye_preprocessed_dataset \\\n")
        f.write("    --model wavlm-base \\\n")
        f.write("    --output 02_OUTPUTS/Buckeye_Outputs_All\n\n")
        f.write("  To use a specific speaker:\n\n")
        f.write("  python main.py \\\n")
        f.write("    --dataset 01_Raw_Phonetic_Annotated_Datasets/02_Buckeye_preprocessed_dataset/speaker01 \\\n")
        f.write("    --model wavlm-base \\\n")
        f.write("    --output 02_OUTPUTS/Buckeye_Outputs_s01\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"Master summary saved to: {master_summary}")
    print("=" * 80)
    
    return 0 if len(failed) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
