#!/usr/bin/env python3
# =============================================================================
# Author        : Dr. Gary B. Doran
# Role          : Data Scientist
# Affiliation   : Jet Propulsion Laboratory, California Institute of Technology,
#                 Pasadena, CA 91109
# Collaboration : Pranav Durai
#                 Stanford Center for Innovation in In Vivo Imaging,
#                 Stanford University School of Medicine, Stanford, CA 94305
#
# Description : HiRISE annotation parser
#               – Extracts unique HiRISE observation IDs from annotation files
#               – Reports class distribution and imbalance statistics
#               – Writes a download list of unique HiRISE IDs
#               – Optionally generates an example download script
#               – Optionally filters out the background/"others" class
# =============================================================================

import re
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def parse_hirise_annotations(file_path: str) -> Dict:
    """
    Parse HiRISE annotation file to extract unique identifiers.
    
    Format: ESP_013049_0950_RED-0067.jpg 7
    Where ESP_013049_0950_RED is the HiRISE image ID and 0067 is the tile number
    
    Args:
        file_path: Path to the annotation text file
        
    Returns:
        Dictionary with statistics and unique identifiers
    """
    unique_hirise_ids = set()
    unique_tiles = set()
    class_distribution = {}
    total_entries = 0
    
    # Pattern to extract HiRISE ID and tile number
    # Format: ESP_XXXXXX_XXXX_RED-XXXX.jpg CLASS
    pattern = r'(ESP_\d+_\d+)_RED-(\d+)\.jpg\s+(\d+)'
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                total_entries += 1
                
                # Try to parse the line
                match = re.match(pattern, line)
                if match:
                    hirise_id = match.group(1)  # ESP_013049_0950
                    tile_num = match.group(2)    # 0067
                    class_label = int(match.group(3))  # 7
                    
                    # Add to unique sets
                    unique_hirise_ids.add(hirise_id)
                    full_tile = f"{hirise_id}_RED-{tile_num}"
                    unique_tiles.add(full_tile)
                    
                    # Track class distribution
                    if class_label not in class_distribution:
                        class_distribution[class_label] = 0
                    class_distribution[class_label] += 1
                else:
                    # Try alternative patterns
                    # Sometimes format might be slightly different
                    alt_pattern = r'(ESP_\d+_\d+)[_-]?RED[_-]?(\d+)\.jpg\s+(\d+)'
                    alt_match = re.match(alt_pattern, line)
                    if alt_match:
                        hirise_id = alt_match.group(1)
                        tile_num = alt_match.group(2)
                        class_label = int(alt_match.group(3))
                        
                        unique_hirise_ids.add(hirise_id)
                        full_tile = f"{hirise_id}_RED-{tile_num}"
                        unique_tiles.add(full_tile)
                        
                        if class_label not in class_distribution:
                            class_distribution[class_label] = 0
                        class_distribution[class_label] += 1
                    else:
                        logger.warning(f"Line {line_num}: Could not parse: {line[:50]}...")
        
        return {
            'total_entries': total_entries,
            'unique_hirise_ids': sorted(list(unique_hirise_ids)),
            'unique_tiles': sorted(list(unique_tiles)),
            'class_distribution': class_distribution,
            'num_unique_images': len(unique_hirise_ids),
            'num_unique_tiles': len(unique_tiles)
        }
    
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error parsing file: {str(e)}")
        return None


def save_hirise_ids(hirise_ids: List[str], output_file: str = "hirise_download_list.txt"):
    """
    Save unique HiRISE IDs to a file for downloading.
    
    Args:
        hirise_ids: List of unique HiRISE identifiers
        output_file: Output filename
    """
    with open(output_file, 'w') as f:
        for hirise_id in hirise_ids:
            # Write in format ready for HiRISE download
            # Full URL would be like: https://hirise.lpl.arizona.edu/PDS/EDR/ESP/ORB_013000_013099/ESP_013049_0950/ESP_013049_0950_RED.JP2
            f.write(f"{hirise_id}_RED\n")
    
    logger.info(f"Saved {len(hirise_ids)} unique HiRISE IDs to {output_file}")


def print_analysis(results: Dict):
    """
    Print detailed analysis of the parsed data.
    
    Args:
        results: Dictionary with parsing results
    """
    logger.info("\n" + "=" * 60)
    logger.info("HiRISE ANNOTATION FILE ANALYSIS")
    logger.info("=" * 60)
    
    logger.info(f"\nTotal entries in file: {results['total_entries']:,}")
    logger.info(f"Unique HiRISE images: {results['num_unique_images']:,}")
    logger.info(f"Unique tiles: {results['num_unique_tiles']:,}")
    logger.info(f"Average tiles per image: {results['num_unique_tiles'] / results['num_unique_images']:.1f}")
    
    # Class distribution
    logger.info("\nClass Distribution:")
    logger.info("-" * 40)
    
    # Map class numbers to names based on provided mapping
    class_names = {
        0: 'others (ignored)',
        1: 'crater',
        2: 'dark_dune', 
        3: 'slope_streak',
        4: 'bright_dune',
        5: 'impact_ejecta',
        6: 'swiss_cheese',
        7: 'spider'
    }
    
    total_samples = sum(results['class_distribution'].values())
    relevant_samples = sum(count for class_id, count in results['class_distribution'].items() if class_id != 0)
    
    # First show overall statistics
    if 0 in results['class_distribution']:
        others_count = results['class_distribution'][0]
        others_pct = (others_count / total_samples) * 100
        logger.info(f"\nClass 0 (others - excluded): {others_count:,} samples ({others_pct:.2f}%)")
        logger.info(f"Relevant samples (classes 1-7): {relevant_samples:,} samples\n")
    
    # Show distribution for relevant classes
    logger.info("Relevant Class Distribution (classes 1-7):")
    for class_id in sorted(results['class_distribution'].keys()):
        if class_id == 0:  # Skip class 0
            continue
        count = results['class_distribution'][class_id]
        percentage_total = (count / total_samples) * 100
        percentage_relevant = (count / relevant_samples) * 100 if relevant_samples > 0 else 0
        class_name = class_names.get(class_id, f"Class_{class_id}")
        logger.info(f"  {class_name:15} (class {class_id}): {count:6,} samples ({percentage_relevant:5.2f}% of relevant, {percentage_total:5.2f}% of total)")
    
    # Check class balance for relevant classes only
    relevant_counts = [count for class_id, count in results['class_distribution'].items() if class_id != 0]
    if relevant_counts:
        min_count = min(relevant_counts)
        max_count = max(relevant_counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        logger.info("\nClass Balance Analysis (excluding class 0):")
        logger.info(f"  Min samples: {min_count:,}")
        logger.info(f"  Max samples: {max_count:,}")
        logger.info(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 3:
            logger.warning("  High class imbalance detected! Consider data augmentation or weighted sampling.")
    
    # Sample of HiRISE IDs
    logger.info("\nSample HiRISE IDs (first 10):")
    logger.info("-" * 40)
    for i, hirise_id in enumerate(results['unique_hirise_ids'][:10], 1):
        logger.info(f"  {i:2}. {hirise_id}")
    
    if results['num_unique_images'] > 10:
        logger.info(f"  ... and {results['num_unique_images'] - 10:,} more")
    
    # Download information
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD INFORMATION")
    logger.info("=" * 60)
    logger.info("\nTo download these HiRISE images:")
    logger.info("Each ID corresponds to a HiRISE observation")
    logger.info("Files can be downloaded from: https://www.uahirise.org/")
    logger.info("Format: ESP_XXXXXX_XXXX_RED.JP2 (full resolution)")
    logger.info("Alternative: Use HiRISE PDS for systematic downloads")
    logger.info(f"\nEstimated download size (at ~1-2GB per image): {results['num_unique_images'] * 1.5:.1f} GB")


def create_download_script(hirise_ids: List[str], output_file: str = "download_hirise.sh"):
    """
    Create a bash script for downloading HiRISE images.
    
    Args:
        hirise_ids: List of HiRISE identifiers
        output_file: Output script filename
    """
    with open(output_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Script to download HiRISE images\n")
        f.write("# Note: URLs may need adjustment based on actual HiRISE repository structure\n\n")
        
        f.write("mkdir -p hirise_images\n")
        f.write("cd hirise_images\n\n")
        
        for hirise_id in hirise_ids[:10]:  # Limit to first 10 for example
            # Parse the ID to get orbit range
            parts = hirise_id.split('_')
            if len(parts) >= 3:
                orbit_num = int(parts[1])
                orbit_range_start = (orbit_num // 100) * 100
                orbit_range_end = orbit_range_start + 99
                orbit_dir = f"ORB_{orbit_range_start:06d}_{orbit_range_end:06d}"
                
                # Construct URL (this is approximate - actual URL structure may vary)
                url = f"https://hirise.lpl.arizona.edu/PDS/EDR/ESP/{orbit_dir}/{hirise_id}/{hirise_id}_RED.JP2"
                f.write(f"# Download {hirise_id}\n")
                f.write(f"wget -c '{url}' || echo 'Failed: {hirise_id}'\n\n")
        
        f.write("echo 'Download complete!'\n")
    
    import os
    os.chmod(output_file, 0o755)
    logger.info(f"Created download script: {output_file} (limited to first 10 images as example)")


def save_filtered_annotations(file_path: str, output_file: str = "filtered_annotations.txt"):
    """
    Save annotations excluding class 0 (others) to a new file.
    
    Args:
        file_path: Path to the original annotation file
        output_file: Output filename for filtered annotations
    """
    pattern = r'(ESP_\d+_\d+)_RED-(\d+)\.jpg\s+(\d+)'
    filtered_count = 0
    excluded_count = 0
    
    with open(file_path, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
                
            match = re.match(pattern, line)
            if match:
                class_label = int(match.group(3))
                if class_label != 0:  # Exclude class 0
                    f_out.write(line + '\n')
                    filtered_count += 1
                else:
                    excluded_count += 1
    
    logger.info(f"Created filtered annotation file: {output_file}")
    logger.info(f"   Kept: {filtered_count:,} samples (classes 1-7)")
    logger.info(f"   Excluded: {excluded_count:,} samples (class 0)")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Parse HiRISE annotation file to extract unique image IDs"
    )
    parser.add_argument(
        'file',
        type=str,
        help='Path to the annotation text file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='hirise_download_list.txt',
        help='Output file for HiRISE IDs'
    )
    parser.add_argument(
        '--create-script',
        action='store_true',
        help='Create a download script'
    )
    parser.add_argument(
        '--filter-annotations',
        action='store_true',
        help='Create filtered annotation file excluding class 0'
    )
    
    args = parser.parse_args()
    
    # Parse the file
    logger.info(f"Parsing file: {args.file}")
    results = parse_hirise_annotations(args.file)
    
    if results:
        # Print analysis
        print_analysis(results)
        
        # Save HiRISE IDs
        save_hirise_ids(results['unique_hirise_ids'], args.output)
        
        # Optionally create download script
        if args.create_script:
            create_download_script(results['unique_hirise_ids'], 'download_hirise.sh')
        
        # Optionally create filtered annotations
        if args.filter_annotations:
            save_filtered_annotations(args.file, 'filtered_annotations.txt')
        
        logger.info(f"\nSummary: {results['num_unique_images']} unique HiRISE images to download")
    else:
        logger.error("Failed to parse annotation file")


if __name__ == "__main__":
    main()