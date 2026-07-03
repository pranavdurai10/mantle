#!/usr/bin/env python
# =============================================================================
# Author        : Dr. Gary B. Doran
# Role          : Data Scientist
# Affiliation   : Jet Propulsion Laboratory, California Institute of Technology,
#                 Pasadena, CA 91109
# Collaboration : Pranav Durai
#                 Stanford Center for Innovation in In Vivo Imaging,
#                 Stanford University School of Medicine, Stanford, CA 94305
#
# Description : MSL Mastcam PDS archive downloader
#               – Auto-detects or accepts a range of MSL volume IDs
#               – Downloads RDRINDEX.TAB/.LBL index files for each volume
#               – Parses the index to extract PATH_NAME for each image
#               – Downloads IMG files and converts them to PNG
#               – Skips existing files to allow resuming interrupted downloads
# =============================================================================
import os
import pdr
import click
import requests
import tempfile
import numpy as np
from PIL import Image
from tqdm import tqdm


BASE_URL = 'https://planetarydata.jpl.nasa.gov/img/data/msl'
INDEX_PATH = '{base_url}/{volume_id}/INDEX/RDRINDEX.TAB'
LABEL_PATH = '{base_url}/{volume_id}/INDEX/RDRINDEX.LBL'


def fetch_index(volume_id):
    """
    Fetch and parse the RDRINDEX for a given volume.

    Returns:
        DataFrame from pdr.read() or None if volume doesn't exist
    """
    index_url = INDEX_PATH.format(base_url=BASE_URL, volume_id=volume_id)
    label_url = LABEL_PATH.format(base_url=BASE_URL, volume_id=volume_id)

    try:
        # Download label file first
        label_resp = requests.get(label_url, timeout=30)
        label_resp.raise_for_status()

        # Download index file
        index_resp = requests.get(index_url, timeout=30)
        index_resp.raise_for_status()

        # Parse using pdr
        with tempfile.TemporaryDirectory() as tmpdir:
            lbl_path = os.path.join(tmpdir, 'RDRINDEX.LBL')
            tab_path = os.path.join(tmpdir, 'RDRINDEX.TAB')

            with open(lbl_path, 'wb') as f:
                f.write(label_resp.content)
            with open(tab_path, 'wb') as f:
                f.write(index_resp.content)

            data = pdr.read(lbl_path)
            return data['INDEX_TABLE']

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return None
        raise
    except Exception as e:
        click.echo(f"Error fetching index for {volume_id}: {e}", err=True)
        return None


def discover_volumes():
    """
    Auto-detect all available volumes by probing the archive.

    Returns:
        List of volume IDs (e.g., ['MSLMST_0001', 'MSLMST_0002', ...])
    """
    volumes = []
    volume_num = 1
    consecutive_failures = 0
    max_consecutive_failures = 5  # Stop after 5 consecutive 404s

    click.echo("Auto-detecting volumes...")

    while consecutive_failures < max_consecutive_failures:
        volume_id = f'MSLMST_{volume_num:04d}'

        # Just check if the label file exists (smaller download)
        label_url = LABEL_PATH.format(base_url=BASE_URL, volume_id=volume_id)

        try:
            resp = requests.head(label_url, timeout=10)
            if resp.status_code == 200:
                volumes.append(volume_id)
                consecutive_failures = 0
                click.echo(f"  Found: {volume_id}")
            else:
                consecutive_failures += 1
        except:
            consecutive_failures += 1

        volume_num += 1

    return volumes


def download_img_to_png(output_path, img_url):
    """
    Download an IMG file and convert it to PNG.

    Args:
        output_path: Path to save the PNG file
        img_url: URL of the IMG file
    """
    # Download IMG file
    img_resp = requests.get(img_url, timeout=60)
    img_resp.raise_for_status()
    img_bytes = img_resp.content

    # Download corresponding LBL file
    lbl_url = img_url.replace('.IMG', '.LBL')
    lbl_resp = requests.get(lbl_url, timeout=60)
    lbl_resp.raise_for_status()
    lbl_bytes = lbl_resp.content

    # Parse and convert
    base = os.path.splitext(os.path.basename(output_path))[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        lbl_path = os.path.join(tmpdir, f'{base}.LBL')
        img_path = os.path.join(tmpdir, f'{base}.IMG')

        with open(lbl_path, 'wb') as f:
            f.write(lbl_bytes)
        with open(img_path, 'wb') as f:
            f.write(img_bytes)

        data = pdr.read(lbl_path)
        array = np.squeeze(np.asarray(data["IMAGE"]))

        # If bands-first (e.g., RGB as 3xHxW instead of HxWx3)
        if array.ndim == 3 and array.shape[0] == 3:
            array = np.transpose(array, (1, 2, 0))

        # Convert to 8-bit if necessary (check using dtype.kind and itemsize)
        # dtype.kind: 'u' = unsigned int, 'i' = signed int
        # dtype.itemsize: number of bytes (2 = 16-bit, 4 = 32-bit)
        if array.dtype.kind in ['u', 'i'] and array.dtype.itemsize > 1:
            # Try to get the valid bit depth from metadata
            valid_bits = 16  # default to full 16-bit

            try:
                # Look for SAMPLE_BIT_MASK in the IMAGE metadata
                img_meta = data.metaget_('IMAGE')
                if 'SAMPLE_BIT_MASK' in img_meta:
                    bit_mask_str = str(img_meta['SAMPLE_BIT_MASK'])
                    # Parse binary notation like "2#0000111111111111#"
                    if bit_mask_str.startswith('2#') and bit_mask_str.endswith('#'):
                        binary_str = bit_mask_str[2:-1]
                        # Count the number of 1s in the mask
                        valid_bits = binary_str.count('1')
            except:
                pass  # Fall back to 16-bit if we can't read metadata

            # Scale from valid_bits range to 8-bit (0-255)
            max_value = (2 ** valid_bits) - 1
            array = np.clip(array, 0, max_value)
            array = (array.astype(np.float32) * 255.0 / max_value).astype(np.uint8)

        Image.fromarray(array).save(output_path)


@click.command()
@click.argument('output_dir', type=click.Path())
@click.option('--start', type=int, default=None,
              help='Starting volume number (e.g., 1 for MSLMST_0001)')
@click.option('--end', type=int, default=None,
              help='Ending volume number (e.g., 50 for MSLMST_0050)')
@click.option('--processing-code', type=click.Choice(['DRXX', 'DRCX', 'DRLX', 'DRCL'], case_sensitive=False),
              default='DRCL',
              help='Processing level: DRXX (radiometric), DRCX (color corrected), DRLX (linearized), DRCL (color + linearized, default)')
@click.option('--max-errors', type=int, default=10,
              help='Maximum consecutive download errors before giving up on a volume')
def main(output_dir, start, end, processing_code, max_errors):
    """
    Download all Mastcam images from the PDS archive.

    OUTPUT_DIR: Directory where PNG images will be saved

    Examples:
        # Auto-detect all volumes, download DRCL (default: color corrected + linearized, 8-bit)
        python download_mastcam.py /path/to/output

        # Download specific volume range with DRXX (radiometric only, 16-bit)
        python download_mastcam.py /path/to/output --start 1 --end 10 --processing-code DRXX

        # Download DRCX (color corrected, 8-bit)
        python download_mastcam.py /path/to/output --processing-code DRCX
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Determine which volumes to process
    if start is not None and end is not None:
        volumes = [f'MSLMST_{i:04d}' for i in range(start, end + 1)]
        click.echo(f"Processing volumes {start} to {end}...")
    elif start is not None or end is not None:
        click.echo("Error: --start and --end must be used together", err=True)
        return
    else:
        volumes = discover_volumes()
        if not volumes:
            click.echo("No volumes found!", err=True)
            return

    # Processing code descriptions
    proc_descriptions = {
        'DRXX': 'radiometric only, 16-bit',
        'DRCX': 'radiometric + color corrected, 8-bit',
        'DRLX': 'radiometric + linearized, 16-bit',
        'DRCL': 'radiometric + color corrected + linearized, 8-bit'
    }

    click.echo(f"\nFound {len(volumes)} volume(s) to process")
    click.echo(f"Processing code: {processing_code.upper()} ({proc_descriptions[processing_code.upper()]})\n")

    # Process each volume
    total_downloaded = 0
    total_skipped = 0
    total_errors = 0

    for volume_id in volumes:
        click.echo(f"\n{'='*60}")
        click.echo(f"Processing {volume_id}")
        click.echo(f"{'='*60}")

        # Fetch and parse index
        df = fetch_index(volume_id)
        if df is None:
            click.echo(f"  Skipping {volume_id} (could not fetch index)")
            continue

        click.echo(f"  Found {len(df)} entries in index")

        # Filter by processing code
        processing_code_upper = processing_code.upper()
        df = df[df['PRODUCT_ID'].str.contains(processing_code_upper, na=False)]
        click.echo(f"  Filtered to {len(df)} {processing_code_upper} entries")

        if len(df) == 0:
            click.echo(f"  No {processing_code_upper} files found in this volume")
            continue

        # Build list of download tasks
        tasks = []
        for _, row in df.iterrows():
            # PATH_NAME is the directory, FILE_NAME is the label file
            path_name = row['PATH_NAME'].strip()
            file_name = row['FILE_NAME'].strip()

            # Convert .LBL to .IMG to get the image file
            img_filename = file_name.replace('.LBL', '.IMG')

            # Construct full path and URL
            relative_path = f"{path_name}{img_filename}"
            img_url = f"{BASE_URL}/{volume_id}/{relative_path}"

            # Output filename (convert to PNG)
            product_id = row['PRODUCT_ID'].strip()
            output_path = os.path.join(output_dir, f"{product_id}.PNG")

            tasks.append((output_path, img_url))

        # Filter out existing files
        remaining = [task for task in tasks if not os.path.exists(task[0])]
        skipped = len(tasks) - len(remaining)

        click.echo(f"  Downloading {len(remaining)} image(s) (skipping {skipped} existing)")

        # Download with progress bar
        consecutive_errors = 0
        with tqdm(remaining, unit='img') as pbar:
            for output_path, img_url in pbar:
                try:
                    download_img_to_png(output_path, img_url)
                    total_downloaded += 1
                    consecutive_errors = 0
                except Exception as e:
                    total_errors += 1
                    consecutive_errors += 1
                    pbar.write(f"  Error downloading {os.path.basename(output_path)}: {e}")

                    if consecutive_errors >= max_errors:
                        click.echo(f"\n  Too many consecutive errors ({max_errors}), skipping rest of volume")
                        break

        total_skipped += skipped

    # Summary
    click.echo(f"\n{'='*60}")
    click.echo("Download Summary")
    click.echo(f"{'='*60}")
    click.echo(f"  Downloaded: {total_downloaded}")
    click.echo(f"  Skipped (existing): {total_skipped}")
    click.echo(f"  Errors: {total_errors}")
    click.echo(f"\nImages saved to: {output_dir}")


if __name__ == '__main__':
    main()
