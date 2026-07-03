# =============================================================================
# Author        : Dr. Gary B. Doran
# Role          : Data Scientist
# Affiliation   : Jet Propulsion Laboratory, California Institute of Technology,
#                 Pasadena, CA 91109
# Collaboration : Pranav Durai
#                 Stanford Center for Innovation in In Vivo Imaging,
#                 Stanford University School of Medicine, Stanford, CA 94305
#
# Description : Full-resolution HiRISE cutout generator
#               – Takes downloaded HiRISE .JP2 images plus browse-resolution
#                 trace/label annotations ('labels-map-proj_v3_2.txt' and
#                 'traces-map-proj_v3_2.json')
#               – Scales annotated bounding boxes to full resolution
#               – Extracts square cutouts with margin around each ROI
#               – Falls back to the least-black-pixel window near borders
#               – Saves cutouts into per-class output directories
# =============================================================================

# Library Imports
import click
from pathlib import Path
import rasterio
import json
import os
import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from shapely import bounds
from shapely.geometry import shape
from shapely.affinity import scale
from collections import defaultdict
from rasterio.windows import Window
from tqdm import tqdm

# Variables 
HIRISE_BROWSE_WIDTH = 2048
MARGIN = 30 # browse pixels
BLACK_THRESHOLD = 0.2 # threshold for computing black_region
NEW_SIZE = (1024, 1024)
SIZE_THRESHOLD = 512


def get_center_cutout(img, size):
    h, w = img.shape
    start_col = (w - size) // 2
    start_row = (h - size) // 2
    return img[start_row:start_row + size, start_col:start_col + size]


def get_best_cutout(img, size):
    '''
    returns the square cutout from a rectangular region with the fewest number
    of black pixels
    '''
    mask = (img == 0).astype(float)

    # Handle even/odd window sizes
    win = size if size % 2 == 0 else size + 1

    filtered = uniform_filter(
        mask, (win, win), origin=(-win // 2, -win // 2),
        mode='constant', cval=np.inf
    )
    filtered[np.isnan(filtered)] = np.inf

    # Get best row/col for window start (with minimal black pixel values)
    argmin = np.argmin(filtered)
    best_row, best_col = np.unravel_index(argmin, filtered.shape)

    return img[best_row:best_row + size, best_col:best_col + size]


def load_cutout_info(classfile, tracefile):

    # Maps each image to a list of cut-outs within the image
    cutout_mapping = defaultdict(list)
    class_ids = set([])

    # Load json file with bounding box data
    with open(tracefile, 'r') as f:
        bbox_data = json.load(f)

    # Load label mappings
    with open(classfile, 'r') as f:
        for line in f:
            line = line.strip().split()
            assert len(line) == 2 # Expect 2 entries per line
            cutout_file, class_id = line

            cutout_file_parts = cutout_file.split('-')
            if len(cutout_file_parts) > 2: continue # Skip augmented cutouts
            img_name = cutout_file_parts[0] + ".JP2"
            cutout_id = os.path.splitext(cutout_file)[0]

            if cutout_id not in bbox_data:
                raise ValueError(f'Missing trace polygon for {cutout_id}')

            class_ids.add(class_id)

            polygon = shape(bbox_data[cutout_id])
            info = (cutout_id, class_id, polygon)
            cutout_mapping[img_name].append(info)

    return cutout_mapping, class_ids


def create_dataset(hirisedir, outputdir, classfile, tracefile):
    '''
    Creates full-resolution cut-outs from HiRISE images

    @param hirisedir: directory with full-resolution HiRISE files
    @param outputdir: base directory for classified cutouts
    @param classfile: space-delimited file mapping cutouts to class ids
    @param tracefile: file containing geojson cutout trace information
    '''

    # Maps each image to a list of cut-outs,
    # labels, and polygons within the image
    cutout_mapping, class_ids = load_cutout_info(classfile, tracefile)

    # Create directory structure for classified cutouts
    for class_id in sorted(class_ids):
        class_dir = os.path.join(outputdir, class_id)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)

    items = sorted(cutout_mapping.items())
    for img_name, cutouts in tqdm(items, 'Creating cutouts'):

        hirisefile = os.path.join(hirisedir, img_name)
        if not os.path.exists(hirisefile):
            print(f'Warning: skipping {img_name} (file not found)')
            continue

        with rasterio.open(hirisefile) as dataset:
            for cutout_id, class_id, browse_polygon in cutouts:
                outputfile = os.path.join(outputdir, class_id, f'{cutout_id}.jpg')
                if os.path.exists(outputfile):
                    continue

                # Conversion factor
                conversion_factor = dataset.width / HIRISE_BROWSE_WIDTH

                # Scale polygon to full resolution
                full_res_polygon = scale(
                    browse_polygon,
                    xfact=conversion_factor, yfact=conversion_factor,
                    origin=(0, 0)
                )

                fr_margin = conversion_factor * MARGIN

                # Get bounds of polygon
                col_lo, row_lo, col_hi, row_hi = bounds(full_res_polygon)

                width = col_hi - col_lo
                height = row_hi - row_lo
                # Square side length (greater of width, height)
                side = int(np.round(max(width, height) + 2 * fr_margin))

                if side < SIZE_THRESHOLD:
                    print(f'Skipping {cutout_id} (too small: {side} x {side})')
                    continue

                # Get region that encompasses all possible square bounding
                # boxes containing original polygon
                left = max(0, int(np.floor(col_hi - side)))
                right = min(dataset.width, int(np.ceil(col_lo + side)))
                top = max(0, int(np.floor(row_hi - side)))
                bottom = min(dataset.height, int(np.ceil(row_lo + side)))

                # Read in window of data
                window = Window.from_slices((top, bottom), (left, right))
                img = dataset.read(1, window=window)

                # Rescale image from 0 to 255
                img = (255. * (img - img.min()) / np.ptp(img)).astype(np.uint8)

                center_img = get_center_cutout(img, side)

                # Calculate the percentage of black pixels in center cutout
                black_percentage = np.average(center_img == 0)

                # Check whether we should fall back to getting best region to
                # exclude borders
                if black_percentage > BLACK_THRESHOLD:
                    best_img = get_best_cutout(img, side)
                else:
                    best_img = center_img

                resized_img = cv2.resize(
                    best_img, dsize=NEW_SIZE,
                    interpolation=cv2.INTER_CUBIC
                )

                # Save the image to the relevant folder
                cv2.imwrite(outputfile, resized_img)

    print("Processing complete!")


@click.command()
@click.argument(
    'hirisedir',
    type=click.Path(path_type=Path, exists=True),
)
@click.argument(
    'outputdir',
    type=click.Path(path_type=Path, exists=True),
)
@click.argument(
    'classfile',
    type=click.Path(path_type=Path, exists=True),
)
@click.argument(
    'tracefile',
    type=click.Path(path_type=Path, exists=True),
)
def main(hirisedir, outputdir, classfile, tracefile):
    create_dataset(hirisedir, outputdir, classfile, tracefile)


if __name__ == '__main__':
    main()
