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
# Description : MSL surface image fetcher via PDS index lookup
#               – Loads requested product IDs from one or more ID files
#               – Cross-references an INDEX_TABLE to resolve volume/sol
#               – Fetches preview JPGs, or raw IMG+LBL pairs converted to PNG
#               – Reports missing and duplicated product IDs
#               – Skips files that already exist on disk
# =============================================================================
import os
import pdr
import click
import requests
import tempfile
import numpy as np
from PIL import Image
from tqdm import tqdm


JPG_PATH = 'https://planetarydata.jpl.nasa.gov/img/data/msl/{volume_id}/EXTRAS/RDR/SURFACE/FULL/{sol:04d}/{product_id}.JPG'
PNG_PATH = 'https://planetarydata.jpl.nasa.gov/img/data/msl/{volume_id}/DATA/RDR/SURFACE/{sol:04d}/{product_id}.IMG'


def load_ids(idfile):
    with open(idfile, 'r') as f:
        return [
            os.path.splitext(line.split(' ')[0])[0]
            for line in f
            if '-' not in line
        ]


def fetch_jpg(outfile, url):
    r = requests.get(url)
    r.raise_for_status()

    with open(outfile, "wb") as f:
        f.write(r.content)


def fetch_img(outfile, url):
    r = requests.get(url)
    r.raise_for_status()
    img_bytes = r.content

    r = requests.get(url.replace('.IMG', '.LBL'))
    r.raise_for_status()
    lbl_bytes = r.content

    base = os.path.splitext(os.path.basename(outfile))[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        lbl_path = os.path.join(tmpdir, f'{base}.LBL')
        img_path = os.path.join(tmpdir, f'{base}.IMG')

        with open(lbl_path, 'wb') as f: f.write(lbl_bytes)
        with open(img_path, 'wb') as f: f.write(img_bytes)

        data = pdr.read(lbl_path)

        array = np.squeeze(np.asarray(data["IMAGE"]))

        # If bands-first
        if array.ndim == 3 and array.shape[0] == 3:
            array = np.transpose(array, (1, 2, 0))

        Image.fromarray(array).save(outfile)


@click.command()
@click.argument('idfiles', nargs=-1)
@click.argument('idxfile')
@click.argument('outputdir')
@click.option('-r', '--raw', is_flag=True)
def main(idfiles, idxfile, outputdir, raw):
    id_list = set(sum([ load_ids(i) for i in idfiles ], []))

    data = pdr.read(idxfile)
    df = data['INDEX_TABLE']

    # Filter
    subset = df[df["PRODUCT_ID"].isin(id_list)]

    # Missing
    found_ids = set(subset["PRODUCT_ID"])
    missing_ids = id_list - found_ids

    # Duplicates
    counts = subset["PRODUCT_ID"].value_counts()
    duplicated_ids = counts[counts > 1]

    print(f"Requested IDs: {len(id_list)}")
    print(f"Rows returned: {len(subset)}")
    print(f"Missing IDs: {len(missing_ids)}")
    print(f"Duplicated IDs: {len(duplicated_ids)}")

    suffix = 'PNG' if raw else 'JPG'
    fmtstr = PNG_PATH if raw else JPG_PATH

    tasks = [
        (
            os.path.join(outputdir, row['PRODUCT_ID'] + f'.{suffix}'),
            fmtstr.format(
                volume_id=row['VOLUME_ID'],
                sol=row['PLANET_DAY_NUMBER'],
                product_id=row['PRODUCT_ID'],
            )
        )
        for _, row in subset.iterrows()
    ]

    # Skip any existing files
    remaining = [
        task for task in tasks
        if not os.path.exists(task[0])
    ]

    fetch_fn = fetch_img if raw else fetch_jpg

    for task in tqdm(remaining):
        fetch_fn(*task)


if __name__ == '__main__':
    main()
