"""
Load bounding boxes from JSON and plot them on the imges
"""

import argparse
import pathlib
import numpy as np
import json
from PIL import Image, ImageDraw
from typing import List


def swap_bbox_format(bbox_tuple):
    """Swap between (row0, col0, row1, col1) and (x0, y0, x1, y1) formats."""
    assert len(bbox_tuple) == 4
    return (bbox_tuple[1], bbox_tuple[0], bbox_tuple[3], bbox_tuple[2])


def parse_args():
    parser = argparse.ArgumentParser(description='plot bounding boxes on images red lights in images.')
    parser.add_argument(
        '-d',
        '--data-folder',
        help='folder of images with red lights',
        default='data/RedLights2011_Medium',
        type=pathlib.Path,
    )
    parser.add_argument(
        '-j',
        '--json-path',
        help='path to bounding box predictions',
        default='results/hw01_preds/matchedfilter/bounding_boxes_preds.json',
        type=pathlib.Path,
    )

    return parser.parse_args()


def draw_on_file(file_path : pathlib.Path, bbox_list : List, output_folder : pathlib.Path, save_images : bool = True):
    """
    file_path: pathlib.Path to image file.
    bbox_list: list of 4-tuples
    output_folder: pathlib.Path to save out drawn-on image
    """
    # read image using PIL:
    image = Image.open(file_path)

    # Draw bounding boxes and save out images
    draw = ImageDraw.Draw(image)
    NEON_GREEN = '#39FF14'
    for bbox in bbox_list:
        assert len(bbox) == 4
        draw.rectangle(swap_bbox_format(bbox), outline=NEON_GREEN)

    if save_images:
        output_path = output_folder.joinpath(file_path.name)
        print('Saving to:', output_path)
        image.save(output_path)


def main():
    args = parse_args()

    # Load bounding boxes
    with args.json_path.open() as f:
        bounding_boxes_preds = json.load(f)

    for file_name, bbox_list in bounding_boxes_preds.items():
        file_path = args.data_folder.joinpath(file_name)
        draw_on_file(file_path, bbox_list, args.json_path.parent, save_images=True)


if __name__ == '__main__':
    main()
