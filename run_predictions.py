import argparse
import pathlib
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, measurements
from skimage.measure import regionprops
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2


def detect_red_light(image_numpy, method):
    """
    This function takes a numpy array <image_numpy> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the
    image. Each element of <bounding_boxes> should itself be a list, containing
    four integers that specify a bounding box: the row and column index of the
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.

    Note that PIL loads images in RGB order, so:
    image_numpy[:,:,0] is the red channel
    image_numpy[:,:,1] is the green channel
    image_numpy[:,:,2] is the blue channel
    """

    if method == 'threshold':
        bounding_boxes = detect_red_light_threshold(image_numpy)
    else:
        raise NotImplementedError

    # Check that boxes each have 4 coordinates
    for bounding_box in bounding_boxes:
        assert len(bounding_box) == 4

    return bounding_boxes


def detect_red_light_threshold(image_numpy):
    """
    This function takes a numpy array <image_numpy> and returns a list <bounding_boxes>.
    Smoothes data with a Gaussian kernel, then draw boxes around values above a threshold
    """
    # This should be a list of lists, each of length 4. See format example below.
    bounding_boxes = []

    # Smooth along x and y axes, but not color axis
    image_numpy = gaussian_filter(image_numpy, sigma=[1, 1, 0])

    image_red = image_numpy[:,:,0]
    image_green = image_numpy[:,:,1]
    image_blue = image_numpy[:,:,2]

    THRESHOLD_RED = 175
    THRESHOLD_NOT_RED = 125
    red_mask = (image_red > THRESHOLD_RED) & (image_green < THRESHOLD_NOT_RED) & (image_blue < THRESHOLD_NOT_RED)

    bounding_boxes = mask_to_bboxes(red_mask)

    return bounding_boxes


def mask_to_bboxes(mask):
    """Convert boolean mask to a list of bounding boxes (around the countors)
    """
    bbox_list = []
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        bbox_list.append((x, y, x + w, y + h))

    return bbox_list


def parse_args():
    parser = argparse.ArgumentParser(description='Detect red lights in images.')
    parser.add_argument(
        '-m',
        '--method',
        help='detection method',
        choices=['threshold', 'matchedfilter'],
        default='threshold',
    )
    parser.add_argument(
        '-d',
        '--data-folder',
        help='folder of images with red lights',
        default='data/RedLights2011_Medium',
        type=pathlib.Path,
    )
    parser.add_argument(
        '-o',
        '--output-folder',
        help='folder to output predictions',
        default='results/hw01_preds',
        type=pathlib.Path,
    )
    parser.add_argument(
        '--save-images',
        help='save images with bounding boxes to output folder',
        action='store_true',
    )

    return parser.parse_args()


args = parse_args()

# create directory if needed
args.output_folder.mkdir(exist_ok=True)

# get sorted list of files:
file_paths = sorted(args.data_folder.iterdir())

# remove any non-JPEG files:
file_paths = [f for f in file_paths if (f.suffix == '.jpg')]

bounding_boxes_preds = {}
for file_path in file_paths:

    # read image using PIL:
    image = Image.open(file_path)

    # convert to numpy array:
    image_numpy = np.asarray(image)

    # Predict bounding boxes and store to dictionary
    bounding_boxes_pred = detect_red_light(image_numpy, args.method)
    bounding_boxes_preds[file_path.name] = bounding_boxes_pred

    if args.save_images:
        # Draw bounding boxes and save out images
        draw = ImageDraw.Draw(image)
        NEON_GREEN = '#39FF14'
        for box in bounding_boxes_pred:
            draw.rectangle(box, outline=NEON_GREEN)
        image.save(args.output_folder.joinpath(file_path.name))


# save preds (overwrites any previous predictions!)
output_path = args.output_folder.joinpath('bounding_boxes_preds.json')
with output_path.open(mode='w') as f:
    json.dump(bounding_boxes_preds, f)
