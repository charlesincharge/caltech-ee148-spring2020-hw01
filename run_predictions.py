import argparse
import pathlib
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats.stats import pearsonr
import json
from PIL import Image, ImageDraw
from joblib import Parallel, delayed


def detect_red_light(image_numpy, method, **kwargs):
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
    elif method == 'matchedfilter':
        bounding_boxes = detect_red_light_matchedfilter(image_numpy, **kwargs)
    elif method == 'matchedfilter_multi':
        bounding_boxes = detect_red_light_matchedfilter_multi(image_numpy)
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

    image_red = image_numpy[:, :, 0]
    image_green = image_numpy[:, :, 1]
    image_blue = image_numpy[:, :, 2]

    THRESHOLD_RED = 175
    THRESHOLD_NOT_RED = 125
    red_mask = (
        (image_red > THRESHOLD_RED)
        & (image_green < THRESHOLD_NOT_RED)
        & (image_blue < THRESHOLD_NOT_RED)
    )

    bounding_boxes = mask_to_bboxes(red_mask)

    return bounding_boxes


def detect_red_light_matchedfilter(
    image_numpy, threshold=0.9, filter_path='filters/redlight.jpg'
):
    """
    This function takes a numpy array <image_numpy> and returns a list <bounding_boxes>.
    Convolves image with a matched filter, then finds areas above the threshold.
    """
    matchedfilter = Image.open(filter_path)
    matchedfilter = np.asarray(matchedfilter)
    # l2-normalize
    matchedfilter = matchedfilter / matchedfilter.sum()

    # no color dimension for filter match
    # Create a filter_match just for visualizsation
    filter_match = np.zeros(image_numpy.shape[:-1])
    bbox_list = []
    # Iterate over left and right of
    for tl_row in range(image_numpy.shape[0] - matchedfilter.shape[0]):
        for tl_col in range(image_numpy.shape[1] - matchedfilter.shape[1]):
            sub_image = image_numpy[
                tl_row : (tl_row + matchedfilter.shape[0]),
                tl_col : (tl_col + matchedfilter.shape[1]),
            ]
            corr = pearsonr(sub_image.flatten(), matchedfilter.flatten())[0]
            filter_match[tl_row, tl_col] = corr

            # Is this sub-image highly correlated (ie, good projection onto) with the filter?
            if corr > threshold:
                bbox_list.append(
                    (
                        tl_row,
                        tl_col,
                        tl_row + matchedfilter.shape[0],
                        tl_col + matchedfilter.shape[1],
                    )
                )

    return bbox_list


def detect_red_light_matchedfilter_multi(
        image_numpy, threshold=0.9, filter_path=['filters/redlight.jpg', 'filters/bigredlight.jpg']):
    """Use multiple matched filters to detect sized red lights of different pixel-sizes."""
    bbox_list = []
    for filter_p in filter_path:
        bbox_list += detect_red_light_matchedfilter(image_numpy, threshold, filter_p)

    return bbox_list


def mask_to_bboxes(mask):
    """Convert boolean mask to a list of bounding boxes (around the countors)"""
    bbox_list = []
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bbox_list.append((x, y, x + w, y + h))

    return bbox_list


def swap_bbox_format(bbox_tuple):
    """Swap between (row0, col0, row1, col1) and (x0, y0, x1, y1) formats."""
    assert len(bbox_tuple) == 4
    return (bbox_tuple[1], bbox_tuple[0], bbox_tuple[3], bbox_tuple[2])


def parse_args():
    parser = argparse.ArgumentParser(description='Detect red lights in images.')
    parser.add_argument(
        '-m',
        '--method',
        help='detection method',
        choices=['threshold', 'matchedfilter', 'matchedfilter_multi'],
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
        '-f',
        '--filter-path',
        help='path to matched filter image. Only valid when using `--method matchedfilter`',
        default='filters/redlight.jpg',
        type=pathlib.Path,
    )
    parser.add_argument(
        '--save-images',
        help='save images with bounding boxes to output folder',
        action='store_true',
    )
    parser.add_argument(
        '-n',
        '--num-images',
        help='number of images to process. defaults to all, set to int to process fewer (eg, for debugging)',
        type=int,
        default=None,
    )

    return parser.parse_args()


def file_to_bounding_boxes(file_path, output_folder=None, save_images=False, **kwargs):
    """
    file_path: path to image file.
    """
    # read image using PIL:
    image = Image.open(file_path)

    # convert to numpy array:
    image_numpy = np.asarray(image)

    # Predict bounding boxes and store to dictionary
    bounding_boxes_pred = detect_red_light(image_numpy, args.method, **kwargs)

    if save_images:
        # Draw bounding boxes and save out images
        draw = ImageDraw.Draw(image)
        NEON_GREEN = '#39FF14'
        for bbox in bounding_boxes_pred:
            draw.rectangle(swap_bbox_format(bbox), outline=NEON_GREEN)
        image.save(output_folder.joinpath(file_path.name))

    return bounding_boxes_pred


args = parse_args()

# create directory if needed
args.output_folder.mkdir(exist_ok=True)

# get sorted list of files:
file_paths = sorted(args.data_folder.iterdir())

# remove any non-JPEG files:
file_paths = [f for f in file_paths if (f.suffix == '.jpg')]
# Limit files.
file_paths = file_paths[: args.num_images]


bbox_list = Parallel(n_jobs=-3)(
    delayed(file_to_bounding_boxes)(
        file_path, args.output_folder.joinpath(args.method), save_images=True, filter_path=args.filter_path
    )
    for file_path in file_paths
)
file_names = map(lambda x: x.name, file_paths)
bounding_boxes_preds = dict(zip(file_names, bbox_list))


# save preds (overwrites any previous predictions!)
output_path = args.output_folder.joinpath(args.method).joinpath('bounding_boxes_preds.json')
with output_path.open(mode='w') as f:
    json.dump(bounding_boxes_preds, f)
