import argparse
import pathlib
import numpy as np
import json
from PIL import Image, ImageDraw

def detect_red_light(image_numpy):
    '''
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
    '''


    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 

    '''
    BEGIN YOUR CODE
    '''

    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''

    box_height = 8
    box_width = 6

    num_boxes = np.random.randint(1,5) 

    for box_idx in range(num_boxes):
        (n_rows,n_cols,n_channels) = np.shape(image_numpy)

        tl_row = np.random.randint(n_rows - box_height)
        tl_col = np.random.randint(n_cols - box_width)
        br_row = tl_row + box_height
        br_col = tl_col + box_width

        bounding_boxes.append([tl_row,tl_col,br_row,br_col]) 

    '''
    END YOUR CODE
    '''

    for bounding_box in bounding_boxes:
        assert len(bounding_box) == 4

    return bounding_boxes


def parse_args():
    parser = argparse.ArgumentParser(description='Detect red lights in images.')
    parser.add_argument('-d', '--data-folder', help='folder of images with red lights',
            default='data/RedLights2011_Medium', type=pathlib.Path)
    parser.add_argument('-o', '--output-folder', help='folder to output predictions',
            default='results/hw01_preds', type=pathlib.Path)
    parser.add_argument('--save-images', help='save images with bounding boxes to output folder',
            action='store_true')

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
    bounding_boxes_pred = detect_red_light(image_numpy)
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
    json.dump(bounding_boxes_preds,f)
