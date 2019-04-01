"""
To be edited
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 humanparsing.py train --dataset=/path/to/humanparsing/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 humanparsing.py train --dataset=/path/to/humanparsing/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 humanparsing.py train --dataset=/path/to/humanparsing/dataset --weights=imagenet

    # Apply color splash to an image
    python3 humanparsing.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 humanparsing.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

# Human Parsing dataset has 18 classes including background
#below is a list of classes
Parsing_classes = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
                   'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt',
                   'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe',
                   'rightShoe']

class HumanConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "human"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 17  # 1 Background + 17 Parsing Catagories

    # Number of training steps per epochc
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class HumanDataset(utils.Dataset):

    def load_human(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        for i in len(Parsing_classes):
            self.add_class("human", i, Parsing_classes[i])

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        # To Be Written
        # annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        # annotations = list(annotations.values())  # don't need the dict keys

        # # The VIA tool saves images in the JSON even if they don't have any
        # # annotations. Skip unannotated images.
        # annotations = [a for a in annotations if a['regions']]

        # # Add images
        # for a in annotations:
        #     # Get the x, y coordinaets of points of the polygons that make up
        #     # the outline of each object instance. These are stores in the
        #     # shape_attributes (see json format above)
        #     # The if condition is needed to support VIA versions 1.x and 2.x.
        #     if type(a['regions']) is dict:
        #         polygons = [r['shape_attributes'] for r in a['regions'].values()]
        #     else:
        #         polygons = [r['shape_attributes'] for r in a['regions']] 

        #     # load_mask() needs the image size to convert polygons to masks.
        #     # Unfortunately, VIA doesn't include it in JSON, so we must read
        #     # the image. This is only managable since the dataset is tiny.
        #     image_path = os.path.join(dataset_dir, a['filename'])
        #     image = skimage.io.imread(image_path)
        #     height, width = image.shape[:2]

        #     self.add_image(
        #         "balloon",
        #         image_id=a['filename'],  # use file name as a unique image id
        #         path=image_path,
        #         width=width, height=height,
        #         polygons=polygons)

    def load_mask(self, image_id):
    # To Be Written
    #     """Generate instance masks for an image.
    #    Returns:
    #     masks: A bool array of shape [height, width, instance count] with
    #         one mask per instance.
    #     class_ids: a 1D array of class IDs of the instance masks.
    #     """
    #     # If not a balloon dataset image, delegate to parent class.
    #     image_info = self.image_info[image_id]
    #     if image_info["source"] != "balloon":
    #         return super(self.__class__, self).load_mask(image_id)

    #     # Convert polygons to a bitmap mask of shape
    #     # [height, width, instance_count]
    #     info = self.image_info[image_id]
    #     mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
    #                     dtype=np.uint8)
    #     for i, p in enumerate(info["polygons"]):
    #         # Get indexes of pixels inside the polygon and set them to 1
    #         rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
    #         mask[rr, cc, i] = 1

    #     # Return mask, and array of class IDs of each instance. Since we have
    #     # one class ID only, we return an array of 1s
    #     return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        pass
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "human":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = HumanDataset()
    dataset_train.load_human(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = HumanDataset()
    dataset_val.load_human(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect human parsing.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/human/dataset/",
                        help='Directory of the human dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = HumanConfig()
    else:
        class InferenceConfig(HumanConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
