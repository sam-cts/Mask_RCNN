"""
Mask R-CNN
Train on the LIP dataset
------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 humanparsing.py train --dataset=/path/to/humanparsing/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 humanparsing.py train --dataset=/path/to/humanparsing/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 humanparsing.py train --dataset=/path/to/humanparsing/dataset --weights=imagenet
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2 as cv
import imgaug
import tensorflow as tf

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
############################################################
#  Configurations
############################################################

class HumanConfig(Config):
    """Configuration for training on the LIP dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "human"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 19  # 1 Background + 19 Parsing Catagories

    # # Reduce training ROIs per image because the images are small and have
    # # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 33

    # Number of training steps per epochc
    STEPS_PER_EPOCH = 100

    # Skip detections with < 95% confidence
    DETECTION_MIN_CONFIDENCE = 0.95

############################################################
#  Dataset
############################################################

class HumanDataset(utils.Dataset):

    def load_human(self, dataset_dir, subset):
        """Load a subset of the LIP dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("human", 1, "Hat")
        self.add_class("human", 2, "Hair")
        self.add_class("human", 3, "Glove")
        self.add_class("human", 4, "Sunglasses")
        self.add_class("human", 5, "UpperCloths")
        self.add_class("human", 6, "Dress")
        self.add_class("human", 7, "Coat")
        self.add_class("human", 8, "Socks")
        self.add_class("human", 9, "Pants")
        self.add_class("human", 10, "Jumpsuits")
        self.add_class("human", 11, "Scarf")
        self.add_class("human", 12, "Skirt")
        self.add_class("human", 13, "Face")
        self.add_class("human", 14, "Left-arm")
        self.add_class("human", 15, "Right-arm")
        self.add_class("human", 16, "Left-leg")
        self.add_class("human", 17, "Right-leg")
        self.add_class("human", 18, "Left-shoe")
        self.add_class("human", 19, "Right-shoe")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        dataset_list = os.path.join(dataset_dir, "{}_id.txt".format(subset))

        f = open(dataset_list, 'r')
        f = f.read().splitlines()
        f = self.check_seg(f, dataset_dir, subset)
        for img in f:
            # Get Image Information
            image_path = os.path.join(dataset_dir, "{}_images/{}.jpg".format(subset,img))
            self.add_image(
                "human", 
                image_id=img, 
                path = image_path,
                subset = subset)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a human dataset image, delegate to parent class.
        info = self.image_info[image_id]
        seg_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), 
        "{}_segmentations".format(info['subset']))

        # Get Image Segmentation
        img_seg = cv.imread(os.path.join(seg_dir, info['id'] +".png"))
        img_seg = cv.cvtColor(img_seg, cv.COLOR_BGR2GRAY)
        classes_seg = np.unique(img_seg)
        mask = []
        class_id = []

        for i in range(len(classes_seg)):
            if i == 0:
                continue

            img_seg2 = img_seg.copy()
            img_seg2[img_seg2 != classes_seg[i]] = 0
            mask.append(img_seg2.astype(np.bool))
            class_id.append(classes_seg[i])

        if class_id:
            mask = np.stack(mask, axis=2).astype(np.bool)
            class_id = np.array(class_id, dtype=np.int32)
            return mask, class_id

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "human":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
    

    def check_seg(self, data_list, dataset_dir, subset):
        seg_dir = os.path.join(dataset_dir, "{}_segmentations".format(subset))
        filtered_list = []
        for img_id in data_list:
            img_seg = cv.imread(os.path.join(seg_dir, img_id +".png"))
            img_seg = cv.cvtColor(img_seg, cv.COLOR_BGR2GRAY)
            if len(np.unique(img_seg)) != 1:
                filtered_list.append(img_id)
        return filtered_list
            



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
    # COCO trained weights.

    augmentation = imgaug.augmenters.Fliplr(0.5)

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads',
                augmentation=augmentation)

    # Fine tune Resnet4 and up layers
    # Passing layers="all" trains all layers. You can also 
    # pass a regular expression to select which layers to
    # train by name pattern.
    print("Fine Tune Resnet4 and up")
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE,
                epochs=120, 
                layers="4+",
                augmentation=augmentation)

    print("Fine Tune all layers")
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE / 10,
                epochs=160, 
                layers="all",
                augmentation=augmentation)

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
              "Use 'train'".format(args.command))
