#%% [markdown]
# # Mask R-CNN Demo for Chip Detection
# ## Method
# 
# I have used Coco-pretrained model to do a transfer training with prepared dataset
#
# ## Dataset
#
# The dataset used for training is self-prepared
# It contains 10 stacked chips photos, and 195 single chips photos for train set and val set together
# the stacked photos are annotated manually, and the single photos are annotated automatically with background removal


#%%
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

#Output directory of results
OUT_DIR = os.path.join(ROOT_DIR, "result")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/chips"))  # To find local version
import chips

get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
MODEL_PATH = os.path.join(ROOT_DIR, "logs/chips20190628T1143/mask_rcnn_chips_0030.h5")

# Directory of TeamPhoto to run detection
IMAGE_DIR = "D:/Datasets/Chips/2nd_Valid_Data/val"

#%% [markdown]
# ## Configurations
# 
# The configurations of this model are in the ```ChipsConfig``` class in ```chips.py```.
# In this trial, no configurations change, it is the same as the demo on github to test the preformance in default setting. 
# 

#%%
class InferenceConfig(chips.ChipsConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

#%% [markdown]
# ## Create Model and Load Trained Weights

#%%
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(MODEL_PATH, by_name=True)

#%% [markdown]
# ## Class Names
# 
# In this demo, we are only classifying background and chips

#%%
# Chips Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'chips']

#%% [markdown]
# ## Run Object Detection
#
# ### Single Chip Detection
# As shown below, the model has no problem on detecting single chip, and able to segmentate the chips
#%%
# Load a Random Photo from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection
results = model.detect([image], verbose=1)

# Visualize and Export results
f, ax = plt.subplots()
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'], show_bbox=False, ax = ax)

plt.show()

#%% [markdown]
# ### Stacked Chips - Partial
# The model seems not having difficulty on partially stacked chips as well.
#%%
STACK_DIR = "D:/Datasets/Chips/test_set/full_size"
file_names = next(os.walk(STACK_DIR))[2]

for file_name in file_names:
    image = skimage.io.imread(os.path.join(STACK_DIR, file_name))

    # Run detection
    results = model.detect([image], verbose=1)
    f, ax = plt.subplots()
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'], show_bbox=False, ax = ax)
plt.show()


#%%
#%% [markdown]
# ### Stacked Chips - Even
# This model only able to detect stacked chips when clear line is separating the chips. For instance, red chips stack with green.
# from the dataset below, there is one more problem is that it classify computer mouse as a chip as well
#%%
STACK_DIR = "D:/Datasets/Chips/Stacked Photos/Even"
file_names = next(os.walk(STACK_DIR))[2]

for file_name in file_names:
    image = skimage.io.imread(os.path.join(STACK_DIR, file_name))

    # Run detection
    results = model.detect([image], verbose=1)
    f, ax = plt.subplots()
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'], show_bbox=False, ax = ax)
plt.show()
