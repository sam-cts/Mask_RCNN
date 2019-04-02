#%% A Test Demo to find contours
import os
import sys
import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
image_dir = ROOT_DIR+"/datasets/lip/train/train_segmentations/"

#%%
A = image_dir+'77_471474.png'
print(A)
img_seg = cv.imread(A)

# plt.subplot(1, 2, 1)
# plt.title('Original')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.imshow(img_org)
plt.subplot(1, 2, 2)
plt.title('Segments')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.imshow(img_seg)
plt.show()

# #%%
# # This is to get mask from segments
# img_seg_cvt= cv.cvtColor(img_seg, cv.COLOR_BGR2GRAY)    # Convert color space
# plt.subplot(1, 2, 1)
# plt.title('Segments')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.imshow(img_seg)
# plt.subplot(1, 2, 2)
# plt.title('Convert')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.imshow(img_seg_cvt)
# plt.show()

#%%
img_seg_cvt= cv.cvtColor(img_seg, cv.COLOR_BGR2GRAY)    # Convert color space
classes_seg = np.unique(img_seg_cvt)

#%%
for i in classes_seg:
    if i == 0:
        continue

    img_seg_cvt2 = img_seg_cvt.copy()
    img_seg_cvt2[img_seg_cvt2 != i] = 0

    plt.subplot(1, 2, 2)
    plt.title(i)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.imshow(img_seg_cvt2)
    plt.show()
#%%
img_seg_cvt2 = img_seg_cvt.copy()
img_seg_cvt2[img_seg_cvt2 != 15] = 0

print(img_seg_cvt2)

boolimg = img_seg_cvt2.astype(np.bool)

print(boolimg)
#%%    
img_seg_cvt2 = img_seg_cvt.copy()
img_seg_cvt2[img_seg_cvt2 != 15] = 0
print("Claculate for class:", 15)

contours, _ = cv.findContours(
    img_seg_cvt2, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)    # Find contours of the shape

contours_gt = []
contours_x = []
contours_y = []
for i in range(len(contours)):
    for j in range(len(contours[i])):
        print(contours[i][j])
        contours_gt.append(contours[i][j][0].tolist())
        contours_x.append(contours[i][j][0][0])
        contours_y.append(contours[i][j][0][1])
#%%
classes_seg = np.unique(img_seg_cvt)
seg_contours = []
class_id = []

for i in range(len(classes_seg)):
    if classes_seg[i] == 0:
        continue
    img_seg_cvt2 = img_seg_cvt.copy()
    img_seg_cvt2[img_seg_cvt2 != classes_seg[i]] = 0
    print("Claculate for class:", classes_seg[i])

    # Find contours of the shape
    contours, _ = cv.findContours(
    img_seg_cvt2, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)    # Find contours of the shape
    contours_y = []
    contours_x = []
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            contours_y.append(contours[i][j][0][0])
            contours_x.append(contours[i][j][0][1])

    seg_contours.append([contours_y, contours_x, i])
    class_id.append(classes_seg[i])




    
