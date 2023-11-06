from keras.models import load_model
import glob
import cv2
import numpy as np
from tensorflow.keras.metrics import MeanIoU
import random
from matplotlib import pyplot as plt

model = load_model("korropad_unet-model_sm_noEncoder_256_b8_e50_2000.hdf5", compile=False)

# Loading the test data
BASE_IMG_PATH = "../../data/Temperatur/1 h/1.4021/"
BASE_MASK_PATH = "../../data/Temperatur/1 h/1.4021/bin/"
SIZE = 256

img_paths = glob.glob(BASE_IMG_PATH + "*.tif") + glob.glob(BASE_IMG_PATH + "*.TIF")
img_paths.sort()
images = [cv2.imread(path, 0) for path in img_paths]

# resize images to fit the model
images_256 = [cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) for image in images]
image_dataset = np.array(images_256)
image_dataset = np.expand_dims(image_dataset, axis=3)
image_dataset = image_dataset / 255.

mask_paths = glob.glob(BASE_MASK_PATH + "*.jpg")
mask_paths.sort()
masks = [cv2.imread(path, 0) for path in mask_paths]
masks_256 = [cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) for mask in masks]
mask_dataset = np.array(masks_256)
mask_dataset = np.expand_dims(mask_dataset, axis=3)
mask_dataset = mask_dataset / 255.

# IOU
y_pred = model.predict(image_dataset)
y_pred_thresholded = y_pred > 0.5
n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_pred_thresholded, mask_dataset)
print(f"Mean IoU = {IOU_keras.result().numpy()} / 1.0")

# Get random image from test data set and compare prediction with actual mask
test_img_number = random.randint(0, len(image_dataset))
test_img = image_dataset[test_img_number]
ground_truth = mask_dataset[test_img_number]
test_img_input = np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input)[0, :, :, 0] > 0.5).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Original Image')
plt.imshow(test_img[:, :, 0], cmap='gray')
plt.subplot(232)
plt.title('Mask')
plt.imshow(ground_truth[:, :, 0], cmap='gray')
plt.subplot(233)
plt.title('Prediction')
plt.imshow(prediction, cmap='gray')

plt.show()
