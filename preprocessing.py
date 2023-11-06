import numpy as np
import glob
import cv2
import os

# For Data augmentation
from skimage import io
import random

import albumentations as A


def resize_images(directory: str, size: int) -> None:
    """
    Reads all images from the specified directory, resizes them and saves them under the same name.
    :param directory: Path to the directory containing the original images.
    :param size: The height and width value of new image.
    """
    count = 0

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        count += 1
        # Open the image file
        image = cv2.imread(os.path.join(directory, filename))
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(directory, filename), image)

    print(f"Images resized from {directory}: {count}")


def convert_images_to_binaries(directory: str, file_extension: str = ".jpg") -> None:
    """
    Reads all images from the specified directory, resizes them and saves them under DIFFERENT name if the file
    extension is NOT ".tif".
    :param directory: Path to the directory containing the original images.
    :param file_extension: The extension of files (all files must have the same extension before converting).
    """
    paths = glob.glob(directory + "*" + file_extension)
    images = [cv2.imread(path, 0) for path in paths]

    threshold = 128

    binary_images = []
    for img in images:
        # Apply thresholding and convert to integers
        binary_img = np.where(img > threshold, 255, 0)
        binary_images.append(binary_img)

    for path, bin in zip(paths, binary_images):
        new_path = path.replace(file_extension, ".png")
        cv2.imwrite(new_path, bin)

    print(f"Images converted to binary from: {directory}")


def augment_data(images_path: str, masks_path: str, img_augmented_path: str, msk_augmented_path: str,
                 images_to_generate: int = 10) -> None:
    """
    This function performs data augmentation for images and their corresponding masks.
    :param images_path: Path to the directory containing the original images.
    :param masks_path: Path to the directory containing the masks.
    :param img_augmented_path: Path to the directory where the augmented images will be saved.
    :param msk_augmented_path: Path to the directory where the augmented masks will be saved.
    :param images_to_generate: Number of augmented images to generate.
    """
    images = []
    masks = []

    for im in os.listdir(images_path):
        images.append(os.path.join(images_path, im))

    for msk in os.listdir(masks_path):
        masks.append(os.path.join(masks_path, msk))

    images.sort()
    masks.sort()

    aug = A.Compose([
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=1),
        A.Transpose(p=1),
        A.GridDistortion(p=1)
    ])

    i = 1
    while i <= images_to_generate:
        number = random.randint(0, len(images) - 1)
        image = images[number]
        mask = masks[number]
        print(image, mask)
        original_image = io.imread(image)
        original_mask = io.imread(mask)

        augmented = aug(image=original_image, mask=original_mask)
        transformed_image = augmented['image']
        transformed_mask = augmented['mask']

        new_image_path = "%s/augmented_image_%s.tif" % (img_augmented_path, i)
        new_mask_path = "%s/augmented_mask_%s.png" % (msk_augmented_path, i)
        io.imsave(new_image_path, transformed_image)
        io.imsave(new_mask_path, transformed_mask)
        i = i + 1


# create 256x256 images
if __name__ == "__main__":
    dir_images = "/home/laurin/images_256/"
    dir_masks = "/home/laurin/masks_256/"
    convert_images_to_binaries(dir_masks, ".jpg")
    resize_images(dir_images, 256)
    resize_images(dir_masks, 256)

    images_path = dir_images
    masks_path = dir_masks
    img_augmented_path = dir_images
    msk_augmented_path = dir_masks
    augment_data(images_path, masks_path, img_augmented_path, msk_augmented_path, 1000)
