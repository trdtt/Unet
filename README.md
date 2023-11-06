# U-Net

The u-net is convolutional network architecture for fast and precise segmentation of images.

## Heads up
The following code is written for [Google Colab](https://colab.research.google.com/), a Jupyter Notebook-like interface, 
since Google provides free graphics cards and appropriate drivers for training the models. Therefore the pictures are 
loaded from my personal Google Drive. If you want to use the data for your own applications you can find the links to 
the photos in the next section.

## Training the U-Net

- Training Datasets:
    - [Original images](https://drive.google.com/drive/folders/1kBbjizByZfB0j_8jbJnjTp_pmxTxbXDC?usp=sharing)
    - [Masks](https://drive.google.com/drive/folders/1UCQdCw1S3uxWtFY4rwOQx1wjZ4muzdbR?usp=sharing)

The file `unet_koropad.ipynb` contains the necessary steps and code snippets to create a CNN model for image
recognition. Below, you will find a breakdown of the contents and instructions on how to use them.

Preprocessing the images is a vital step before training the CNN model.

- To ensure compatibility with the model architecture, it's crucial to resize the images to the nearest power of 2.
- A thresholding operation can be utilized to convert grayscale images into binary format.
- The use of image augmentation techniques such as rotations, flips, and changes in brightness can help increase the
  number of training images.

The `preprocessing.py` file contains code that provides the implementation details for these steps. By optimizing the
image data for training, these preprocessing steps significantly boost the performance of the CNN model.

## Results

The relevant changes in each line are highlighted in bold.

| Model                                                             | Number of Images | Encoder Weights | Batch Size | Epochs  | Test Data (%) | Backbone | Optimizer | Loss Function       | Metrics  | Mean IoU (%) |
|-------------------------------------------------------------------|------------------|-----------------|------------|---------|---------------|----------|-----------|---------------------|----------|--------------|
| U-Net [(GitHub)](https://github.com/bnsreenu/python_for_image_processing_APEER/blob/master/tutorial118_binary_semantic_segmentation_using_unet.ipynb)                       | 1000             | None            | 16         | 25      | 10            | None     | Adam      | binary_crossentropy | accuracy | 77.56        |
| **U-Net [Python Segmentation Models](https://shorturl.at/nxJRZ)** | 1000             | None            | 16         | 25      | 10            | resnet34 | Adam      | binary_crossentropy | accuracy | 91.56        |
| U-Net [Python Segmentation Models](https://shorturl.at/nxJRZ)     | 1000             | None            | 16         | **40**  | 10            | resnet34 | Adam      | binary_crossentropy | accuracy | 92.20        |
| U-Net [Python Segmentation Models](https://shorturl.at/nxJRZ)     | 1000             | **imagenet**    | 16         | 25      | 10            | resnet34 | Adam      | binary_crossentropy | accuracy | 80.02        |
| U-Net [Python Segmentation Models](https://shorturl.at/nxJRZ)     | 1000             | None            | **8**      | **50**  | 10            | resnet34 | Adam      | binary_crossentropy | accuracy | 91.90        |
| U-Net [Python Segmentation Models](https://shorturl.at/nxJRZ)     | 1000             | None            | 16         | 25      | **20**        | resnet34 | Adam      | binary_crossentropy | accuracy | 90.58        |
| U-Net [Python Segmentation Models](https://shorturl.at/nxJRZ)     | 1000             | None            | 16         | **50**  | 20            | resnet34 | Adam      | binary_crossentropy | accuracy | 91.29        |
| U-Net [Python Segmentation Models](https://shorturl.at/nxJRZ)     | 1000             | None            | **8**      | 25      | 20            | resnet34 | Adam      | binary_crossentropy | accuracy | 91.33        |
| U-Net [Python Segmentation Models](https://shorturl.at/nxJRZ)     | 1000             | None            | **8**      | **100** | 20            | resnet34 | Adam      | binary_crossentropy | accuracy | 91.51        |
| U-Net [Python Segmentation Models](https://shorturl.at/nxJRZ)     | **2000**         | None            | 8          | 50      | 20            | resnet34 | Adam      | binary_crossentropy | accuracy | 92.13        |

The last row of the table corresponds to the data associated with the model `
korropad_unet-model_sm_noEncoder_256_b8_e50_2000.hdf5`.

### Possible Improvements

- Using different models.
- Employing higher image resolutions. However, this leads to RAM issues (12 GB was not sufficient).

## Testing the CNN

A pre-trained model named `korropad_unet-model_sm_noEncoder_256_b8_e50_2000.hdf5` is provided for immediate
testing. To test the pre-trained model, use the `test_trained_model.ipynb` to load the model and perform predictions on
your any image from the repository. An Example is provided, but feel free to change the path and test other images.
