#  Scale-Invariant Feature Transform (SIFT) Algorithm

The Scale-Invariant Feature Transform (SIFT) algorithm is a powerful computer vision technique 
for detecting and describing local features in images. 
These features can be used for various computer vision tasks such as object recognition, image stitching, and more. 

This Python implementation of the SIFT algorithm uses TensorFlow for efficient image processing.

![](demo_figs\\demo_gif.gif)

## **Prerequisites**

Before using this SIFT implementation, ensure you have the following prerequisites:

* Python 3.9
* TensorFlow
* OpenCV (optional for matching): If you want to use OpenCV for matching.

Please note that while TensorFlow can be used for matching, having access to a GPU is recommended for improved performance.

## **Getting Started**

To utilize the SIFT algorithm, follow these steps:

1. Import the necessary modules: 

`from viz import show_key_points, show_images, plot_matches_TF, plot_matches_CV2`

`from utils import load_image, templet_matching_TF, templet_matching_CV2`

`from sift import SIFT`

2. Initialize the SIFT object with your desired parameters:

`sift = SIFT(
    sigma=1.6,
    assume_blur_sigma=0.5,
    n_intervals=3,
    n_octaves=None,  # You can specify the number of octaves if needed
    border_width=5,
    convergence_iter=5
)`

3. Load your input image as a TensorFlow tensor and preprocess it. You can use the provided utils.load_image function or any other library

Note: The input image should be in grayscale.

4. Detect keypoints and compute descriptors:

`keypoints, descriptors = sift.keypoints_with_descriptors(input_tensor, keep_as_templet=False)`

If you intend to use the SIFT algorithm as a template for matching with other images, set keep_as_template to True.

Optionally, you can release the captured template to free up memory.

## Helper Functions and Classes
1. `load_image(name, color_mode='grayscale')`:
This function loads an image from the specified file path and returns it as a TensorFlow tensor. 
It accepts an optional `color_mode` parameter to specify the color mode of the image (default is 'grayscale').

2. `templet_matching_TF(scr_kp, dst_kp, scr_dsc, dst_dsc, ratio_threshold=0.7)`:
This function performs template matching using TensorFlow. It matches keypoints (scr_kp and dst_kp) based on 
their descriptors (scr_dsc and dst_dsc) and a specified ratio_threshold. It returns the matched source and destination keypoints.

3. `templet_matching_CV2(scr_kp, dst_kp, scr_dsc, dst_dsc, ratio_threshold=0.7)`:
This function performs template matching using OpenCV (cv2). 

4. KeyPoints Class:
The KeyPoints class represents keypoints detected in an image. It provides methods for manipulating and working with keypoints.

5. Octave Class: 
The Octave class represents an octave in the Scale-Invariant Feature Transform (SIFT) algorithm. 
It stores Gaussian scale-space images and provides methods for working with them.

## **Customization**
You can customize the SIFT algorithm by adjusting the parameters when initializing the SIFT object. 

Experiment with different parameter values to optimize the performance for your specific use case.

## **Acknowledgments**

This SIFT algorithm implementation is based on the original SIFT paper by David G. Lowe: Distinctive Image Features from Scale-Invariant Keypoints.

## ____________
Notably, it runs in around 5-7 seconds on CPU (depend on the size of the key points slower than OpenCV but still efficient) and faster with GPU acceleration.