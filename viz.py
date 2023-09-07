from matplotlib import pyplot as plt
import matplotlib
import tensorflow as tf
import numpy as np

matplotlib.use("Qt5Agg")


def show(image, ax):
    if tf.is_tensor(image):
        image = image.numpy()
    if image.max() > 1:
        image = image.astype('uint8')
    if len(image.shape) == 2 or image.shape[-1] == 1:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)


def show_images(images, subplot_y=None, subplot_x=None):
    assert isinstance(images, (np.ndarray, tf.Tensor, list, tuple))
    subplot_x = min([len(images), 4]) if subplot_x is None else subplot_x
    subplot_y = len(images) // subplot_x if subplot_y is None else subplot_y

    fig, _ = plt.subplots(subplot_x, subplot_y, subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(wspace=0, hspace=0.05)

    for i in range(min([subplot_x * subplot_y, len(images)])):
        show(images[i], fig.axes[i])


