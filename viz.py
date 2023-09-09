from matplotlib import pyplot as plt
import matplotlib
import tensorflow as tf
import numpy as np
from utils import KT
from typing import Union
import cv2

matplotlib.use("Qt5Agg")


def show(
        image: Union[np.ndarray, tf.Tensor],
        ax: plt.Axes
):
    if tf.is_tensor(image):
        image = image.numpy()
    assert isinstance(image, np.ndarray)
    if image.max() > 1:
        image = image.astype('uint8')
    if len(image.shape) == 2 or image.shape[-1] == 1:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)


def show_images(
        images: Union[np.ndarray, tf.Tensor],
        subplot_y: Union[None, int],
        subplot_x: Union[None, int]
):
    assert isinstance(images, (np.ndarray, tf.Tensor, list, tuple))
    subplot_x = min([len(images), 4]) if subplot_x is None else subplot_x
    subplot_y = len(images) // subplot_x if subplot_y is None else subplot_y

    fig, _ = plt.subplots(subplot_x, subplot_y, subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(wspace=0, hspace=0.05)

    for i in range(min([subplot_x * subplot_y, len(images)])):
        show(images[i], fig.axes[i])


def show_key_points(
        key_points: KT,
        img: tf.Tensor
):
    if not isinstance(key_points, type(KT)): raise TypeError('Key points need to be of type "KeyPoints"')
    key_points = key_points.to_image_size()

    if key_points.n_batches() > 1: raise ValueError("number of batches in the key points > 1")
    if not isinstance(img, tf.Tensor): raise TypeError('image need to be of type "Tensor"')

    shape = img.get_shape().as_list()

    if shape[0] > 1: raise ValueError("number of batches in the image > 1")
    if not (shape[-1] == 1 or shape[-1] == 3):
        raise ValueError("image need to be with 3 channels (RGB) or gray level with one channel")

    points = tf.concat((key_points.pt, tf.zeros((key_points.shape[0], 1), tf.float32)), -1)
    points = tf.cast(points, tf.int32)

    cross = [
        [0, -2, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -2, 0],
        [0, 0, -1, 0],
        [0, 0, 1, 0],
        [0, 0, 2, 0],
        [0, 1, 0, 0],
        [0, 2, 0, 0]
    ]

    cross = tf.constant(cross, shape=(1, 8, 4), dtype=tf.int32)
    neighbor = cross + tf.expand_dims(points, 1)
    neighbor = tf.reshape(neighbor, (-1, 4))

    _, y, x, _ = tf.unstack(neighbor, 4, -1)
    mask = tf.where((y > 0) & (y < shape[1]) & (x > 0) & (x < shape[2]), True, False)
    neighbor = tf.boolean_mask(neighbor, mask)

    kpt_image = tf.zeros([*shape[:-1], 1], dtype=img.dtype)
    kpt_image = tf.tensor_scatter_nd_add(kpt_image, neighbor, tf.ones((neighbor.get_shape()[0],), tf.float32))
    kpt_image = tf.where(kpt_image > 0, 1, 0)
    kpt_image = tf.concat((kpt_image * 242, kpt_image * 140, kpt_image * 40), -1)

    if shape[-1] == 1:
        img_del = tf.tensor_scatter_nd_update(img, neighbor, tf.zeros((neighbor.get_shape()[0],), tf.float32))
        img_del = tf.cast(tf.repeat(img_del, 3, -1), tf.int32)
    else:
        img_del = img
        for _ in range(3):
            img_del = tf.tensor_scatter_nd_update(img_del, neighbor, tf.zeros((neighbor.get_shape()[0],), tf.float32))
            neighbor = neighbor + tf.constant([0, 0, 0, 1], dtype=tf.int32)

    mark = kpt_image + img_del
    show_images(tf.cast(mark, tf.uint8), 1, 1)


def _make_line(
        x0: Union[tf.Tensor, int, float],
        x1: Union[tf.Tensor, int, float],
        y0: Union[tf.Tensor, int, float],
        y1: Union[tf.Tensor, int, float],
        h_limit: Union[tf.Tensor, int],
        w_limit: Union[tf.Tensor, int]
):
    m = (y1 - y0) / (x1 - x0)
    x = tf.range(x0, x1 + 1, dtype=tf.float32)
    w = tf.sqrt(1 + tf.math.abs(m)) / 2
    y = x * m + (x1 * y0 - x0 * y1) / (x1 - x0)

    t = tf.math.ceil(w / 2)

    yy = (tf.reshape(tf.math.floor(y), [-1, 1]) + tf.reshape(tf.range(-t - 1, t + 2, dtype=tf.float32), [1, -1]))
    xx = tf.repeat(x, yy.get_shape()[1])

    v = tf.clip_by_value(
        tf.minimum(yy + 1 + 1 / 2 - tf.reshape(y, (-1, 1)), -yy + 1 + 1 / 2 + tf.reshape(y, (-1, 1))), 0, 1
    )
    v = tf.reshape(v, (-1,))
    yy = tf.reshape(yy, (-1,))

    limits = tf.where((yy >= 0) & (xx >= 0) & (yy < h_limit) & (xx < w_limit) & (v == 1.), True, False)

    xx = tf.boolean_mask(xx, limits)
    yy = tf.boolean_mask(yy, limits)
    v = tf.boolean_mask(v, limits)

    cords = tf.cast(tf.stack((yy, xx), -1), tf.int32)
    cords = tf.pad(cords, [[0, 0], [1, 1]])
    return cords, v


def plot_matches_TF(
        scr_img: tf.Tensor,
        dst_img: tf.Tensor,
        src_pt: tf.Tensor,
        dst_pt: tf.Tensor
) -> tf.Tensor:
    if not isinstance(scr_img, tf.Tensor) or not isinstance(dst_img, tf.Tensor):
        raise TypeError('descriptors need to be of type "Tensor"')
    if not isinstance(src_pt, tf.Tensor) or not isinstance(dst_pt, tf.Tensor):
        raise TypeError('points need to be of type "Tensor"')
    if src_pt.get_shape()[0] != dst_pt.get_shape()[0]:
        raise ValueError('points need to be with the same size')

    _, h_scr, w_scr, c_scr = scr_img.get_shape().as_list()
    _, h_dst, w_dst, c_dst = dst_img.get_shape().as_list()
    if not (c_scr == 1 or c_scr == 3) or not (c_dst == 1 or c_dst == 3):
        raise ValueError("images need to be with 3 channels (RGB) or gray level with one channel")

    if c_scr == 1:
        scr_img = tf.repeat(scr_img, 3, -1)
    if c_dst == 1:
        dst_img = tf.repeat(dst_img, 3, -1)

    h_new = max(h_scr, h_dst)
    w_new = w_scr + w_dst

    h_diff = h_new - min(h_scr, h_dst)
    h_up = h_diff // 2
    h_down = h_diff - h_up

    if h_scr < h_dst:
        scr_img = tf.pad(scr_img, [[0, 0], [h_up, h_down], [0, 0], [0, 0]])
    elif h_scr > h_dst:
        dst_img = tf.pad(dst_img, [[0, 0], [h_up, h_down], [0, 0], [0, 0]])

    marked_image = tf.concat((scr_img, dst_img), 2)

    src_b, src_y, src_x = tf.unstack(src_pt, 3, -1)
    dst_b, dst_y, dst_x = tf.unstack(dst_pt, 3, -1)

    if h_scr < h_dst:
        src_y = src_y + h_up
    elif h_scr > h_dst:
        dst_y = dst_y + h_up

    dst_x = dst_x + w_scr
    lines = tf.zeros([*marked_image.get_shape()[:-1], 1])

    for y1, x1, y2, x2 in zip(src_y, src_x, dst_y, dst_x):
        c, val = _make_line(x1, x2, y1, y2, h_new, w_new)
        lines = tf.tensor_scatter_nd_update(lines, c, val)

    lines = tf.concat((lines * 9, lines * 121, lines * 105), -1)

    marked_image = tf.where(lines > 0, lines, marked_image)
    show_images(tf.cast(marked_image, tf.uint8), 1, 1)
    return marked_image


def plot_matches_CV2(
        scr_img: tf.Tensor,
        dst_img: tf.Tensor,
        src_pt: KT,
        dst_pt: KT
) -> tf.Tensor:
    if not isinstance(scr_img, tf.Tensor) or not isinstance(dst_img, tf.Tensor):
        raise TypeError('descriptors need to be of type "Tensor"')
    if not isinstance(src_pt, tf.Tensor) or not isinstance(dst_pt, tf.Tensor):
        raise TypeError('points need to be of type "Tensor"')
    if src_pt.get_shape()[0] != dst_pt.get_shape()[0]:
        raise ValueError('points need to be with the same size')
    _, h_scr, w_scr, c_scr = scr_img.get_shape().as_list()
    _, h_dst, w_dst, c_dst = dst_img.get_shape().as_list()

    if not (c_scr == 1 or c_scr == 3) or not (c_dst == 1 or c_dst == 3):
        raise ValueError("images need to be with 3 channels (RGB) or gray level with one channel")

    if c_scr == 1:
        scr_img = tf.repeat(scr_img, 3, -1)
    if c_dst == 1:
        dst_img = tf.repeat(dst_img, 3, -1)

    h_new = max(h_scr, h_dst)
    h_diff = h_new - min(h_scr, h_dst)
    h_up = h_diff // 2
    h_down = h_diff - h_up

    if h_scr < h_dst:
        scr_img = tf.pad(scr_img, [[0, 0], [h_up, h_down], [0, 0], [0, 0]])
    elif h_scr > h_dst:
        dst_img = tf.pad(dst_img, [[0, 0], [h_up, h_down], [0, 0], [0, 0]])

    marked_image = tf.concat((scr_img, dst_img), 2)
    marked_image = tf.squeeze(marked_image).numpy().astype('uint8')

    _, src_y, src_x = tf.unstack(src_pt, 3, -1)
    _, dst_y, dst_x = tf.unstack(dst_pt, 3, -1)

    if h_scr < h_dst:
        src_y = src_y + h_up
    elif h_scr > h_dst:
        dst_y = dst_y + h_up

    dst_x = dst_x + w_scr
    src_pt = tf.stack((src_y, src_x), -1).numpy().astype(int)
    dst_pt = tf.stack((dst_y, dst_x), -1).numpy().astype(int)

    for i in range(src_pt.shape[0]):
        pt1 = (int(src_pt[i, 1]), int(src_pt[i, 0]))
        pt2 = (int(dst_pt[i, 1]), int(dst_pt[i, 0]))
        cv2.line(marked_image, pt1, pt2, (9, 121, 105))

    show_images([marked_image], 1, 1)
    return tf.constant(marked_image, shape=(1, marked_image.shape[0], marked_image.shape[1], 3))
