import tensorflow as tf
from typing import Union, TypeVar
from collections import namedtuple

PI = tf.cast(
    tf.math.angle(tf.constant(-1, dtype=tf.complex64)), tf.float32
)

KT = TypeVar('KT', bound='KeyPoints')
unpacked_octave = namedtuple('unpacked_octave', 'octave, layer, scale')


class Octave:
    def __init__(self, index: int, gss: tf.Tensor):
        self.__shape = gss.get_shape().as_list()
        self.index = index
        self.gss = gss
        self.dx, self.dy, self.magnitude, self.orientation = compute_grad_mag_ori(gss)

    @property
    def shape(self):
        return self.__shape


class KeyPoints:
    def __init__(self, pt: Union[None, tf.Tensor] = None, size: Union[None, tf.Tensor] = None,
                 angle: Union[None, tf.Tensor] = None, octave: Union[None, tf.Tensor] = None,
                 response: Union[None, tf.Tensor] = None, as_image_size: bool = False):
        self.scale_index = None
        self.__shape = (0, )
        self.pt = tf.constant([[]], shape=(0, 3), dtype=tf.float32)
        self.size = tf.constant([[]], shape=(0, 1), dtype=tf.float32)
        self.angle = tf.constant([[]], shape=(0, 1), dtype=tf.float32)
        self.octave = tf.constant([[]], shape=(0, 1), dtype=tf.float32)
        self.response = tf.constant([[]], shape=(0, 1), dtype=tf.float32)
        self.as_image_size = as_image_size
        if pt is not None: self.__constructor(pt, size, angle, octave, response)

    def __add__(self, other: KT) -> KT:
        if not isinstance(other, KeyPoints): raise TypeError
        if self.as_image_size ^ other.as_image_size: raise ValueError('the as_image_size parameter not inconsistent')
        if (self.scale_index is None) ^ (other.scale_index is None):
            if self.scale_index is None:
                self.scale_index = tf.ones((self.shape[0], 1), tf.float32) * -1
            else:
                other.scale_index = tf.ones((other.shape[0], 1), tf.float32) * -1
        ints = self.from_array(tf.concat((self.as_array(), other.as_array()), axis=0), inplace=False)
        return ints

    def __iadd__(self, other: KT) -> KT:
        if not isinstance(other, KeyPoints): raise TypeError
        if self.as_image_size ^ other.as_image_size: raise ValueError('the as_image_size parameter not inconsistent')
        if (self.scale_index is None) ^ (other.scale_index is None):
            if self.scale_index is None:
                self.scale_index = tf.ones((self.shape[0], 1), tf.float32) * -1
            else:
                other.scale_index = tf.ones((other.shape[0], 1), tf.float32) * -1
        self.from_array(tf.concat((self.as_array(), other.as_array()), axis=0), inplace=True)
        return self

    def __constructor(self, pt, size, angle, octave, response):
        if not isinstance(pt, tf.Tensor): raise ValueError('All the fields need to be type of tf.Tensor')
        _shape = pt.get_shape().as_list()
        if len(_shape) > 2:
            pt = tf.squeeze(pt)
            _shape = pt.get_shape().as_list()
        if len(_shape) != 2 or _shape[-1] < 3: raise ValueError(
            'expected "pt" to be 2D tensor with size of (None, 3 or 4)')
        if _shape[-1] == 4:
            pt, scale_index = tf.split(pt, [3, 1], axis=-1)
        else:
            scale_index = None
        valid = [pt]
        for f in [size, angle, octave, response]:
            if not isinstance(f, tf.Tensor): raise ValueError('All the fields need to be type of tf.Tensor')
            f = tf.reshape(f, (-1, 1))
            if f.get_shape()[0] != _shape[0]: raise ValueError('All the fields need to be with the same first dim size')
            valid.append(f)
        self.pt, self.size, self.angle, self.octave, self.response = valid
        self.scale_index = scale_index
        self.__shape = (_shape[0], )

    @property
    def shape(self) -> tuple:
        return self.__shape

    def as_array(self) -> tf.Tensor:
        _array = [self.pt]
        _array += [self.scale_index] if self.scale_index is not None else []
        _array += [self.size, self.angle, self.octave, self.response]
        return tf.concat(_array, axis=-1)

    def from_array(self, array: tf.Tensor, inplace=False) -> Union[None, KT]:
        _shape = array.get_shape().as_list()
        if len(_shape) != 2 or _shape[1] < 7: raise ValueError('array rank need to be 2 with size of (None, 7 or 8)')
        splits = [4] if _shape[1] == 8 else [3]
        splits += [1, 1, 1, 1]
        split = tf.split(array, splits, axis=-1)
        if not inplace: return KeyPoints(*split, as_image_size=self.as_image_size)
        self.__constructor(*split)

    def to_image_size(self, inplace=False) -> Union[None, KT]:
        if self.shape[0] == 0 or self.as_image_size: return self if not inplace else None
        pt_unpack = self.pt * tf.constant([1.0, 0.5, 0.5], dtype=tf.float32)
        size_unpack = self.size * 0.5
        octave_unpack = tf.cast(self.octave, dtype=tf.int64) ^ 255
        if inplace:
            self.pt, self.size, self.octave = pt_unpack, size_unpack, octave_unpack
            self.as_image_size = True
            return
        if self.scale_index is not None: pt_unpack = tf.concat((pt_unpack, self.scale_index), -1)
        unpack_key_points = KeyPoints(
            pt_unpack, size_unpack, self.angle, tf.cast(octave_unpack, dtype=tf.float32), self.response, True
        )
        return unpack_key_points

    def relies_scale_index(self):
        self.scale_index = None

    def unpack_octave(self) -> unpacked_octave:
        if self.shape[0] == 0: return unpacked_octave(None, None, None)
        up_key_points = self.to_image_size(inplace=False) if not self.as_image_size else self

        octave_unpack = tf.cast(up_key_points.octave, tf.int64)
        octave = octave_unpack & 255
        octave = (octave ^ 255) - 1

        layer = tf.bitwise.left_shift(octave_unpack, 8)
        layer = layer & 255

        scale = tf.where(
            octave >= 0, tf.cast(1 / tf.bitwise.left_shift(1, octave), dtype=tf.float32),
            tf.cast(tf.bitwise.left_shift(1, -octave), dtype=tf.float32)
        )
        octave = octave + 1
        return unpacked_octave(tf.cast(octave, dtype=tf.float32), tf.cast(layer, dtype=tf.float32), scale)

    def unpack_batch(self) -> list[KT]:
        if self.shape[0] == 0: return None
        part = tf.reshape(tf.cast(tf.split(self.pt, [1, 2], -1)[0], tf.int32), (-1,))
        part = tf.dynamic_partition(self.as_array(), part, tf.reduce_max(part) + 1)
        out = [self.from_array(p, inplace=False) for p in part]
        return out


def gaussian_kernel(
        kernel_size: int,
        sigma: Union[None, float] = None
) -> tf.Tensor:
    if kernel_size == 0 and (sigma is None or sigma < 0.8):
        if sigma is None:
            raise ValueError('need sigma parameter when the kernel size is 0')
        raise ValueError('minimum kernel need to be size of 3 --> sigma > 0.8')

    if kernel_size == 0:
        kernel_size = ((((sigma - 0.8) / 0.3) + 1) * 2) + 1
        kernel_size = kernel_size + 1 if (kernel_size % 2) == 0 else kernel_size

    assert kernel_size % 2 != 0 and kernel_size > 2

    if sigma is None:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = tf.meshgrid(ax, ax)
    normal = 1 / (2.0 * PI * (sigma ** 2))
    kernel = tf.exp(
        -((xx ** 2) + (yy ** 2)) / (2.0 * (sigma ** 2))
    ) * normal
    return kernel / tf.reduce_sum(kernel)


def make_neighborhood2D(
        init_cords: tf.Tensor,
        con: int = 3,
        origin_shape: Union[None, tuple, list, tf.TensorShape] = None
) -> tf.Tensor:
    B, ndim = init_cords.get_shape()
    con = int(con)
    assert ndim == 4

    ax = tf.range(-con // 2 + 1, (con // 2) + 1, dtype=tf.int64)

    con_kernel = tf.stack(tf.meshgrid(ax, ax)[::-1], axis=-1)

    con_kernel = tf.reshape(con_kernel, shape=(1, con ** 2, 2))

    b, yx, d = tf.split(init_cords, [1, 2, 1], axis=1)
    yx = yx[:, tf.newaxis, ...]

    yx = yx + con_kernel

    b = tf.repeat(b[:, tf.newaxis, ...], repeats=con ** 2, axis=1)
    d = tf.repeat(d[:, tf.newaxis, ...], repeats=con ** 2, axis=1)

    neighbor = tf.concat((b, yx, d), axis=-1)
    if origin_shape is None:
        return neighbor

    assert len(origin_shape) == 4
    neighbor = neighbor + 1
    b, y, x, d = tf.unstack(neighbor, num=4, axis=-1)

    y_cast = tf.logical_and(tf.math.greater_equal(y, 1), tf.math.less_equal(y, origin_shape[1]))
    x_cast = tf.logical_and(tf.math.greater_equal(x, 1), tf.math.less_equal(x, origin_shape[2]))

    valid = tf.cast(tf.logical_and(y_cast, x_cast), dtype=tf.int32)
    valid = tf.math.reduce_prod(valid, axis=-1)
    cords_valid = tf.where(valid == 1)
    neighbor = tf.gather_nd(neighbor, cords_valid) - 1
    return neighbor


def compute_extrema3D(
        X: tf.Tensor,
        threshold: Union[tf.Tensor, float, None] = None,
        con: Union[tf.Tensor, int, tuple, list] = 3,
        border_width: Union[tf.Tensor, tuple, list, None] = None,
        epsilon: Union[tf.Tensor, float] = 1e-07
) -> tf.Tensor:
    _shape = tf.shape(X)
    _n_dims = len(_shape)
    if _n_dims != 4:
        raise ValueError(
            'expected the inputs to be 4D tensor with size of (None, H, W, C)'
        )
    b, h, w, d = tf.unstack(tf.cast(_shape, dtype=tf.int64), num=4, axis=-1)

    X = tf.cast(X, dtype=tf.float32)

    threshold = tf.cast(threshold, dtype=tf.float32) if threshold is not None else None

    if tf.is_tensor(con):
        con = tf.get_static_value(tf.reshape(con, shape=(-1,)))
        con = tuple(con) if len(con) != 1 else int(con)

    if isinstance(con, int):
        con = (con, con, con)

    if len(con) > 3:
        raise ValueError('con parameter need to be int or iterable with size 3')

    half_con = [c // 2 for c in con]

    x_con = tf.concat((tf.expand_dims(X, -1), tf.expand_dims(X, -1) * -1.), -1)

    extrema = tf.nn.max_pool3d(x_con, ksize=con, strides=[1, 1, 1], padding='VALID')

    extrema_max, extrema_min = tf.unstack(extrema, 2, -1)
    extrema_min = extrema_min * -1.

    compare_array = tf.slice(X, [0, *half_con], [b, h - 2 * half_con[0], w - 2 * half_con[1], d - 2 * half_con[2]])

    def _equal_with_epsilon(arr):
        return tf.logical_and(
            tf.math.greater_equal(arr, compare_array - epsilon),
            tf.math.less_equal(arr, compare_array + epsilon)
        )

    extrema_cond = tf.logical_or(
        _equal_with_epsilon(extrema_max),
        _equal_with_epsilon(extrema_min)
    )
    if threshold is not None:
        extrema_cond = tf.logical_and(extrema_cond, tf.math.greater(tf.abs(compare_array), threshold))

    byxd = tf.where(extrema_cond)

    byxd = byxd + tf.constant([[0] + half_con], dtype=tf.int64)

    if border_width is not None:
        if tf.is_tensor(border_width):
            border_width = tf.get_static_value(tf.reshape(border_width, shape=(-1,)))
            border_width = tuple(border_width)
        if len(border_width) != 3:
            raise ValueError('border_width need to be with len of 3')
        cb, cy, cx, cd = tf.unstack(byxd, num=4, axis=-1)
        by, bx, bd = tf.unstack(tf.cast(border_width, dtype=tf.int64), num=3)
        y_cond = tf.logical_and(tf.math.greater_equal(cy, by), tf.math.less_equal(cy, h - by))
        x_cond = tf.logical_and(tf.math.greater_equal(cx, bx), tf.math.less_equal(cx, w - bx))
        d_cond = tf.logical_and(tf.math.greater_equal(cd, bd), tf.math.less_equal(cd, d - bd))

        casted_ = tf.logical_and(tf.logical_and(y_cond, x_cond), d_cond)
        byxd = tf.boolean_mask(byxd, casted_)

    return byxd


def compute_central_gradient3D(
        X: tf.Tensor
) -> tf.Tensor:
    _shape = tf.shape(X)
    _n_dims = len(_shape)
    if _n_dims != 4:
        raise ValueError(
            'expected the inputs to be 4D tensor with size of (None, H, W, C)'
        )

    X = tf.cast(X, dtype=tf.float32)

    kx = tf.constant([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=tf.float32)
    kx = tf.pad(
        tf.reshape(kx, shape=(3, 3, 1, 1, 1)),
        paddings=tf.constant([[0, 0], [0, 0], [1, 1], [0, 0], [0, 0]]),
        constant_values=0.0
    )
    ky = tf.constant([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=tf.float32)
    ky = tf.pad(
        tf.reshape(ky, shape=(3, 3, 1, 1, 1)),
        paddings=tf.constant([[0, 0], [0, 0], [1, 1], [0, 0], [0, 0]]),
        constant_values=0.0
    )
    kz = tf.zeros_like(kx)
    kz = tf.tensor_scatter_nd_update(kz, tf.constant([[1, 1, 0, 0, 0], [1, 1, 2, 0, 0]]), tf.constant([-1.0, 1.0]))

    kernels_dx = tf.concat((kx, ky, kz), axis=-1)

    X = tf.expand_dims(X, axis=-1)
    grad = tf.nn.convolution(X, kernels_dx, padding='VALID') * 0.5
    return grad


def compute_hessian_3D(
        X: tf.Tensor
) -> tf.Tensor:
    _shape = tf.shape(X)
    _n_dims = len(_shape)
    if _n_dims != 4:
        raise ValueError(
            'expected the inputs to be 4D tensor with size of (None, H, W, C)'
        )

    X = tf.cast(X, dtype=tf.float32)

    dxx = tf.constant([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]], dtype=tf.float32)
    dxx = tf.pad(
        tf.reshape(dxx, shape=(3, 3, 1, 1, 1)),
        paddings=tf.constant([[0, 0], [0, 0], [1, 1], [0, 0], [0, 0]]),
        constant_values=0.0
    )
    dyy = tf.constant([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]], dtype=tf.float32)
    dyy = tf.pad(
        tf.reshape(dyy, shape=(3, 3, 1, 1, 1)),
        paddings=tf.constant([[0, 0], [0, 0], [1, 1], [0, 0], [0, 0]]),
        constant_values=0.0
    )
    dzz = tf.zeros_like(dxx)
    dzz = tf.tensor_scatter_nd_update(
        dzz, tf.constant([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 2, 0, 0]]), tf.constant([1.0, -2.0, 1.0])
    )

    kww = tf.concat((dxx, dyy, dzz), axis=-1)

    dxy = tf.constant([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]], dtype=tf.float32)
    dxy = tf.pad(
        tf.reshape(dxy, shape=(3, 3, 1, 1, 1)),
        paddings=tf.constant([[0, 0], [0, 0], [1, 1], [0, 0], [0, 0]]),
        constant_values=0.0
    )

    dxz = tf.zeros_like(dxy)
    dxz = tf.tensor_scatter_nd_update(
        dxz,
        tf.constant([[1, 0, 0, 0, 0], [1, 2, 2, 0, 0], [1, 0, 2, 0, 0], [1, 2, 0, 0, 0]]),
        tf.constant([1.0, 1.0, -1.0, -1.0])
    )

    dyz = tf.zeros_like(dxy)
    dyz = tf.tensor_scatter_nd_update(
        dyz,
        tf.constant([[0, 1, 0, 0, 0], [2, 1, 2, 0, 0], [0, 1, 2, 0, 0], [2, 1, 0, 0, 0]]),
        tf.constant([1.0, 1.0, -1.0, -1.0])
    )

    kws = tf.concat((dxy, dyz, dxz), axis=-1)

    X = tf.expand_dims(X, axis=-1)

    dFww = tf.nn.convolution(X, kww, padding='VALID')

    dFws = tf.nn.convolution(X, kws, padding='VALID') * 0.25

    dxx, dyy, dzz = tf.unstack(dFww, 3, axis=-1)
    dxy, dyz, dxz = tf.unstack(dFws, 3, axis=-1)
    hessian_mat = tf.stack(
        (
            tf.stack((dxx, dxy, dxz), axis=-1),
            tf.stack((dxy, dyy, dyz), axis=-1),
            tf.stack((dxz, dyz, dzz), axis=-1)
        ), axis=-1
    )
    return hessian_mat


def compute_grad_mag_ori(
        gss: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    kx = tf.constant([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], shape=(3, 3, 1, 1, 1), dtype=tf.float32)
    ky = tf.constant([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], shape=(3, 3, 1, 1, 1), dtype=tf.float32)
    gradient_kernel = tf.concat((kx, ky), axis=-1)

    gradient = tf.nn.convolution(tf.expand_dims(gss, -1), gradient_kernel, padding='VALID')
    dx, dy = tf.unstack(gradient, 2, axis=-1)

    magnitude = tf.math.sqrt(dx * dx + dy * dy)
    orientation = tf.math.atan2(dy, dx) * (180.0 / PI)

    return dx, dy, magnitude, orientation
