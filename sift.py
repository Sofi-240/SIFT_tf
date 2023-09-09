from typing import Union
import tensorflow as tf
from utils import PI, gaussian_kernel, compute_extrema3D, load_image, templet_matching_TF, templet_matching_CV2, \
    make_neighborhood2D, compute_central_gradient3D, compute_hessian_3D, Octave, KeyPoints
from tensorflow.python.keras import backend
from viz import show_key_points, show_images, plot_matches_TF, plot_matches_CV2

# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

backend.set_floatx('float32')
linalg_ops = tf.linalg
math_ops = tf.math
bitwise_ops = tf.bitwise
image_ops = tf.image


class SIFT:
    def __init__(self, sigma: float = 1.6, assume_blur_sigma: float = 0.5, n_intervals: int = 3,
                 n_octaves: Union[int, None] = None, border_width: int = 5, convergence_iter: int = 5):
        self.sigma = sigma
        self.assume_blur_sigma = assume_blur_sigma
        self.n_intervals = n_intervals
        self.n_octaves = n_octaves
        self.border_width = border_width
        self.convergence_N = convergence_iter
        self.octave_pyramid: list[Octave] = []

    def __init_graph(self, inputs: tf.Tensor) -> tuple[tf.Tensor, list[tf.Tensor]]:
        if not isinstance(inputs, tf.Tensor): raise ValueError('Input image need to be of type Tensor')

        _shape = inputs.get_shape().as_list()
        if len(_shape) != 4 or _shape[-1] != 1:
            raise ValueError('expected the inputs to be grayscale images with size of (None, h, w, 1)')

        inputs = tf.cast(inputs, dtype=tf.float32)
        _, h_, w_, _ = _shape

        kernels = self.__pyramid_kernels()

        min_shape = int(kernels[-1].get_shape()[0])
        s_ = tf.cast(min([h_ * 2, w_ * 2]), dtype=tf.float32)
        diff = math_ops.log(s_)
        if min_shape > 1: diff = diff - math_ops.log(tf.cast(min_shape, dtype=tf.float32))
        max_n_octaves = int(tf.round(diff / math_ops.log(2.0)) + 1)

        if self.n_octaves is not None and max_n_octaves > self.n_octaves: max_n_octaves = self.n_octaves
        self.n_octaves = max_n_octaves
        return inputs, kernels

    def __pyramid_kernels(self) -> list[tf.Tensor]:
        delta_sigma = (self.sigma ** 2) - ((2 * self.assume_blur_sigma) ** 2)
        delta_sigma = math_ops.sqrt(tf.maximum(delta_sigma, 0.64))

        base_kernel = gaussian_kernel(kernel_size=0, sigma=delta_sigma)
        base_kernel = tf.expand_dims(tf.expand_dims(base_kernel, axis=-1), axis=-1)

        images_per_octaves = self.n_intervals + 3
        K = 2 ** (1 / self.n_intervals)
        K = tf.cast(K, dtype=tf.float32)

        kernels = [base_kernel]

        for i in range(1, images_per_octaves):
            s_prev = self.sigma * (K ** (i - 1))
            s = math_ops.sqrt((K * s_prev) ** 2 - s_prev ** 2)
            kernel_ = gaussian_kernel(kernel_size=0, sigma=s)
            kernels.append(tf.expand_dims(tf.expand_dims(kernel_, axis=-1), axis=-1))
        return kernels

    def __assign_descriptors(self, descriptors: tf.Tensor, bins: tf.Tensor, magnitude: tf.Tensor) -> tf.Tensor:
        N_bins, window_width = 8, 4
        _, y, x, _ = tf.unstack(bins, 4, -1)
        mask = tf.where((y > -1) & (y < window_width) & (x > -1) & (x < window_width), True, False)
        magnitude = tf.boolean_mask(magnitude, mask)

        b, y, x, z = tf.unstack(tf.boolean_mask(bins, mask), 4, -1)

        while tf.reduce_min(z) < 0:
            z = tf.where(z < 0, z + N_bins, z)

        while tf.reduce_max(z) >= N_bins:
            z = tf.where(z >= N_bins, z - N_bins, z)

        bin_floor = [b] + [tf.round(tf.floor(h)) for h in [y, x, z]]
        bin_frac = [tf.reshape(h - hf, (-1,)) for h, hf in zip([y, x, z], bin_floor[1:])]

        y, x, z = bin_frac

        _C0 = magnitude * (1 - y)
        _C1 = magnitude * y

        # interpolation in x direction
        _C00 = _C0 * (1 - x)
        _C01 = _C0 * x

        _C10 = _C1 * (1 - x)
        _C11 = _C1 * x

        # interpolation in z direction
        _C000 = _C00 * (1 - z)
        _C001 = _C00 * z
        _C010 = _C01 * (1 - z)
        _C011 = _C01 * z
        _C100 = _C10 * (1 - z)
        _C101 = _C10 * z
        _C110 = _C11 * (1 - z)
        _C111 = _C11 * z

        b, y, x, z = [tf.cast(c, tf.int32) for c in bin_floor]
        descriptors = tf.tensor_scatter_nd_add(descriptors, tf.stack((b, y + 1, x + 1, z), -1), _C000)
        descriptors = tf.tensor_scatter_nd_add(descriptors, tf.stack((b, y + 1, x + 1, (z + 1) % N_bins), -1), _C001)

        descriptors = tf.tensor_scatter_nd_add(descriptors, tf.stack((b, y + 1, x + 2, z), -1), _C010)
        descriptors = tf.tensor_scatter_nd_add(descriptors, tf.stack((b, y + 1, x + 2, (z + 1) % N_bins), -1), _C011)

        descriptors = tf.tensor_scatter_nd_add(descriptors, tf.stack((b, y + 2, x + 1, z), -1), _C100)
        descriptors = tf.tensor_scatter_nd_add(descriptors, tf.stack((b, y + 2, x + 1, (z + 1) % N_bins), -1), _C101)

        descriptors = tf.tensor_scatter_nd_add(descriptors, tf.stack((b, y + 2, x + 2, z), -1), _C110)
        descriptors = tf.tensor_scatter_nd_add(descriptors, tf.stack((b, y + 2, x + 2, (z + 1) % N_bins), -1), _C111)

        return descriptors

    def __descriptors_per_octave(self, octave: Octave, key_points: KeyPoints) -> tf.Tensor:
        scale_multiplier, window_width, N_bins, descriptor_max_value = 3, 4, 8, 0.2
        bins_per_degree = N_bins / 360.
        weight_multiplier = -1.0 / (0.5 * window_width * window_width)
        descriptors = tf.zeros((key_points.shape[0], window_width + 2, window_width + 2, N_bins), tf.float32)

        key_points = key_points.to_image_size()
        unpack_oct = key_points.unpack_octave()

        scale_ = tf.pad(tf.repeat(unpack_oct.scale, 2, axis=1), tf.constant([[0, 0], [1, 0]]), constant_values=1.0)
        points = tf.round(tf.concat((key_points.pt * scale_, unpack_oct.layer), -1))
        histogram_width = scale_multiplier * 0.5 * unpack_oct.scale * key_points.size
        radius = tf.round(histogram_width * math_ops.sqrt(2.0) * (window_width + 1.0) * 0.5)

        _, y, x, _ = tf.split(points, [1] * 4, -1)
        radius = math_ops.minimum(
            math_ops.minimum(octave.shape[1] - 3 - y, octave.shape[2] - 3 - x),
            math_ops.minimum(math_ops.minimum(y, x), radius)
        )
        radius = tf.reshape(radius, (-1,))
        parallel = tf.unique(radius)

        indexes = tf.dynamic_partition(tf.reshape(tf.range(key_points.shape[0], dtype=tf.int32), (-1, 1)),
                                       parallel.idx, tf.reduce_max(parallel.idx) + 1)

        wrap = tf.concat((points, key_points.angle, histogram_width), -1)
        wrap = tf.dynamic_partition(wrap, parallel.idx, parallel.y.get_shape()[0])

        M = octave.magnitude
        T = octave.orientation % 360.0

        for index, wrap_i, r in zip(indexes, wrap, parallel.y):
            points, angle, width = tf.split(wrap_i, [4, 1, 1], -1)
            angle = 360.0 - angle
            n = points.get_shape()[0]
            cos = math_ops.cos((PI / 180) * angle)
            sin = math_ops.sin((PI / 180) * angle)

            neighbor = make_neighborhood2D(tf.constant([[0, 0, 0, 0]], dtype=tf.int64), con=(r * 2) + 1)
            block = tf.expand_dims(tf.cast(points, tf.int64), axis=1) + neighbor

            neighbor = tf.cast(tf.repeat(tf.split(neighbor, [1, 2, 1], -1)[1], n, 0), tf.float32)
            y, x = tf.unstack(neighbor, 2, -1)
            b = tf.cast(tf.ones(y.get_shape(), dtype=tf.int32) * index, tf.float32)

            rotate = [((x * sin) + (y * cos)) / width, ((x * cos) - (y * sin)) / width]
            weight = tf.reshape(math_ops.exp(weight_multiplier * (rotate[0] ** 2 + rotate[1] ** 2)), (-1,))

            magnitude = tf.gather_nd(M, tf.reshape(block, (-1, 4))) * weight
            orientation = tf.reshape(tf.gather_nd(T, tf.reshape(block, (-1, 4))), (n, -1))
            orientation = ((orientation - angle) * bins_per_degree)

            hist_bin = [b] + [rot + 0.5 * window_width - 0.5 for rot in rotate] + [orientation]
            hist_bin = tf.reshape(tf.stack(hist_bin, -1), (-1, 4))

            descriptors = self.__assign_descriptors(descriptors, hist_bin, magnitude)

        descriptors = tf.slice(descriptors, [0, 1, 1, 0], [key_points.shape[0], window_width, window_width, N_bins])
        descriptors = tf.reshape(descriptors, (key_points.shape[0], -1))

        threshold = tf.norm(descriptors, ord=2, axis=1, keepdims=True) * descriptor_max_value
        threshold = tf.repeat(threshold, N_bins * window_width * window_width, 1)
        descriptors = tf.where(descriptors > threshold, threshold, descriptors)
        descriptors = descriptors / tf.maximum(tf.norm(descriptors, ord=2, axis=1, keepdims=True), 1e-7)
        descriptors = tf.round(descriptors * 512)
        descriptors = tf.maximum(descriptors, 0)
        descriptors = tf.minimum(descriptors, 255)
        return descriptors

    def localize_extrema(self, octave: Octave) -> KeyPoints:
        if not isinstance(octave, Octave): raise ValueError('octave need to by of type "Octave"')
        dim = octave.shape[-1]
        con, extrema_offset, contrast_threshold, eigen_ration = 3, 0.5, 0.03, 10
        octave_index = octave.index

        """Extract all the extrema point in the octave scale space"""
        # D(batch, y, x, s)
        dog = math_ops.subtract(tf.split(octave.gss, [1, dim - 1], -1)[1],
                                tf.split(octave.gss, [dim - 1, 1], -1)[0])
        dog_shape = dog.get_shape().as_list()

        # e = (batch, y, x, s) (local extrema)
        border_width = self.border_width - 2
        extrema = compute_extrema3D(tf.round(dog), con=con, border_width=[border_width, border_width, 0])

        dog = dog / 255.0

        """Compute the key points conditions for all the image"""
        # DD / Dx
        grad = compute_central_gradient3D(dog)
        grad = tf.expand_dims(grad, -1)

        # D^2D / Dx^2
        hess = compute_hessian_3D(dog)

        # X' = - (D^2D / Dx^2) * (DD / Dx)
        extrema_update = - linalg_ops.lstsq(hess, grad, l2_regularizer=0.0, fast=False)
        extrema_update = tf.squeeze(extrema_update, axis=-1)

        # (DD / Dx) * X'
        dot_ = linalg_ops.matmul(tf.expand_dims(extrema_update, 4), grad)
        dot_ = tf.squeeze(tf.squeeze(dot_, -1), -1)

        mid_cube_values = tf.slice(dog, [0, 1, 1, 1],
                                   [dog_shape[0], dog_shape[1] - 2, dog_shape[2] - 2, dog_shape[3] - 2])

        # D(X') = D + 0.5 * (DD / Dx) * X'
        update_response = mid_cube_values + 0.5 * dot_

        hess_shape = hess.get_shape().as_list()
        # H[[Dxx, Dxy], [Dyx, Dyy]]
        hess_xy = tf.slice(hess, [0, 0, 0, 0, 0, 0], [*hess_shape[:-2], 2, 2])
        # Dxx + Dyy
        hess_xy_trace = linalg_ops.trace(hess_xy)
        # Dxx * Dyy - Dxy * Dyx
        hess_xy_det = linalg_ops.det(hess_xy)

        # |X'| <= 0.5
        # (X' is larger than 0.5 in any dimension, means that the extreme lies closer to a different sample point)
        kp_cond1 = math_ops.less_equal(math_ops.reduce_max(math_ops.abs(extrema_update), axis=-1), extrema_offset)

        # |D(X')| >= 0.03 (threshold on minimum contrast)
        kp_cond2 = math_ops.greater_equal(math_ops.abs(update_response), contrast_threshold)

        # (Dxx + Dyy) ^ 2 / Dxx * Dyy - Dxy * Dyx < (r + 1) ^ 2 / r
        # ---> ((Dxx + Dyy) ^ 2) * r < (Dxx * Dyy - Dxy * Dyx) * ((r + 1) ^ 2)
        # (threshold on ratio of principal curvatures)
        kp_cond3 = math_ops.logical_and(
            eigen_ration * (hess_xy_trace ** 2) < ((eigen_ration + 1) ** 2) * hess_xy_det, hess_xy_det != 0
        )
        cond = tf.where(kp_cond1 & kp_cond2 & kp_cond3, True, False)

        kp_cond4 = tf.scatter_nd(extrema, tf.ones((extrema.shape[0],), dtype=tf.bool), dog_shape)
        kp_cond4 = tf.slice(kp_cond4, [0, 1, 1, 1],
                            [dog_shape[0], dog_shape[1] - 2, dog_shape[2] - 2, dog_shape[3] - 2])

        """Localize the extrema points"""
        sure_key_points = math_ops.logical_and(cond, kp_cond4)
        attempts = math_ops.logical_and(kp_cond4, ~sure_key_points)

        shape_ = sure_key_points.get_shape().as_list()

        for _ in range(self.convergence_N):
            attempts_cords = tf.where(attempts)
            if attempts_cords.shape[0] == 0: break
            # if ist only one point the shape will bw (4, )
            attempts_cords = tf.reshape(attempts_cords, (-1, 4))
            attempts_update = tf.gather_nd(extrema_update, attempts_cords)

            ex, ey, ez = tf.unstack(attempts_update, num=3, axis=-1)
            cd, cy, cx, cz = tf.unstack(tf.cast(attempts_cords, tf.float32), num=4, axis=1)
            attempts_next = [cd, cy + ey, cx + ex, cz + ez]

            # check that the new cords will lie within the image shape
            cond_next = tf.where(
                (attempts_next[1] >= 0) & (attempts_next[1] < shape_[1]) & (attempts_next[2] > 0) & (
                        attempts_next[2] < shape_[2]) & (attempts_next[3] > 0) & (
                        attempts_next[3] < shape_[3]))

            attempts_next = tf.stack(attempts_next, -1)
            attempts_next = tf.cast(tf.gather(attempts_next, tf.squeeze(cond_next)), dtype=tf.int64)
            if attempts_next.shape[0] == 0: break
            attempts_next = tf.reshape(attempts_next, (-1, 4))

            attempts_mask = tf.scatter_nd(attempts_next, tf.ones((attempts_next.shape[0],), dtype=tf.bool), shape_)

            # add new key points
            new_cords = tf.where(attempts_mask & ~sure_key_points & cond)
            sure_key_points = tf.tensor_scatter_nd_update(sure_key_points, new_cords,
                                                          tf.ones((new_cords.shape[0],), dtype=tf.bool))
            # next points
            attempts = math_ops.logical_and(attempts_mask, ~sure_key_points)

        """Construct the key points"""
        cords = tf.where(sure_key_points)
        if cords.shape[0] == 0: return KeyPoints()
        kp_cords = cords + tf.constant([[0, 1, 1, 1]], dtype=tf.int64)

        # X' = - (D^2D / Dx^2) * (DD / Dx)
        extrema_update = tf.gather_nd(extrema_update, cords)
        octave_index = tf.cast(octave_index, dtype=tf.float32)

        # x', y', s'
        ex, ey, ez = tf.unstack(extrema_update, num=3, axis=1)

        # batch, y, x, s
        cd, cy, cx, cz = tf.unstack(tf.cast(kp_cords, tf.float32), num=4, axis=1)

        # pt = (batch, y = (y + y') * (1 << octave), (x + x') * (1 << octave), s) points in size of octave 0
        kp_pt = tf.stack(
            (cd, (cy + ey) * (2 ** octave_index), (cx + ex) * (2 ** octave_index), cz), axis=-1
        )
        # octave = octave_index + s * (1 << 8) + round((s' + 0.5) * 255) * (1 << 16)
        kp_octave = octave_index + cz * (2 ** 8) + tf.round((ez + 0.5) * 255.0) * (2 ** 16)

        # size = (sigma << ((s + s') / sn)) << (octave_index + 1)
        kp_size = self.sigma * (2 ** ((cz + ez) / (dim - 3))) * (2 ** (octave_index + 1.0))

        # D(X') = D + 0.5 * (DD / Dx) * X'
        kp_response = math_ops.abs(tf.gather_nd(update_response, cords))

        key_points = KeyPoints(
            pt=tf.reshape(kp_pt, (-1, 4)),
            size=tf.reshape(kp_size, (-1, 1)),
            angle=tf.reshape(tf.ones_like(kp_size) * -1.0, (-1, 1)),
            octave=tf.reshape(kp_octave, (-1, 1)),
            response=tf.reshape(kp_response, (-1, 1))
        )
        return key_points

    def orientation_assignment(self, octave: Octave, key_points: KeyPoints) -> KeyPoints:
        if not isinstance(octave, Octave): raise ValueError('octave need to by of type "Octave"')
        if not isinstance(key_points, KeyPoints): raise ValueError('key_points need to by of type "KeyPoints"')

        orientation_N_bins, scale_factor, radius_factor = 36, 1.5, 3
        histogram = tf.zeros((key_points.shape[0], orientation_N_bins), dtype=tf.float32)

        # scale = 1.5 * sigma  * (1 << ((s + s') / sn)
        scale = scale_factor * key_points.size / (2 ** (octave.index + 1))

        # r[N_points, ] = 3 * scale
        radius = tf.cast(tf.round(radius_factor * scale), dtype=tf.int64)

        # wf[N_points, ]
        weight_factor = -0.5 / (scale ** 2)

        # points back to octave resolution
        _prob = 1.0 / (1 << octave.index)
        _prob = tf.stack((tf.ones_like(_prob), _prob, _prob), axis=-1)
        _prob = tf.squeeze(_prob)

        # [batch, x + x', y + y', s] * N_points
        region_center = tf.cast(key_points.pt * _prob, dtype=tf.int64)
        region_center = tf.concat((region_center, tf.cast(key_points.scale_index, dtype=tf.int64)), -1)

        # check that the radius in the image size
        _, y, x, _ = tf.split(region_center, [1] * 4, -1)
        radius = math_ops.minimum(
            math_ops.minimum(octave.shape[1] - 3 - y, octave.shape[2] - 3 - x),
            math_ops.minimum(math_ops.minimum(y, x), radius)
        )
        radius = tf.reshape(radius, (-1,))

        # parallel computation
        parallel = tf.unique(radius)
        split_region = tf.dynamic_partition(
            tf.concat((tf.cast(region_center, tf.float32), weight_factor), -1), parallel.idx,
            tf.reduce_max(parallel.idx) + 1
        )
        index = tf.dynamic_partition(tf.reshape(tf.range(key_points.shape[0], dtype=tf.int64), (-1, 1)),
                                     parallel.idx, tf.reduce_max(parallel.idx) + 1)

        M = octave.magnitude
        T = octave.orientation

        for region_weight, r, hist_index in zip(split_region, parallel.y, index):
            region, weight = tf.split(region_weight, [4, 1], -1)

            neighbor = make_neighborhood2D(tf.constant([[0, 0, 0, 0]], dtype=tf.int64), con=(r * 2) + 1)
            block = tf.expand_dims(tf.cast(region, tf.int64), axis=1) + neighbor

            magnitude = tf.gather_nd(M, tf.reshape(block, (-1, 4)))
            orientation = tf.gather_nd(T, tf.reshape(block, (-1, 4)))

            _, curr_y, curr_x, _ = tf.unstack(tf.cast(neighbor, dtype=tf.float32), 4, axis=-1)
            weight = tf.reshape(math_ops.exp(weight * (curr_y ** 2 + curr_x ** 2)), (-1,))

            hist_deg = tf.cast(tf.round(orientation * orientation_N_bins / 360.), dtype=tf.int64) % orientation_N_bins

            hist_index = tf.ones(block.get_shape()[:-1], dtype=tf.int64) * tf.reshape(hist_index, (-1, 1))
            hist_index = tf.stack((tf.reshape(hist_index, (-1,)), hist_deg), -1)
            histogram = tf.tensor_scatter_nd_add(histogram, hist_index, weight * magnitude)

        """ find peaks in the histogram """
        # histogram smooth
        gaussian1D = tf.constant([1, 4, 6, 4, 1], dtype=tf.float32) / 16.0
        gaussian1D = tf.reshape(gaussian1D, shape=(-1, 1, 1))

        pad_ = tf.split(tf.expand_dims(histogram, axis=-1), [2, orientation_N_bins - 4, 2], 1)
        pad_ = tf.concat([pad_[-1], *pad_, pad_[0]], 1)

        smooth_histogram = tf.nn.convolution(pad_, gaussian1D, padding='VALID')
        smooth_histogram = tf.squeeze(smooth_histogram, axis=-1)

        orientation_max = tf.reduce_max(smooth_histogram, axis=-1)

        peak = tf.nn.max_pool1d(tf.expand_dims(smooth_histogram, -1), ksize=3, padding="SAME", strides=1)
        peak = tf.squeeze(peak, -1)

        value_cond = tf.repeat(tf.reshape(orientation_max, shape=(-1, 1)), repeats=36, axis=-1) * 0.8

        peak = tf.where((peak == smooth_histogram) & (smooth_histogram > value_cond))

        p_idx, p_deg = tf.unstack(peak, num=2, axis=-1)

        # interpolate the peak position - parabola
        kernel = tf.constant([1., 0, -1.], shape=(3, 1, 1))
        kernel = tf.concat((kernel, tf.constant([1., -2., 1.], shape=(3, 1, 1))), -1)

        pad_ = tf.split(smooth_histogram, [1, 34, 1], -1)
        pad_ = tf.concat([pad_[-1], *pad_, pad_[0]], -1)

        interp = tf.unstack(tf.nn.convolution(tf.expand_dims(pad_, -1), kernel, padding="VALID"), 2, -1)
        interp = 0.5 * (interp[0] / interp[1]) % 36
        interp = tf.cast(p_deg, tf.float32) + tf.gather_nd(interp, peak)

        orientation = 360. - interp * 360. / 36

        orientation = tf.where(math_ops.abs(orientation - 360.) < 1e-7, 0.0, orientation)

        wrap = key_points.as_array()
        wrap = tf.gather(wrap, p_idx)
        pt, size, _, oc, response = tf.split(wrap, [4, 1, 1, 1, 1], axis=-1)
        key_points.from_array(tf.concat((pt, size, tf.reshape(orientation, (-1, 1)), oc, response), axis=-1),
                              inplace=True)
        key_points.relies_scale_index()
        return key_points

    def write_descriptors(self, key_points: KeyPoints) -> tf.Tensor:
        if not isinstance(key_points, KeyPoints): raise ValueError('key_points need to by of type "KeyPoints"')
        unpack_oct = key_points.unpack_octave()
        parallel = tf.unique(tf.squeeze(unpack_oct.octave))

        if parallel.y.get_shape()[0] == 1:
            return self.__descriptors_per_octave(self.octave_pyramid[int(parallel.y)], key_points)

        indexes = tf.dynamic_partition(tf.reshape(tf.range(key_points.shape[0], dtype=tf.int32), (-1, 1)),
                                       parallel.idx, tf.reduce_max(parallel.idx) + 1)

        split_by_oc = key_points.partition_by_index(parallel.idx)

        condition_indices = []
        partitioned_data = []

        for keys, index, oc_id in zip(split_by_oc, indexes, parallel.y):
            oc_desc = self.__descriptors_per_octave(self.octave_pyramid[int(oc_id)], keys)
            condition_indices.append(tf.squeeze(index))
            partitioned_data.append(oc_desc)

        descriptors = tf.dynamic_stitch(condition_indices, partitioned_data)
        return descriptors

    def build_pyramid(self, I: tf.Tensor):
        def conv_with_pad(x: tf.Tensor, h: tf.Tensor) -> tf.Tensor:
            k_ = h.get_shape()[0] // 2
            x = tf.pad(x, tf.constant([[0, 0], [k_, k_], [k_, k_], [0, 0]], tf.int32), 'SYMMETRIC')
            return tf.nn.convolution(x, h, padding='VALID')

        I, kernels = self.__init_graph(I)
        self.octave_pyramid = []
        _, h_, w_, _ = I.get_shape()

        I = image_ops.resize(I, size=[h_ * 2, w_ * 2], method='bilinear')
        I = conv_with_pad(I, kernels[0])

        size_ = [h_, w_]

        for oc_id in range(self.n_octaves):
            oc_cap = [I]
            for kernel in kernels[1:]:
                I = conv_with_pad(I, kernel)
                oc_cap.append(I)
            if oc_id < self.n_octaves - 1:
                I = image_ops.resize(oc_cap[-3], size=size_, method='nearest')
                size_ = [size_[0] // 2, size_[1] // 2]

            gss = tf.concat(oc_cap, -1)
            oc = Octave(oc_id, gss)
            self.octave_pyramid.append(oc)

    def keypoints_with_descriptors(self, inputs: tf.Tensor) -> tuple[KeyPoints, tf.Tensor]:
        self.build_pyramid(inputs)

        key_points = KeyPoints()
        key_points.relies_scale_index()

        for oc in self.octave_pyramid:
            oc_kp = self.localize_extrema(oc)
            if oc_kp.shape[0] == 0: continue
            oc_kp = self.orientation_assignment(oc, oc_kp)
            key_points += oc_kp

        descriptors = self.write_descriptors(key_points)
        self.octave_pyramid = []
        return key_points, descriptors


if __name__ == '__main__':
    image1 = load_image('box.png')

    alg = SIFT()

    kp1, desc1 = alg.keypoints_with_descriptors(image1)
    # show_key_points(kp1, image1)

    image2 = load_image('box_in_scene.png')
    kp2, desc2 = alg.keypoints_with_descriptors(image2)

    src_pt, dst_pt = templet_matching_TF(kp1, kp2, desc1, desc2)
    plot_matches_TF(image1, image2, src_pt, dst_pt)

    src_pt, dst_pt = templet_matching_CV2(kp1, kp2, desc1, desc2)
    plot_matches_CV2(image1, image2, src_pt, dst_pt)
