from viz import show_key_points, show_images, plot_matches_TF, plot_matches_CV2
from utils import load_image, templet_matching_TF, templet_matching_CV2
from sift import SIFT

if __name__ == '__main__':
    image1 = load_image('box.png')

    alg = SIFT()

    kp1, desc1 = alg.keypoints_with_descriptors(image1)
    # show_key_points(kp1, image1)

    image2 = load_image('box_in_scene.png')
    kp2, desc2 = alg.keypoints_with_descriptors(image2)

    # # Run this only with GPU if number of points is large
    src_pt, dst_pt = templet_matching_TF(kp1, kp2, desc1, desc2)
    out = plot_matches_TF(image1, image2, src_pt, dst_pt)

    # src_pt, dst_pt = templet_matching_CV2(kp1, kp2, desc1, desc2)
    # out = plot_matches_CV2(image1, image2, src_pt, dst_pt)