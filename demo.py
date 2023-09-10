from viz import show_key_points, show_images, plot_matches_TF, plot_matches_CV2
from utils import load_image, templet_matching_TF, templet_matching_CV2
from sift import SIFT

if __name__ == '__main__':
    alg = SIFT()

    image1 = load_image('demo_figs\\templet1.jpeg')
    kp1, desc1 = alg.keypoints_with_descriptors(image1)
    show_key_points(kp1, image1)

    image2 = load_image('demo_figs\\scan4.jpeg')
    kp2, desc2 = alg.keypoints_with_descriptors(image2)
    show_key_points(kp2, image2)

    src_pt, dst_pt = templet_matching_CV2(kp1, kp2, desc1, desc2, ratio_threshold=0.7)
    out = plot_matches_CV2(image1, image2, src_pt, dst_pt)
