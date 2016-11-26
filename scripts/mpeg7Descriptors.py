import cv2
import sys
import random
import scipy.misc
import numpy as np
import scipy.fftpack
import scipy.interpolate
import scipy.ndimage.interpolation
import matplotlib.pyplot as plt
from rag_segmentation import RAGSegmentation
from metric_methods import meas_sim_texture_desc, meas_sim_col, meas_sim_shape
from const_data import CEN_FREQ, OCT_BAND, ANGULAR_BAND, ANGULAR_CEN

sys.setrecursionlimit(20000)
gabor_radial = 5
gabor_angular = 6
default_wres = 320


class MPEG7Descriptors(object):

    def __init__(self, clusters, image):
        self.image_rgb = image
        self.image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        self.image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.clusters = clusters
        self.segments = set(clusters.flatten())
        self.segments.discard(-1)

    def get_px_ngs(self, ref_pxs, used_pxs, cent_idxs, all_seg_pxs):
        while True:
            ref_pxs = list(set(ref_pxs) - set(used_pxs))
            if len(ref_pxs) <= 0:
                break
            ref_px = ref_pxs[0]
            ref_pxs.remove(ref_px)

            x_r = [ref_px[0] - 1, ref_px[0] + 2]
            y_r = [ref_px[1] - 1, ref_px[1] + 2]
            # Constraints
            x_r[0] = x_r[0] if x_r[0] >= 0 else 0
            x_r[1] = x_r[1] if x_r[1] < 255 else 255
            y_r[0] = y_r[0] if y_r[0] >= 0 else 0
            y_r[1] = y_r[1] if y_r[1] < 255 else 255

            # Get pixels indexes
            reg_pxs = []
            for i in xrange(x_r[0], x_r[1]):
                for j in xrange(y_r[0], y_r[1]):
                    reg_pxs.append((i, j))

            reg_ng_pxs = []
            for reg_px in reg_pxs:
                if (reg_px != ref_px) and reg_px in cent_idxs:
                    reg_ng_pxs.append(reg_px)
                    if reg_px not in all_seg_pxs:
                        all_seg_pxs.append(reg_px)

            used_pxs.append(ref_px)
            reg_ng_pxs = list(set(reg_ng_pxs) - set(used_pxs))
            if not reg_ng_pxs:
                continue
            else:
                self.get_px_ngs(reg_ng_pxs, used_pxs, cent_idxs, all_seg_pxs)

    def dominant_colour(self, num_dominant, segment, seg_idxs):
        old_cen_sum = 0
        segment = np.asarray(segment, dtype=np.float32)
        centers = random.sample(segment, num_dominant)
        centers = [np.asarray(cnt, dtype=np.float32) for cnt in centers]
        clust_points = [[] for _ in xrange(0, num_dominant)]
        old_cen = [0 for _ in xrange(0, num_dominant)]
        clusters = -np.ones(segment.shape[:-1])

        while True:
            distance = -np.log(np.zeros(segment.shape[:-1]))
            for idc, cntr in enumerate(centers):
                col_difs = segment - cntr
                dist = np.sqrt(np.sum(np.square(col_difs), axis=1))
                idx = distance > dist
                distance[idx] = dist[idx]
                clusters[idx] = idc
            for cnt_idx in xrange(0, num_dominant):
                cl_idx = (clusters == cnt_idx)
                clust_points[cnt_idx] = segment[cl_idx]
                centers[cnt_idx] = np.sum(clust_points[cnt_idx], axis=0, dtype=np.float32)
                points_number = np.sum(cl_idx)
                if points_number:
                    centers[cnt_idx] /= points_number
                else:
                    centers[cnt_idx] = random.sample(segment, 1)[0]
            cent_sum = sum(sum([abs(cnt - old) for cnt, old in zip(centers, old_cen)]))
            # print abs(old_cen_sum - cent_sum)
            if abs(old_cen_sum - cent_sum) < 0.1:
                break
            old_cen = [o_cnt for o_cnt in centers]
            old_cen_sum = cent_sum
        # For each dominant colour
        dcd_features = {}
        for cnt_idx in xrange(0, num_dominant):
            dc_idx = (clusters == cnt_idx)
            cent_points = segment[dc_idx].tolist()
            cent_idxs = seg_idxs[dc_idx]
            dc_mean = centers[cnt_idx]
            dc_var = np.var(cent_points, axis=0)
            dc_percentage = len(cent_points) / float(len(segment))

            # Find spatial coherence
            reg_in_cent = []
            cent_idxs = [tuple(c_idx) for c_idx in cent_idxs]
            while len(cent_idxs) > 0:
                visited_edges = []
                all_edges = []
                self.get_px_ngs([cent_idxs[0]], visited_edges, cent_idxs, all_edges)
                reg_in_cent.append(visited_edges)
                cent_idxs = list(set(cent_idxs) - set(visited_edges))
            spatial_homo = [len(ric) for ric in reg_in_cent]
            spatial_coh = max(spatial_homo) / float(len(cent_points))
            dcd_features[cnt_idx] = [dc_mean, dc_var, dc_percentage, spatial_coh]
        return dcd_features

    def mpeg7_dominant_colours(self, max_cols):
        dcd_per_seg = {}
        old = None
        for segment_idx in self.segments:
            print "Extracting data from segment: {}".format(segment_idx)
            idx = (self.clusters == segment_idx)
            seg_idxs = np.transpose(np.nonzero(idx == True))
            segment_points = self.image_lab[idx]
            dc = self.dominant_colour(max_cols, segment_points, seg_idxs)
            dcd_per_seg[segment_idx] = dc
            if old is not None:
                meas_sim_col(dc, old)
            old = dc
        return dcd_per_seg

    @staticmethod
    def cart2radial(image, one_dim=False, interp=True, dtype=np.uint8):
        if one_dim:
            width, height = image.shape
            blank_image = np.zeros((width, height), dtype)
        else:
            width, height = image.shape[:-1]
            blank_image = np.zeros((width, height, 3), dtype)

        width, height = width - 1, height - 1
        max_rad = np.sqrt(width**2 + height**2) / 2.0
        r_scale = max_rad / width
        ang_scale = (2 * np.pi) / height
        for y_a in xrange(0, height):
            for x_a in range(0, width):
                angle = y_a * ang_scale
                radius = x_a * r_scale

                polar_x = radius * np.cos(angle) + width / 2.0
                polar_y = radius * np.sin(angle) + height / 2.0

                if interp:
                    x_r = [int(polar_x), int(polar_x) + 2]
                    y_r = [int(polar_y), int(polar_y) + 2]

                    x_r[0] = x_r[0] if x_r[0] >= 0 else 0
                    x_r[1] = x_r[1] if x_r[1] <= width else width
                    y_r[0] = y_r[0] if y_r[0] >= 0 else 0
                    y_r[1] = y_r[1] if y_r[1] <= height else height

                    img_col = []
                    image_square = image[x_r[0]:x_r[1], y_r[0]:y_r[1]]
                    for im_px in image_square:
                        img_col += im_px.tolist()
                    if len(img_col) == 0:
                        continue
                    new_col = np.sum(img_col, axis=0) / len(img_col)
                else:
                    new_px = [int(polar_x), int(polar_y)]
                    new_px[0] = new_px[0] if new_px[0] <= width else width
                    new_px[1] = new_px[1] if new_px[1] <= height else height
                    new_col = image[new_px[0], new_px[1]]
                blank_image[x_a, y_a] = new_col
        return blank_image

    @staticmethod
    def polar2cart(pol_image, one_dim=False, interp=True, dtype=np.uint8):
        if one_dim:
            width, height = pol_image.shape
            blank_image = np.zeros((width, height), dtype)
        else:
            width, height = pol_image.shape[:-1]
            blank_image = np.zeros((width, height, 3), dtype)
        width, height = width - 1, height - 1
        max_rad = np.sqrt(width**2 + height**2) / 2.0
        r_scale = max_rad / width
        ang_scale = (2 * np.pi) / height
        for y_a in xrange(0, height):
            for x_a in range(0, width):
                dy = y_a - height / 2
                dx = x_a - width / 2

                angle = np.arctan2(dy, dx) % (2 * np.pi)
                radius = np.sqrt(dx**2 + dy**2)
                image_y = angle / ang_scale
                image_x = radius / r_scale
                if interp:
                    x_r = [int(image_x), int(image_x) + 2]
                    y_r = [int(image_y), int(image_y) + 2]

                    x_r[0] = x_r[0] if x_r[0] >= 0 else 0
                    x_r[1] = x_r[1] if x_r[1] <= width else width
                    y_r[0] = y_r[0] if y_r[0] >= 0 else 0
                    y_r[1] = y_r[1] if y_r[1] <= height else height

                    img_col = []
                    image_square = pol_image[x_r[0]:x_r[1], y_r[0]:y_r[1]]
                    for im_px in image_square:
                        img_col += im_px.tolist()
                    if len(img_col) == 0:
                        continue
                    new_col = np.sum(img_col, axis=0) / len(img_col)
                else:
                    new_px = [int(image_x), int(image_y)]
                    new_px[0] = new_px[0] if new_px[0] <= width else width
                    new_px[1] = new_px[1] if new_px[1] <= height else height
                    new_col = pol_image[new_px[0], new_px[1]]
                blank_image[x_a, y_a] = new_col
        return blank_image

    def art_transform(self, image):
        coeffs_n, coeffs_m = (3, 12)
        height, width = image.shape[:-1]
        ang_scale = (2 * np.pi) / height
        r_scale = 1 / float(width)
        polar_image = cv2.cvtColor(self.cart2radial(image), cv2.COLOR_BGR2GRAY)
        result_art = np.zeros((coeffs_n, coeffs_m), dtype=np.float64)
        angle_mat = ang_scale * (np.ones((width, 1)) * np.arange(height)).transpose()
        radius_mat = r_scale * np.arange(width)
        for n in xrange(coeffs_n):
            for m in xrange(coeffs_m):
                a_m = (1 / (2 * np.pi)) * np.exp(1j * m * angle_mat)
                if n == 0:
                    r_n = np.ones((1, width))
                else:
                    r_n = 2 * np.cos(np.pi * n * radius_mat)
                r_n = r_n * np.eye(width)
                v_base = np.dot(a_m, r_n)
                image_mult = polar_image * v_base * r_scale * (np.ones((height, 1)) * np.arange(width))
                high_sum = np.sum(image_mult, axis=1, dtype=np.complex128) * r_scale
                art_coeff = np.sum(high_sum, dtype=np.complex128) * ang_scale
                result_art[n, m] = np.abs(art_coeff)
        result_art /= result_art[0, 0]
        return result_art

    def mpeg7_region_shape(self):
        feature_per_sg = {}
        for segment_idx in self.segments:
            print "Extracting data from segment: {}".format(segment_idx)
            new_image = self.image_rgb.copy()
            for negative_idx in self.segments:
                if negative_idx != segment_idx:
                    neg_seg = (self.clusters == negative_idx)
                    new_image[neg_seg] = 0
            feat_lst = self.art_transform(new_image)
            feature_per_sg[segment_idx] = feat_lst
        return feature_per_sg

    def mpeg7_homogeneus_texture(self):
        feature_per_sg = {}
        for segment_idx in self.segments:
            print "Extracting data from segment: {}".format(segment_idx)
            new_image = self.image_gray.copy()
            for negative_idx in self.segments:
                if negative_idx != segment_idx:
                    neg_seg = (self.clusters == negative_idx)
                    new_image[neg_seg] = 0
            feat_lst = self.radon_feature_extract(new_image)
            feature_per_sg[segment_idx] = feat_lst
        return feature_per_sg

    @staticmethod
    def gabor_filters(num_rad, num_ang, plane_len):
        # Define plane points
        ang_w, rad_h = plane_len
        rad = lambda x: ((np.pi * x) / 180)
        angle_p = np.array([(2 * np.pi * ang) / ang_w for ang in xrange(ang_w)])
        radius_p = np.linspace(0, 1, rad_h)
        radius_p, angle_p = np.meshgrid(radius_p, angle_p)
        radius_p = radius_p.flatten()
        angle_p = angle_p.flatten()
        g_filter_list = []
        for r_idx in xrange(num_rad):
            for a_idx in xrange(num_ang):
                rad_std = OCT_BAND[r_idx] / 2 * np.sqrt(2 * np.log(2))
                ang_std = rad(ANGULAR_BAND[a_idx]) / 2 * np.sqrt(2 * np.log(2))
                gabor_func = (np.exp((-((radius_p - 0.5) - CEN_FREQ[r_idx])**2) / (2.0 * rad_std**2)) *
                              np.exp((-(angle_p - rad(ANGULAR_CEN[a_idx]))**2) / (2.0 * ang_std**2)))\
                    .reshape(plane_len)
                g_filter_list.append(gabor_func)
        return g_filter_list

    @staticmethod
    def radon_feature_extract(image):
        # Convert to gray scale
        c_num = 329
        image = np.asarray(image, dtype=np.float32) / 255.0
        width, height = image.shape
        if height > width:
            add_new = np.zeros((height - width, height))
            image = np.concatenate((image, add_new), axis=0)
        else:
            add_new = np.zeros((width, width - height))
            image = np.concatenate((image, add_new), axis=1)
        fourier_res = height

        # Compute Radon Transform
        sinogram = np.array([np.sum(
            scipy.ndimage.interpolation.rotate(
                image,
                np.rad2deg((np.pi * per) / c_num),
                reshape=False,
                mode='constant',
                cval=0.0
            ), axis=0
        ) for per in xrange(c_num)])

        # Take Radon FFT, Central Slice Theorem
        sinogram_fft_rows = scipy.fftpack.fftshift(
            scipy.fftpack.fft(
                scipy.fftpack.ifftshift(
                    sinogram,
                    axes=1
                )
            ), axes=1
        )

        # Generate points in polar coordinates
        angle_p = np.array([(2 * np.pi * ang) / c_num for ang in xrange(c_num)])
        radius_p = np.linspace(0, 1, fourier_res)
        radius_p, angle_p = np.meshgrid(radius_p, angle_p)
        g_filters = MPEG7Descriptors.gabor_filters(gabor_radial, gabor_angular, (c_num, fourier_res))
        feature_list = []
        for g_func in g_filters:
            feature = {}
            p = (g_func * radius_p * sinogram_fft_rows)**2
            sum_p = np.sum(np.sum(p, axis=1))
            e = np.log10(1 + sum_p)
            q = ((g_func * radius_p * sinogram_fft_rows)**2 - sum_p)**2
            sum_q = np.sqrt(np.sum(np.sum(q, axis=1)))
            d = np.log10(1 + sum_q)
            feature['d'] = d
            feature['e'] = e
            feature_list.append(feature)

        image_mean = np.mean(image)
        image_std = np.std(image)
        feature_list.append(image_mean)
        feature_list.append(image_std)
        return feature_list

if __name__ == '__main__':
    test_image = cv2.imread('../data/ss1.png', 1)
    im_res = test_image.shape[:-1]
    factor = im_res[0] / float(im_res[1])
    n_image = cv2.resize(test_image, (default_wres, int(default_wres * factor)))
    test_image_2 = n_image.copy()
    mpeg = MPEG7Descriptors(np.array([]), n_image)
    test_case_feat = mpeg.art_transform(test_image_2)
    check_imgs = ["kwadrat.jpg", "circ.jpg", "fig.jpg", "gw.jpg", "trojkat.jpg", "gw2.png", "plane_2.jpg",
                  "apple2.jpg", "home.jpg", "ss.jpeg"]
    ranks = []
    for image in check_imgs:
        print "Extracting features for image: {}".format(image)
        test_image = cv2.imread('../data/' + image, 1)
        im_res = test_image.shape[:-1]
        factor = im_res[0] / float(im_res[1])
        n_image = cv2.resize(test_image, (default_wres, int(default_wres * factor)))
        test_image_2 = n_image.copy()
        mpeg = MPEG7Descriptors(np.array([]), n_image)
        base_feat = mpeg.art_transform(test_image_2)
        ranks.append(meas_sim_shape(test_case_feat, base_feat))
    print "Item Index: " + str(ranks.index(min(ranks)))
    print "Grades: " + str(ranks)
    # mpeg.art_transform()
    # rag = RAGSegmentation(n_image, slic_clust_num=200, slic_cw=15, median_blur=7)
    # t_clusters = rag.run_slic()
    # # rag.slic_sp.plot()
    #
    # # take mean
    # clust_col_rgb = rag.slic_mean_rgb(t_clusters)
    # clust_col_t = rag.slic_mean_lab(t_clusters)
    # # calculate edges
    # cn = rag.neighbours_regions(t_clusters)
    # ed = rag.find_edges(cn, clust_col_t)
    #
    # concat_params = rag.concat_similar_regs(ed, t_clusters, c_factor=0.20)
    # n_clusters = concat_params[0]
    # edge_mst = concat_params[1]
    # clust_col_rgb = rag.slic_mean_rgb(n_clusters)
    # rag.plot_regions(t_clusters, edge_mst)
    # mpeg = MPEG7Descriptors(n_clusters, n_image)
    # nn = mpeg.mpeg7_dominant_colours(6)
    # # nn = mpeg.polar2cart(nn)
    # # nn = mpeg.cart2radial(n_image)
    # # nn = mpeg.polar2cart(nn)
    # # dcd = mpeg.find_dominant_colours(max_cols=8)
    # print "aa"

    # cv2.imshow('contours1', n_image)
    # # cv2.imshow('contours2', nn)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
