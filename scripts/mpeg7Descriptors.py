import cv2
import sys
import random
import numpy as np
from rag_segmentation import RAGSegmentation
from metric_methods import euclidean_dist

sys.setrecursionlimit(20000)


class MPEG7Descriptors(object):

    def __init__(self, clusters, image):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
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
            print abs(old_cen_sum - cent_sum)
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
            counter = 0
            reg_in_cent = {}
            cent_idxs = map(lambda c_idx: tuple(c_idx), cent_idxs)
            while len(cent_idxs) > 0:
                visited_edges = []
                all_edges = []
                self.get_px_ngs([cent_idxs[0]], visited_edges, cent_idxs, all_edges)
                reg_in_cent[counter] = visited_edges
                cent_idxs = list(set(cent_idxs) - set(visited_edges))
                counter += 1
            spatial_homo = map(lambda ric: len(ric), reg_in_cent.values())
            spatial_coh = max(spatial_homo) / float(len(cent_points))
            dcd_features[cnt_idx] = [dc_mean, dc_var, dc_percentage, spatial_coh]
        return dcd_features

    def find_dominant_colours(self, max_cols):
        dcd_per_seg = {}
        for segment_idx in self.segments:
            idx = (self.clusters == segment_idx)
            seg_idxs = np.transpose(np.nonzero(idx == True))
            segment_points = self.image[idx]
            dc = self.dominant_colour(max_cols, segment_points, seg_idxs)
            dcd_per_seg[segment_idx] = dc
        return dcd_per_seg

if __name__ == '__main__':
    test_image = cv2.imread('../data/fox.png', 1)
    test_image_2 = test_image.copy()
    rag = RAGSegmentation(test_image, slic_clust_num=200, slic_cw=15, median_blur=7)
    t_clusters = rag.run_slic()
    # rag.slic_sp.plot()

    # take mean
    clust_col_rgb = rag.slic_mean_rgb(t_clusters)
    clust_col_t = rag.slic_mean_lab(t_clusters)
    # calculate edges
    cn = rag.neighbours_regions(t_clusters)
    ed = rag.find_edges(cn, clust_col_t)

    concat_params = rag.concat_similar_regs(ed, t_clusters, c_factor=0.4)
    n_clusters = concat_params[0]
    edge_mst = concat_params[1]
    clust_col_rgb = rag.slic_mean_rgb(n_clusters)
    rag.plot_regions(t_clusters, edge_mst)
    mpeg = MPEG7Descriptors(n_clusters, test_image)
    dcd = mpeg.find_dominant_colours(max_cols=8)
    print "aa"

    # mpeg.find_dominant_colours(8)
    #
    # cv2.imshow('contours1', rag.image_mean)
    # cv2.imshow('contours2', test_image_2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
