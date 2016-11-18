import cv2
import random
import numpy as np
from rag_segmentation import RAGSegmentation
from metric_methods import euclidean_dist


class MPEG7Descriptors(object):

    def __init__(self, clusters, image):
        self.image = image
        self.clusters = clusters
        self.segments = set(clusters.flatten())
        self.segments.discard(-1)

    def dominant_colour(self, num_dominant, segment):
        old_cen_sum = 0
        centers = random.sample(segment, num_dominant)
        centers = [np.asarray(cnt, dtype="int32") for cnt in centers]
        clust_points = [[] for _ in range(0, num_dominant)]
        old_cen = [0 for _ in range(0, num_dominant)]

        while True:
            for point in segment:
                new_point = np.asarray(point, dtype="int32")
                rank_dist = [euclidean_dist(new_point, cent) for cent in centers]
                clust_points[rank_dist.index(min(rank_dist))].append(point)
            clust_points = [cpts if cpts else random.sample(segment, 1) for cpts in clust_points]
            centers = [np.mean(cpts, axis=0) for cpts in clust_points]
            cent_sum = sum(sum([abs(cnt - old) for cnt, old in zip(centers, old_cen)]))
            print abs(old_cen_sum - cent_sum)
            if abs(old_cen_sum - cent_sum) < 0.1:
                break
            old_cen = centers
            old_cen_sum = cent_sum
        print "aa"

    def find_dominant_colours(self, max_cols):
        for segment_idx in self.segments:
            idx = (self.clusters == segment_idx)
            segment_points = self.image[idx]
            dc = self.dominant_colour(max_cols, segment_points)



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
    mpeg.find_dominant_colours(8)
    #
    # cv2.imshow('contours1', rag.image_mean)
    # cv2.imshow('contours2', test_image_2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
