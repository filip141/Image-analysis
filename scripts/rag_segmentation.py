import copy

import cv2
import numpy as np
from slic import SLICSuperPixel
from metric_methods import euclidean_dist


class RAGSegmentation(object):

    def __init__(self, image, slic_clust_num=200, slic_cw=15, slic_steps=10, median_blur=7):
        image = cv2.medianBlur(image, median_blur)
        self.image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        self.image_mean = image.copy()
        self.image = image
        self.slic_sp = None
        self.slic_clusters = None
        self.slic_clust_num = slic_clust_num
        self.slic_cw = slic_cw
        self.slic_steps = slic_steps

    def run_slic(self):
        self.slic_sp = SLICSuperPixel(self.image, self.slic_clust_num, colour_weight=self.slic_cw,
                                      num_iter=self.slic_steps)
        self.slic_sp.run()
        return self.slic_sp.get_clusters()

    def slic_mean_rgb(self, clusters):
        mean_dict = {}
        for clust_idx in xrange(self.slic_clust_num):
            idx = (clusters == clust_idx)
            cluster_cols = self.image_mean[idx]
            colour_sum = np.sum(cluster_cols, axis=0)
            colour_mean = colour_sum / np.sum(idx)
            self.image_mean[idx] = colour_mean
            mean_dict[clust_idx] = colour_mean
        return mean_dict

    def slic_mean_lab(self, clusters):
        mean_dict = {}
        for clust_idx in xrange(self.slic_clust_num):
            idx = (clusters == clust_idx)
            cluster_cols = self.image_lab[idx]
            colour_sum = np.sum(cluster_cols, axis=0)
            colour_mean = colour_sum / np.sum(idx)
            self.image_lab[idx] = colour_mean
            mean_dict[clust_idx] = colour_mean
        return mean_dict

    def neighbours_regions(self, clusters):
        clust_ng = {}

        # Iterate over clusters
        for clust_idx in xrange(self.slic_clust_num):
            ng_set = set([])
            idx = (clusters == clust_idx)
            cluster_idx = np.transpose(np.nonzero(idx))
            for cord_x, cord_y in cluster_idx:
                # X constraints
                cord_x_l = cord_x - 1
                cord_x_r = cord_x + 2
                cord_x_l = cord_x_l if cord_x_l >= 0 else 0
                cord_x_r = cord_x_r if cord_x_r < clusters.shape[0] else clusters.shape[0] - 1
                # Y constraints
                cord_y_l = cord_y - 1
                cord_y_r = cord_y + 2
                cord_y_l = cord_y_l if cord_y_l >= 0 else 0
                cord_y_r = cord_y_r if cord_y_r < clusters.shape[1] else clusters.shape[1] - 1

                around = clusters[cord_x_l: cord_x_r, cord_y_l: cord_y_r]
                ng_set.update(set(around[around != clust_idx].flatten()))
            ng_set.discard(-1)
            clust_ng[clust_idx] = ng_set
        return clust_ng

    @staticmethod
    def find_edges(ng_dict, clust_col):
        edges_dict = {}
        for curr_region, ng_set in ng_dict.items():
            ng_edge = []
            for ng_reg in ng_set:
                curr_region_col = np.asarray(clust_col[curr_region], dtype="int32")
                ng_reg_col = np.asarray(clust_col[ng_reg], dtype="int32")
                dist = euclidean_dist(curr_region_col, ng_reg_col)
                ng_edge.append((ng_reg, dist))
            edges_dict[curr_region] = ng_edge
        return edges_dict

    @staticmethod
    def __convert2arrays(edges_dict):
        edges_list = []
        weight_list = []
        for vert_1, ng_list in edges_dict.items():
            for vert_2, weight in ng_list:
                new_edge = [vert_1, int(vert_2)]
                same_edge = [new_edge[1], new_edge[0]]
                if same_edge not in edges_list:
                    edges_list.append(new_edge)
                    weight_list.append(weight)
        return edges_list, weight_list

    @staticmethod
    def __sort_edges(edges_list, weight_list):
        for i in xrange(1, len(edges_list)):
            temp_edge = edges_list[i]
            temp_weight = weight_list[i]
            j = i
            while j > 0 and weight_list[j - 1] > temp_weight:
                weight_list[j] = weight_list[j - 1]
                edges_list[j] = edges_list[j - 1]
                j -= 1
            weight_list[j] = temp_weight
            edges_list[j] = temp_edge
        return edges_list, weight_list

    @staticmethod
    def __findset(vertex, vertex_list):
        for i in range(len(vertex_list)):
            for element in vertex_list[i]:
                if element == vertex:
                    return i
        return None

    def __union(self, vertex1, vertex2, vertex_list):
        index1 = self.__findset(vertex1, vertex_list)
        index2 = self.__findset(vertex2, vertex_list)
        for element in vertex_list[index2]:
            vertex_list[index1].append(element)
        vertex_list.pop(index2)
        return vertex_list

    def kruskal_alg(self, edges_list, weight_list, vertex_list):
        vertex_list = copy.deepcopy(vertex_list)
        edges_mst = []
        weight_mst = []
        edge_idx = 0
        while len(vertex_list) > 1 and edge_idx < len(edges_list):
            ver_idx_one = self.__findset(edges_list[edge_idx][0], vertex_list)
            ver_idx_two = self.__findset(edges_list[edge_idx][1], vertex_list)
            if ver_idx_one != ver_idx_two:
                edges_mst.append(edges_list[edge_idx])
                weight_mst.append(weight_list[edge_idx])
                vertex_list = self.__union(edges_list[edge_idx][0], edges_list[edge_idx][1], vertex_list)
            edge_idx += 1
        return edges_mst, weight_mst, vertex_list

    def concat_similar_regs(self, edges_dict, clusters, c_factor=2.0):

        # Kruskal algorithm
        new_clusters = copy.deepcopy(clusters)
        edges_list, weight_list = self.__convert2arrays(edges_dict)
        # make vertex list
        vertex_list = edges_dict.keys()
        for vert_idx in xrange(0, len(vertex_list)):
            vertex_list[vert_idx] = [vertex_list[vert_idx]]

        # sort edges
        edges_list, weight_list = self.__sort_edges(edges_list, weight_list)
        edge_mst, weight_mst, vertex_mst = self.kruskal_alg(edges_list, weight_list, vertex_list)

        new_edges = []
        new_weight = []
        for edge_idx in xrange(len(edge_mst)):
            weight_mst_copy = copy.deepcopy(weight_mst)
            weight_mst_copy.remove(weight_mst_copy[edge_idx])
            mst_mean = np.mean(weight_mst_copy)
            mst_std = np.std(weight_mst_copy)
            if weight_mst[edge_idx] < mst_mean + c_factor * mst_std:
                new_edges.append(edge_mst[edge_idx])
                new_weight.append(weight_mst[edge_idx])

        # Calculate new tree
        edge_mst, weight_mst, vertex_mst = self.kruskal_alg(new_edges, new_weight, vertex_list)

        new_cluster = 1
        for tree in vertex_mst:
            ng_mask = np.zeros(new_clusters.shape, dtype=bool)
            for vertex in tree:
                ng_mask = np.logical_or(ng_mask, new_clusters == vertex)
            new_clusters[ng_mask] = new_cluster
            new_cluster += 1

        return new_clusters, edge_mst, weight_mst, vertex_mst

    def plot_regions(self, clusters, edge_mst):
        reg_points = {}
        for clust_idx in xrange(self.slic_clust_num):
            ng_mask = clusters == clust_idx
            cluster_idx = np.transpose(np.nonzero(ng_mask))
            if cluster_idx.size != 0:
                cntr = cv2.findContours((1 * ng_mask).astype(np.uint8).copy(),
                                        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
                M = cv2.moments(cntr)
                c_point = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                reg_points[clust_idx] = c_point
                cv2.circle(self.image_mean, c_point, 5, (0, 0, 255), -1)
                cv2.putText(self.image_mean, str(clust_idx), c_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        for edge in edge_mst:
            point_one = edge[0]
            point_two = edge[1]
            cv2.line(self.image_mean, reg_points[point_one], reg_points[point_two], (255, 0, 0), 1)

if __name__ == '__main__':
    test_image = cv2.imread('../data/road.jpg', 1)
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

    concat_params = rag.concat_similar_regs(ed, t_clusters, c_factor=0.22)
    n_clusters = concat_params[0]
    edge_mst = concat_params[1]
    clust_col_rgb = rag.slic_mean_rgb(n_clusters)
    # rag.plot_regions(t_clusters, edge_mst)

    cv2.imshow('contours1', rag.image_mean)
    cv2.imshow('contours2', test_image_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
