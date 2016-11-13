import cv2
import numpy as np
from slic import SLICSuperPixel


class RAGSegmentation(object):

    def __init__(self, image, slic_clust_num=200, slic_cw=15, slic_steps=10):
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
            clust_ng[clust_idx] = ng_set
        return clust_ng

    @staticmethod
    def euclidean_dist(col_one, col_two):
        return np.sqrt((col_one[0] - col_two[0])**2 + (col_one[1] - col_two[1])**2 + (col_one[2] - col_two[2])**2)

    def find_edges(self, ng_dict, clust_col):
        edges_dict = {}
        for curr_region, ng_set in ng_dict.items():
            ng_edge = []
            for ng_reg in ng_set:
                curr_region_col = np.asarray(clust_col[curr_region], dtype="int32")
                ng_reg_col = np.asarray(clust_col[ng_reg], dtype="int32")
                dist = self.euclidean_dist(curr_region_col, ng_reg_col)
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

    def concat_similar_regs(self, edges_dict, clust_col, clusters, image, thresh=40):
        edges_mst = []
        weight_mst = []
        # Kruskal algorithm
        edges_list, weight_list = self.__convert2arrays(edges_dict)
        # make vertex list
        vertex_list = edges_dict.keys()
        for vert_idx in xrange(0, len(vertex_list)):
            vertex_list[vert_idx] = [vertex_list[vert_idx]]

        # sort edges
        edges_list, weight_list = self.__sort_edges(edges_list, weight_list)
        edge_idx = 0
        while len(vertex_list) > 1 and edge_idx < len(edges_list):
            if self.__findset(edges_list[edge_idx][0], vertex_list) != self.__findset(edges_list[edge_idx][1],
                                                                                      vertex_list):
                edges_mst.append(edges_list[edge_idx])
                weight_mst.append(weight_list[edge_idx])
                vertex_list = self.__union(edges_list[edge_idx][0], edges_list[edge_idx][1], vertex_list)
            edge_idx += 1
        vertex_mst = vertex_list[0]
        print "xx"
        # for curr_region, ng_set in edges_dict.items():
        #     col_mean = clust_col[curr_region]
        #     ng_mask = clusters == curr_region
        #     ng_list = []
        #     col_counter = 1
        #     for ng_reg, ng_dist in ng_set:
        #         if ng_dist < thresh:
        #             col_mean += clust_col[ng_reg]
        #             ng_mask = np.logical_or(ng_mask, clusters == ng_reg)
        #             ng_list.append(ng_reg)
        #             col_counter += 1
        #     col_mean /= col_counter
        #     image[ng_mask] = col_mean
        # return image

    def plot_regions(self, clusters):
        for clust_idx in xrange(self.slic_clust_num):
            ng_mask = clusters == clust_idx
            c = max(self.image[ng_mask], key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

if __name__ == '__main__':
    test_image = cv2.imread('../data/beach.jpg', 1)
    rag = RAGSegmentation(test_image)
    t_clusters = rag.run_slic()
    # take mean
    clust_col_rgb = rag.slic_mean_rgb(t_clusters)
    clust_col_t = rag.slic_mean_lab(t_clusters)
    # calculate edges
    cn = rag.neighbours_regions(t_clusters)
    ed = rag.find_edges(cn, clust_col_t)

    rag.plot_regions(t_clusters)
    image_cct = rag.concat_similar_regs(ed, clust_col_rgb, t_clusters, test_image.copy())
    cv2.imshow('contours', rag.image_mean)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
