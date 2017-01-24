import cv2
import numpy as np


class SLICSuperPixel(object):

    def __init__(self, image, num_clusters, colour_weight=10, num_iter=10, step=None):

        # If image loaded not correctly throw exception
        if not isinstance(image, np.ndarray):
            raise ValueError("Image value is None. Probably not loaded correctly!")

        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        # Define image size
        height = image_lab.shape[0]
        width = image_lab.shape[1]

        self.centers = []
        self.clusters = None
        self.center_counts = []
        self.image_rgb = image
        self.image = np.asarray(image_lab, dtype="int32")
        self.num_iter = num_iter
        self.num_clusters = num_clusters
        self.colour_weight = colour_weight

        # If step not defined define default
        if step is None:
            self.step = np.sqrt((height * width) / num_clusters)
        else:
            self.step = step

        self.step = int(self.step)

    def init_clusters(self):
        height = self.image.shape[0]
        width = self.image.shape[1]

        self.centers = []
        self.center_counts = []
        # Initialize cluster centers
        for i in xrange(self.step, width - self.step / 2, self.step):
            for j in xrange(self.step, height - self.step / 2, self.step):
                center_feature = []
                pixel_pos = (i, j)
                new_pos = self.find_local_minimum(self.image, pixel_pos)
                colour = self.image[new_pos[1], new_pos[0]]

                # Add center features
                center_feature += [int(col_x) for col_x in colour]
                center_feature += new_pos

                self.centers.append(center_feature)
                self.center_counts.append(0.0)

    @staticmethod
    def find_local_minimum(image, center):
        min_grad = float("inf")
        loc_min = center

        for i in xrange(center[0] - 1, center[0] + 2):
            for j in xrange(center[1] - 1, center[1] + 2):

                # Read neighbour pixels
                px_1 = image[j + 1, i]
                px_2 = image[j, i + 1]
                px_3 = image[j, i]

                l_1 = int(px_1[0])
                l_2 = int(px_2[0])
                l_3 = int(px_3[0])

                if (np.sqrt((l_1 - l_3)**2) + np.sqrt((l_2 - l_3)**2)) < min_grad:
                    min_grad = abs(l_1 - l_3) + abs(l_2 - l_3)
                    loc_min = (i, j)
        return loc_min

    def run(self):
        # Initialize start points
        self.init_clusters()
        clusters = -np.ones(self.image.shape[:-1])

        # Repeat iteration
        for iteration in range(1, self.num_iter):
            distance = -np.log(np.zeros(self.image.shape[:-1]))

            # For each center point
            for cntr in xrange(0, len(self.centers)):
                # Iterate over pixels in 2S
                x_range = [self.centers[cntr][3] - self.step, self.centers[cntr][3] + self.step]
                y_range = [self.centers[cntr][4] - self.step, self.centers[cntr][4] + self.step]

                # Prepare constraints
                x_l = x_range[0] if x_range[0] >= 0 else 0
                x_r = x_range[1] if x_range[1] < self.image.shape[1] else self.image.shape[1] - 1
                y_l = y_range[0] if y_range[0] >= 0 else 0
                y_r = y_range[1] if y_range[1] < self.image.shape[0] else self.image.shape[0] - 1

                # Colour distance
                image_piece = self.image[y_l:y_r, x_l:x_r]
                colour_dif = image_piece - self.image[self.centers[cntr][4], self.centers[cntr][3]]
                dc = np.sqrt(np.sum(np.square(colour_dif), axis=2))

                # Distance spatial
                x_coords = np.arange(x_l, x_r)
                y_coords = np.arange(y_l, y_r)
                y_coords = y_coords.reshape(y_coords.shape + (1,))
                ds = np.sqrt((x_coords - self.centers[cntr][3])**2 + (y_coords - self.centers[cntr][4])**2)
                eucl_dist = np.sqrt((dc / self.colour_weight)**2 + (ds / self.step)**2)

                # Append new distances
                dist_pr = distance[y_l:y_r, x_l:x_r]
                idx = dist_pr > eucl_dist
                dist_pr[idx] = eucl_dist[idx]
                distance[y_l:y_r, x_l:x_r] = dist_pr
                clusters[y_l:y_r, x_l:x_r][idx] = cntr

            for cen_idx in xrange(len(self.centers)):
                idx = (clusters == cen_idx)
                cluster_cols = self.image[idx]
                cluster_dist = np.where(idx == True)
                sum_y = np.sum(cluster_dist[0])
                sum_x = np.sum(cluster_dist[1])
                self.centers[cen_idx][0:3] = np.sum(cluster_cols, axis=0)
                self.centers[cen_idx][3:] = sum_x, sum_y
                self.centers[cen_idx] /= np.sum(idx)
        self.clusters = clusters

    def find_contours(self):
        contours = []
        dx8 = [-1, -1, 0, 1, 1, 1, 0, -1]
        dy8 = [0, -1, -1, -1, 0, 1, 1, 1]
        istaken = np.zeros(self.image.shape[:-1])

        for j in xrange(0, self.image.shape[0]):
            for i in xrange(0, self.image.shape[1]):
                nr_p = 0
                for k in xrange(0, 8):
                    x = i + dx8[k]
                    y = j + dy8[k]
                    if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
                        if istaken[y, x] == 0 and self.clusters[j, i] != self.clusters[y, x]:
                            nr_p += 1

                if nr_p >= 2:
                    contours.append([j, i])
                    istaken[j, i] = 1
        return contours

    def plot(self):
        contours = self.find_contours()
        for cont in contours:
            self.image_rgb[cont[0], cont[1]] = [0, 0, 255]

        cv2.imshow('superPixel', self.image_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_clusters(self):
        return self.clusters


if __name__ == '__main__':
    test_image = cv2.imread('../data/road.jpg', 1)
    slic_sp = SLICSuperPixel(test_image, 200, colour_weight=15)
    slic_sp.run()
    slic_sp.plot()