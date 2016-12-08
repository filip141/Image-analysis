import cv2
import numpy as np


class SpatialExtractor(object):

    def contours_generator(self):
        for segment_idx in self.segments:
            idx = 1 * (self.clusters == segment_idx)
            cntr = cv2.findContours(idx.astype(np.uint8).copy(),
                                    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            yield cntr, idx, segment_idx

    def __init__(self, clusters):
        self.clusters = clusters
        self.segments = set(clusters.flatten())
        self.segments.discard(-1)

    @staticmethod
    def linear_coeffs(point_o, point_t):
        a = (point_o[0] - point_t[0]) / (point_o[1] - point_t[1])
        b = point_t[0] - a * point_t[1]
        return a, b

    def ratio_indexing(self, nw, curr_idx):
        # Numpy Indexing
        np_idx = np.zeros(self.clusters.shape, dtype=np.bool)
        for point in nw:
            np_idx[point] = True
        segs_in_reg = self.clusters[np_idx]
        segs_set = set(segs_in_reg)
        ratio_rel = {}
        for sngl_seg in segs_set:
            if sngl_seg != curr_idx:
                s_idx = np.where(self.clusters == sngl_seg)
                ratio = segs_in_reg.tolist().count(sngl_seg) / float(s_idx[0].size)
                ratio_rel[int(sngl_seg)] = ratio
        return ratio_rel

    def find_spatial(self, c_param):
        im_h, im_w = self.clusters.shape
        relations = {}
        for cntr, mask, idx in self.contours_generator():
            print "Extracting spatial relations for segment: {}".format(idx)
            local_rels = {}
            # Compute MBR
            x, y, w, h = cv2.boundingRect(cntr)
            # Calculate center
            x_cntr = x + w / 2.0
            y_cntr = y + h / 2.0
            # Calculate new height and width
            new_h = np.sqrt(1 + c_param) * h
            new_w = np.sqrt(1 + c_param) * w
            # New x and y
            new_x = int(x_cntr - new_w / 2)
            new_y = int(y_cntr - new_h / 2)
            new_h = int(new_h)
            new_w = int(new_w)
            new_x = new_x if new_x > 0 else 0
            new_y = new_y if new_y > 0 else 0
            new_w = new_w if (new_x + new_w) < im_w else (im_w - new_x - 1)
            new_h = new_h if (new_y + new_h) < im_h else (im_h - new_y - 1)
            new_x_cntr = new_x + new_w / 2.0
            new_y_cntr = new_y + new_h / 2.0
            # Find points on edges
            point_l_y = [(y_p, new_x) for y_p in np.linspace(new_y, new_y + new_h, num=5)]
            point_r_y = [(y_p, new_x + new_w) for y_p in np.linspace(new_y, new_y + new_h, num=5)]
            point_u_x = [(new_y, x_p) for x_p in np.linspace(new_x, new_x + new_w, num=5)]
            point_d_x = [(new_y + new_h, x_p) for x_p in np.linspace(new_x, new_x + new_w, num=5)]

            # NW
            nw_f = []
            a1, b1 = self.linear_coeffs([new_y_cntr, new_x_cntr], [point_u_x[1][0], point_u_x[1][1]])
            a2, b2 = self.linear_coeffs([new_y_cntr, new_x_cntr], [point_l_y[1][0], point_l_y[1][1]])
            for x_i in xrange(new_x, new_x + new_w / 2):
                up_const = int(a1 * x_i + b1) if int(a1 * x_i + b1) > new_y else new_y
                dn_const = int(a2 * x_i + b2)
                for y_i in xrange(up_const, dn_const):
                    nw_f.append((y_i, x_i))

            local_rels['NW'] = self.ratio_indexing(nw_f, idx)

            # N
            n_f = []
            a1, b1 = self.linear_coeffs([new_y_cntr, new_x_cntr], [point_u_x[1][0], point_u_x[1][1]])
            a2, b2 = self.linear_coeffs([new_y_cntr, new_x_cntr], [point_u_x[3][0], point_u_x[3][1]])
            for x_i in xrange(int(point_u_x[1][1]), int(point_u_x[2][1])):
                up_const = int(a1 * x_i + b1) if int(a1 * x_i + b1) > new_y else new_y
                for y_i in xrange(new_y, up_const):
                    n_f.append((y_i, x_i))
            for x_i in xrange(int(point_u_x[2][1]), int(point_u_x[3][1])):
                up_const = int(a2 * x_i + b2) if int(a2 * x_i + b2) > new_y else new_y
                for y_i in xrange(new_y, up_const):
                    n_f.append((y_i, x_i))

            local_rels['N'] = self.ratio_indexing(n_f, idx)

            # NE
            ne_f = []
            a1, b1 = self.linear_coeffs([new_y_cntr, new_x_cntr], [point_u_x[3][0], point_u_x[3][1]])
            a2, b2 = self.linear_coeffs([new_y_cntr, new_x_cntr], [point_r_y[1][0], point_r_y[1][1]])
            for x_i in xrange(new_x + new_w / 2, new_x + new_w):
                up_const = int(a1 * x_i + b1) if int(a1 * x_i + b1) > new_y else new_y
                dn_const = int(a2 * x_i + b2)
                for y_i in xrange(up_const, dn_const):
                    ne_f.append((y_i, x_i))

            local_rels['NE'] = self.ratio_indexing(ne_f, idx)

            # E
            e_f = []
            a1, b1 = self.linear_coeffs([new_y_cntr, new_x_cntr], [point_r_y[1][0], point_r_y[1][1]])
            a2, b2 = self.linear_coeffs([new_y_cntr, new_x_cntr], [point_r_y[3][0], point_r_y[3][1]])
            for x_i in xrange(new_x + new_w / 2, new_x + new_w):
                up_const = int(a1 * x_i + b1)
                dn_const = int(a2 * x_i + b2)
                for y_i in xrange(up_const, dn_const):
                    e_f.append((y_i, x_i))

            local_rels['E'] = self.ratio_indexing(e_f, idx)

            # SE
            se_f = []
            a1, b1 = self.linear_coeffs([new_y_cntr, new_x_cntr], [point_r_y[3][0], point_r_y[3][1]])
            a2, b2 = self.linear_coeffs([new_y_cntr, new_x_cntr], [point_d_x[3][0], point_d_x[3][1]])
            for x_i in xrange(new_x + new_w / 2, new_x + new_w):
                up_const = int(a1 * x_i + b1)
                dn_const = int(a2 * x_i + b2) if int(a2 * x_i + b2) < (new_y + new_h) else (new_y + new_h)
                for y_i in xrange(up_const, dn_const):
                    se_f.append((y_i, x_i))

            local_rels['SE'] = self.ratio_indexing(se_f, idx)

            # S
            s_f = []
            a1, b1 = self.linear_coeffs([new_y_cntr, new_x_cntr], [point_d_x[1][0], point_d_x[1][1]])
            a2, b2 = self.linear_coeffs([new_y_cntr, new_x_cntr], [point_d_x[3][0], point_d_x[3][1]])
            for x_i in xrange(int(point_d_x[1][1]), int(point_d_x[2][1])):
                up_const = int(a1 * x_i + b1)
                for y_i in xrange(up_const, new_y + new_h):
                    s_f.append((y_i, x_i))
            for x_i in xrange(int(point_d_x[2][1]), int(point_d_x[3][1])):
                up_const = int(a2 * x_i + b2)
                for y_i in xrange(up_const, new_y + new_h):
                    s_f.append((y_i, x_i))

            local_rels['S'] = self.ratio_indexing(s_f, idx)

            # SW
            sw_f = []
            a1, b1 = self.linear_coeffs([new_y_cntr, new_x_cntr], [point_l_y[3][0], point_l_y[3][1]])
            a2, b2 = self.linear_coeffs([new_y_cntr, new_x_cntr], [point_d_x[1][0], point_d_x[1][1]])
            for x_i in xrange(new_x, new_x + new_w / 2):
                up_const = int(a1 * x_i + b1)
                dn_const = int(a2 * x_i + b2) if int(a2 * x_i + b2) < (new_y + new_h) else (new_y + new_h)
                for y_i in xrange(up_const, dn_const):
                    sw_f.append((y_i, x_i))

            local_rels['SW'] = self.ratio_indexing(sw_f, idx)

            # W
            w_f = []
            a1, b1 = self.linear_coeffs([new_y_cntr, new_x_cntr], [point_l_y[1][0], point_l_y[1][1]])
            a2, b2 = self.linear_coeffs([new_y_cntr, new_x_cntr], [point_l_y[3][0], point_l_y[3][1]])
            for x_i in xrange(new_x, new_x + new_w / 2):
                up_const = int(a1 * x_i + b1)
                dn_const = int(a2 * x_i + b2)
                for y_i in xrange(up_const, dn_const):
                    w_f.append((y_i, x_i))

            local_rels['W'] = self.ratio_indexing(w_f, idx)
            relations[int(idx)] = local_rels
        return relations


if __name__ == '__main__':
    pass