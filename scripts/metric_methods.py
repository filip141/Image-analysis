import copy
import numpy as np


def euclidean_dist(col_one, col_two):
    return np.sqrt((col_one[0] - col_two[0])**2 + (col_one[1] - col_two[1])**2 + (col_one[2] - col_two[2])**2)


def meas_sim_texture_desc(obs_one, obs_two):
    dist = 0
    weight = 20
    for feat_o, feat_t in zip(obs_one, obs_two):
        if type(feat_o) != type(feat_t):
            raise ValueError("Observation features should have same type!")
        if isinstance(feat_o, dict):
            dist += abs(feat_o['e'] - feat_t['e'])**2 * weight / 2.0
            dist += abs(feat_o['d'] - feat_t['d'])**2 * weight
        elif isinstance(feat_o, float):
            dist += (feat_o - feat_t)**2 * weight
        else:
            raise ValueError("Wrong attribute type!")
    dist = np.sqrt(dist)
    return dist


def meas_sim_shape(obs_one, obs_two):
    dist = 0
    for n_feat_o, n_feat_t in zip(obs_one, obs_two):
        for s_om_o, s_om_t in zip(n_feat_o, n_feat_t):
            dist += (s_om_o - s_om_t)**2
    dist = np.sqrt(dist)
    return dist


def meas_sim_col(obs_one, obs_two):
    percentage_weight = 600
    spatial_coh_weight = 600
    obo = copy.deepcopy(obs_one.values())
    obt = copy.deepcopy(obs_two.values())
    dist = 0
    for cluster in obo:
        clust_o_mean = cluster[0]
        sim_lst = []
        for cl_sus in obt:
            cl_sus_mean = cl_sus[0]
            sim_lst.append(euclidean_dist(clust_o_mean, cl_sus_mean))
        min_val = min(sim_lst)
        min_idx = sim_lst.index(min_val)
        dist += min_val
        dist += euclidean_dist(cluster[1], obt[min_idx][1])
        dist += (cluster[2] - obt[min_idx][2])**2 * percentage_weight
        dist += (cluster[3] - obt[min_idx][3])**2 * spatial_coh_weight
        obt = [elem for idx, elem in enumerate(obt) if idx != min_idx]
    dist = np.sqrt(dist)
    return dist