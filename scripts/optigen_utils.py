import os
import cv2
import json
import copy
import numpy as np
from rag_segmentation import RAGSegmentation
from mpeg7Descriptors import MPEG7Descriptors
from spatialextract import SpatialExtractor

__directions = 8
side2id = {"E": 0, "N": 1, "NE": 2, "NW": 3, "S": 4, "SE": 5, "SW": 6, "W": 7}
descriptor_dir = "examples"
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.split(dir_path)[0]
desc_path = os.path.join(dir_path, "data", descriptor_dir)


def prepare_image(img):
    default_wres = 320
    im_res = img.shape[:-1]
    factor = im_res[0] / float(im_res[1])
    n_image = cv2.resize(img, (default_wres, int(default_wres * factor)))
    return n_image


def extract_descriptors(n_image, dominant_col=8, c_param=0.85, saveToJSON=False, loadFromJson=False, file_name=None):
    if loadFromJson:
        if file_name is None:
            raise ValueError("When loadFromJson is true path cant be None!")
        json_path = os.path.join(desc_path, "{}.json".format(file_name))
        with open(json_path, 'r') as fp:
            json_res = json.load(fp)
            new_ll = {}
            for key, value in json_res['low_level'].iteritems():
                new_ll[int(key)] = value
            json_res['low_level'] = new_ll
            return json_res

    # Image Segmentation
    rag = RAGSegmentation(n_image, slic_clust_num=200, slic_cw=15, median_blur=7)
    t_clusters = rag.run_slic()
    clust_col_t = rag.slic_mean_lab(t_clusters)
    cn = rag.neighbours_regions(t_clusters)
    ed = rag.find_edges(cn, clust_col_t)
    concat_params = rag.concat_similar_regs(ed, t_clusters, c_factor=0.65)
    n_clusters = concat_params[0]
    # Mpeg 7 descriptors
    mpeg = MPEG7Descriptors(n_clusters, n_image)
    # Spatial Extractor
    spatial = SpatialExtractor(n_clusters)
    # Extract features
    color_desc = mpeg.mpeg7_dominant_colours(dominant_col)
    tex_desc = mpeg.mpeg7_homogeneus_texture()
    shape_desc = mpeg.mpeg7_region_shape()
    spatial_rels = spatial.find_spatial(c_param)

    # Low Level Descriptors Extraction
    seg_desc = {}
    segment_set = set(color_desc.keys())
    for seg_idx in segment_set:
        record = []
        # Add texture features
        seg_tex_des = tex_desc[seg_idx]
        for tex_val in seg_tex_des[:-2]:
            record.append(tex_val['e'])
            record.append(tex_val['d'])
        record += seg_tex_des[-2:]
        # Add shape features
        seg_shape_des = shape_desc[seg_idx]
        for shape_val in seg_shape_des:
            record += shape_val
        seg_col_des = color_desc[seg_idx]
        ex_dataset = []
        for clust_key, clust_val in seg_col_des.items():
            new_rec = copy.deepcopy(record)
            for cl_elem in clust_val:
                if isinstance(cl_elem, list):
                    new_rec += cl_elem
                else:
                    new_rec.append(cl_elem)
            if np.isnan(new_rec).any():
                continue
            ex_dataset.append(new_rec)
        seg_desc[seg_idx] = ex_dataset

    # Extract Spatial Relations
    rel_matrix = np.zeros((len(segment_set), len(segment_set), __directions))
    for ref_seg_id, ref_seg_val in spatial_rels.iteritems():
        for rkey, rval in ref_seg_val.iteritems():
            side_id = side2id[rkey]
            for p_seg_k, p_seg_val in rval.iteritems():
                rel_matrix[ref_seg_id, p_seg_k, side_id] += p_seg_val

    rel_mat_sum = np.sum(rel_matrix, axis=2)
    rel_mat_sum = np.expand_dims(rel_mat_sum, axis=2)
    rel_mat_sum = np.tile(rel_mat_sum, (1, 1, __directions))
    rel_matrix = rel_matrix / rel_mat_sum
    nan_idx = np.isnan(rel_matrix)
    rel_matrix[nan_idx] = 0

    result = {"spatial_rels": rel_matrix, "low_level": seg_desc, "segments": list(segment_set)}

    if saveToJSON:
        if file_name is None:
            raise ValueError("When saveToJson is true path cant be None!")
        if not os.path.exists(desc_path):
            os.mkdir(desc_path)
        json_path = os.path.join(desc_path, "{}.json".format(file_name))
        with open(json_path, 'w') as fp:
            result["spatial_rels"] = result["spatial_rels"].tolist()
            json.dump(result, fp)
    return result