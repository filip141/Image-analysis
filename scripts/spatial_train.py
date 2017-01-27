import argparse
import os
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")
side2id = {"E": 0, "N": 1, "NE": 2, "NW": 3, "S": 4, "SE": 5, "SW": 6, "W": 7}
prohibited_class = ['nk', 'bc', 'sie']
__directions = 8


def prepare_spatial(path):
    name2id_list = []
    files_in_dir = os.listdir(path)
    rels_per_base = []
    for file in files_in_dir:
        f_path = os.path.join(path, file)
        with open(f_path, "r") as fp:
            img_json = json.load(fp)
        spatial_rels = img_json['relations']
        segments = img_json['segments']
        name2id_list += list(set(segments.values()) - set(name2id_list))
        name2id = dict([(word, idx) for idx, word in enumerate(name2id_list)])
        rel_matrix = np.zeros((len(name2id_list), len(name2id_list), __directions))

        for ref_seg_id, ref_seg_val in spatial_rels.iteritems():
            ref_seg_name = segments[str(ref_seg_id)]
            rref_id = name2id[ref_seg_name]
            for rkey, rval in ref_seg_val.iteritems():
                side_id = side2id[rkey]
                for p_seg_k, p_seg_val in rval.iteritems():
                    p_seg_name = segments[str(p_seg_k)]
                    rp_id = name2id[p_seg_name]
                    rel_matrix[rref_id, rp_id, side_id] += p_seg_val
        rel_mat_sum = np.sum(rel_matrix, axis=2)
        rel_mat_sum = np.expand_dims(rel_mat_sum, axis=2)
        rel_mat_sum = np.tile(rel_mat_sum, (1, 1, __directions))
        rel_matrix = rel_matrix / rel_mat_sum
        nan_idx = np.isnan(rel_matrix)
        rel_matrix[nan_idx] = 0
        rels_per_base.append(rel_matrix)
    # Create classifier folder
    descriptor_dir = "classifier"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.split(dir_path)[0]
    desc_path = os.path.join(dir_path, "data", descriptor_dir)
    if not os.path.exists(desc_path):
        os.mkdir(desc_path)
    print "Generating spatial relation for database..."
    final_matrix = np.zeros(rels_per_base[-1].shape)
    for rel_mat in rels_per_base:
        h, w, d = rel_mat.shape
        final_matrix[0: h, 0: w, 0: d] += rel_mat
    final_matrix /= len(rels_per_base)
    rels_path = os.path.join(desc_path, "spatial_relations.npz")
    np.savez(rels_path, spatial_relations=final_matrix, name2id=name2id.items())


if __name__ == '__main__':
        # Script description
    description = 'Script from Image-analysis package to create spatial relations matrix\n'

    # Set command line arguments
    parser = argparse.ArgumentParser(description)
    parser.add_argument('-ph', '--path', dest='path', action='store', default="../data/base")
    args = parser.parse_args()

    data_path = args.path

    print "Reading Spatial Data from files..."
    prepare_spatial(data_path)