import os
import json
import copy
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


def prepare_dataset(path):
    base_dataset = []
    name2id_list = []
    seg_classes = []
    files_in_dir = os.listdir(path)
    for file in files_in_dir:
        f_path = os.path.join(path, file)
        with open(f_path, "r") as fp:
            img_json = json.load(fp)
        color_desc = img_json['colour']
        shape_desc = img_json['shape']
        tex_desc = img_json['texture']
        segments = img_json['segments']
        name2id_list += list(set(segments.values()) - set(name2id_list))
        name2id = dict([(word, idx) for idx, word in enumerate(name2id_list)])
        segment_set = set(segments.keys())
        for seg_idx in segment_set:
            record = []
            # Add texture features
            seg_tex_des = tex_desc[seg_idx.split('.')[0]]
            for tex_val in seg_tex_des[:-2]:
                record.append(tex_val['e'])
                record.append(tex_val['d'])
            record += seg_tex_des[-2:]
            # Add shape features
            seg_shape_des = shape_desc[seg_idx.split('.')[0]]
            for shape_val in seg_shape_des:
                record += shape_val
            seg_col_des = color_desc[seg_idx.split('.')[0]]
            segment_name = segments[str(seg_idx)]
            class_id = name2id[segment_name]
            for clust_key, clust_val in seg_col_des.items():
                new_rec = copy.deepcopy(record)
                for cl_elem in clust_val:
                    if isinstance(cl_elem, list):
                        new_rec += cl_elem
                    else:
                        new_rec.append(cl_elem)
                if np.isnan(new_rec).any():
                    continue
                base_dataset.append(new_rec)
                seg_classes.append(class_id)

    X_train, X_test, y_train, y_test = train_test_split(base_dataset, seg_classes, test_size=0.4, random_state=0)
    clf = svm.SVC(gamma=0.001, C=100)
    scores = cross_val_score(clf, base_dataset, seg_classes, cv=10)
    print scores
    # import csv
    # with open('test.csv', 'w') as fp:
    #     a = csv.writer(fp, delimiter=',')
    #     a.writerows(base_dataset)
                #
            # print "aa"

if __name__ == '__main__':
    prepare_dataset("../data/descriptors")