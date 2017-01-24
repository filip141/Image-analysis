import os
import csv
import json
import copy
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

prohibited_class = ['nk', 'bc', 'sie']


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
            if segment_name in prohibited_class:
                continue
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

    # Create classifier folder
    descriptor_dir = "classifier"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.split(dir_path)[0]
    desc_path = os.path.join(dir_path, "data", descriptor_dir)
    if not os.path.exists(desc_path):
        os.mkdir(desc_path)
    # Normalization
    print "Data Normalization..."
    base_mean = np.mean(base_dataset, axis=0)
    base_std = np.std(base_dataset, axis=0)
    base_dataset = (base_dataset - base_mean) / base_std
    # Remove Nans
    nan_idx = np.isnan(base_dataset)
    base_dataset[nan_idx] = 0
    # Train Test classifier
    print "Training model..."
    X_train, X_test, y_train, y_test = train_test_split(base_dataset, seg_classes, test_size=0.4, random_state=0)
    test_clf = svm.SVC(gamma=0.001, C=100, probability=False)
    test_clf.fit(X_train, y_train)
    # Check Test Set
    print "Evaluate model..."
    y_pred = []
    for pred_vec in X_test:
        y_pred.append(test_clf.predict([pred_vec]))
    print "Confusion Matrix"
    print confusion_matrix(y_test, y_pred)
    print test_clf.score(X_test, y_test)
    # Learn with full set
    print "Training using full dataset"
    clf = svm.SVC(gamma=0.001, C=100, probability=True)
    clf.fit(base_dataset, seg_classes)
    print "Saving data to CSV..."
    csv_path = os.path.join(desc_path, "base_low_descriptors.csv")
    with open(csv_path, 'w') as fp:
        csv_basegen = csv.writer(fp, delimiter=',')
        csv_basegen.writerows(base_dataset)
    print "Saving Training set mean and std"
    norm_path = os.path.join(desc_path, "norm_coeff.npz")
    np.savez(norm_path, base_mean=base_mean, base_std=base_std)
    print "Saving SVM to File..."
    model_path = os.path.join(desc_path, "low_desc_svm.pkl")
    joblib.dump(clf, model_path)


if __name__ == '__main__':
    print "Reading Data from files..."
    prepare_dataset("../data/descriptors")