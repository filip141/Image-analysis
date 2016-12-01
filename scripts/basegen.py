import os
import cv2
import json
import warnings
import matplotlib.pyplot as plt
from rag_segmentation import RAGSegmentation
from mpeg7Descriptors import MPEG7Descriptors

warnings.filterwarnings("ignore")
default_wres = 320


def generate_image_json(image, json_name, dominant_col=8):
    image_dict = {}
    descriptor_dir = "descriptors"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.split(dir_path)[0]
    desc_path = os.path.join(dir_path, "data", descriptor_dir)
    file_path = os.path.join(desc_path, "{}.json".format(json_name))
    if os.path.isfile(file_path):
        print "File already segmented, omitting..."
        return
    # Resize image
    im_res = image.shape[:-1]
    factor = im_res[0] / float(im_res[1])
    n_image = cv2.resize(test_image, (default_wres, int(default_wres * factor)))
    # Image Segmentation
    rag = RAGSegmentation(n_image, slic_clust_num=200, slic_cw=15, median_blur=7)
    t_clusters = rag.run_slic()
    clust_col_t = rag.slic_mean_lab(t_clusters)
    cn = rag.neighbours_regions(t_clusters)
    ed = rag.find_edges(cn, clust_col_t)
    concat_params = rag.concat_similar_regs(ed, t_clusters, c_factor=0.65)
    n_clusters = concat_params[0]
    # Make json folder
    if not os.path.exists(desc_path):
        os.mkdir(desc_path)
    # Mpeg 7 descriptors
    mpeg = MPEG7Descriptors(n_clusters, n_image)
    # label segments
    segment_names = {}
    segment_regions = {}
    segment_list = mpeg.segment_generator(one_dim=False)
    for segment in segment_list:
        print "Waiting for new figure..."
        # Show figure
        plt.figure()
        plt.axis("off")
        plt.title("Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.figure()
        plt.axis("off")
        plt.title("Segment number {}".format(segment[1]))
        plt.imshow(cv2.cvtColor(segment[0], cv2.COLOR_BGR2RGB))
        plt.show()
        # Ask for label
        new_label = raw_input("Name image segment: ")
        segment_names[int(segment[1])] = new_label
        segment_regions[int(segment[1])] = segment[0].tolist()
    # Extract features
    colors_desc = mpeg.mpeg7_dominant_colours(dominant_col)
    texture_desc = mpeg.mpeg7_homogeneus_texture()
    shape_desc = mpeg.mpeg7_region_shape()
    image_dict["colour"] = colors_desc
    image_dict["texture"] = texture_desc
    image_dict["shape"] = shape_desc
    image_dict["segments"] = segment_names
    image_dict["segment_regions"] = segment_regions
    print "Writing data to JSON File..."
    with open(file_path, "w") as jfp:
        json.dump(image_dict, jfp)


if __name__ == '__main__':
    base_dir = '../data/base'
    files_in_dir = os.listdir(base_dir)
    for img_file in files_in_dir:
        print "Segmentng file: {}".format(img_file)
        test_image = cv2.imread(os.path.join(base_dir, img_file), 1)
        generate_image_json(test_image, img_file.split('.')[0])
    print "Training Base Created."
