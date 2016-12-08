import os
import cv2
import random
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from metric_methods import spatial_distance
from optigen_utils import prepare_image, extract_descriptors, plot

prohibited_class = ['nk', 'bc', 'sie']
descriptor_dir = "classifier"
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.split(dir_path)[0]
desc_path = os.path.join(dir_path, "data", descriptor_dir)


class DescriptorData:

    def __init__(self, descr, clf, spatial_rel, classes):
        self.descriptors = descr
        self.classifier = clf
        self.relations = np.array(spatial_rel)
        self.classes = classes


class Chromosome(object):

    alpha = 0.5

    def __init__(self, desc_data):
        self.descriptors = desc_data.descriptors
        self.spatial_rels = desc_data.relations
        self.svm_clf = desc_data.classifier
        self.classes = desc_data.classes
        self.fitness_attrs = None
        self.classify = []

    def gen_random(self):
        self.classify = []
        for _ in xrange(0, len(self.descriptors["segments"])):
            self.classify.append(random.choice(self.classes))

    def update_fitness_attrs(self):
        ll = []
        sc_l = []
        spat_rels = self.descriptors['spatial_rels']
        # Normalization coeffs
        base_mean, base_std = self.load_normal_coeff()
        for seg_id, pred_cls in enumerate(self.classify):

            # ##########################################
            # #########Spatial relations################
            # ##########################################

            seg_spat = spat_rels[seg_id]
            base_spat = self.spatial_rels[pred_cls]
            d = []
            for sp_id, spat_rel in enumerate(seg_spat):
                d.append(spatial_distance(spat_rel, base_spat[self.classify[sp_id]]))
            di = 1.0 - np.array(d)
            sr = np.sum(di) / di.size
            sc_l.append(sr)
            # ##########################################
            # #########Low level descriptors############
            # ##########################################

            colour_mult = 1
            segs = self.descriptors['low_level'][seg_id]
            for pred in segs:
                # Normalize
                base_record = (pred - base_mean) / base_std
                nan_idx = np.isnan(base_record)
                base_record[nan_idx] = 0
                # Predict proba
                pred_res = self.svm_clf.predict_proba([base_record])[0][self.classes.index(pred_cls)]
                colour_mult *= pred_res
            ll.append(colour_mult**(1 / float(len(segs))))
        sc = np.sum(sc_l) / len(sc_l)
        fs = np.sum(ll) / len(ll)
        self.fitness_attrs = (sc, fs)

    def set_classify(self, clsfy):
        self.classify = clsfy

    @staticmethod
    def load_normal_coeff():
        norm_path = os.path.join(desc_path, "norm_coeff.npz")
        with np.load(norm_path) as X:
            return X["base_mean"], X["base_std"]

    @staticmethod
    def crossover(chr_o, chr_t):
        # Random point to crossover
        pos = random.randint(0, len(chr_o.classify))
        classify_o = chr_o.classify
        classify_t = chr_t.classify
        # Crossover
        class_new_o = classify_o[: pos] + classify_t[pos:]
        class_new_t = classify_t[: pos] + classify_o[pos:]
        # Create new Chromosomes
        desc = DescriptorData(chr_o.descriptors, chr_o.svm_clf, chr_o.spatial_rels, chr_o.classes)
        new_chr_o = Chromosome(desc)
        new_chr_o.set_classify(class_new_o)
        new_chr_t = Chromosome(desc)
        new_chr_t.set_classify(class_new_t)
        return new_chr_o, new_chr_t


class Population(object):

    def __init__(self, size, crossover, mutation, desc_data):
        self.species = []
        self.fitness = 0
        self.size = size
        self.mutation = mutation
        self.crossover = crossover
        self.desc_data = desc_data

    def init_population(self):
        for chr_id in xrange(0, self.size):
            chrom = Chromosome(self.desc_data)
            chrom.gen_random()
            chrom.update_fitness_attrs()
            self.species.append(chrom)

    def set_species(self, species):
        self.species = species

    def get_grade(self):
        return self.fitness

    def generate_population(self):
        fitness_at_list = np.array([ch.fitness_attrs for ch in self.species])
        fitness_at_list = ((fitness_at_list - np.mean(fitness_at_list, axis=0)) / np.std(fitness_at_list, axis=0))
        fitness_at_list = fitness_at_list + np.abs(np.min(fitness_at_list, axis=0))
        fitness_at_list = np.sum(fitness_at_list * [Chromosome.alpha, 1 - Chromosome.alpha], axis=1)
        fitness_sum = np.sum(fitness_at_list)
        fitness_at_list = fitness_at_list / fitness_sum
        cum_sum = np.cumsum(fitness_at_list, axis=0)
        new_species = []

        print "Overall Population Rate: {}".format(fitness_sum)
        for _ in range(0, self.size):
            rnd = random.uniform(0, 1)
            # If small than first
            if rnd < cum_sum[0]:
                new_species.append(self.species[0])
                continue
            # for others
            counter = 0
            while rnd > cum_sum[counter]:
                counter += 1
            new_species.append(self.species[counter])

        # # CROSSOVER
        species_nc = []
        crossover_list = []
        for n_chrom in new_species:
            rnd = random.uniform(0, 1)
            if rnd < self.crossover:
                crossover_list.append(n_chrom)
            else:
                species_nc.append(n_chrom)

        crossover_tuples = []
        cr_iterate = list(enumerate(crossover_list))
        while cr_iterate:
            cch_idx, c_chrom = cr_iterate.pop()
            if not cr_iterate:
                species_nc.append(c_chrom)
                break
            cb_idx, cross_buddy = random.choice(cr_iterate)
            cr_iterate = [(x_k, x_v) for x_k, x_v in cr_iterate if x_k != cb_idx]
            crossover_tuples.append((c_chrom, cross_buddy))

        # Crossover to list
        after_cover = []
        for cr_tup in crossover_tuples:
            cr_o, cr_t = Chromosome.crossover(cr_tup[0], cr_tup[1])
            after_cover.append(cr_o)
            after_cover.append(cr_t)

        # New population
        new_species = after_cover + species_nc

        ## MUTATION
        for ch_idx in xrange(0, len(new_species)):
            for b_idx in xrange(0, len(new_species[ch_idx].classify)):
                rnd = random.uniform(0, 1)
                if rnd < self.mutation:
                    rnd_choice = random.choice(new_species[ch_idx].classes)
                    new_species[ch_idx].classify[b_idx] = rnd_choice

        # Update fitness
        for ch_idx in xrange(0, len(new_species)):
            new_species[ch_idx].update_fitness_attrs()

        new_pop = Population(self.size, self.crossover, self.mutation, self.desc_data)
        new_pop.set_species(new_species)
        return new_pop


class OptiGen(object):

    def __init__(self, generations, size=200, crossover=0.7, mutation=0.008):
        # ## Define Genetic Algoritm properties
        self.size = size
        self.generations = generations
        self.crossover = crossover
        self.mutation = mutation
        # ## Define Descriptor Properties
        self.clf = self.prepare_svm(desc_path)
        self.name2id, self.spatial_rels = self.prepare_spatial_matrix(desc_path)
        self.id2name = dict([(int(arr[::-1][0]), arr[::-1][1]) for arr in self.name2id])
        self.name2id = dict(self.name2id)
        proh_idx = [int(self.name2id[pr]) for pr in prohibited_class]
        self.classes = np.sort([x for x in self.id2name.keys() if x not in proh_idx]).tolist()

    def predict(self, image):
        # ## Image Processing
        image = prepare_image(image)
        image_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        descriptors = extract_descriptors(image, loadFromJson=False, file_name="road_example")
        desc_data = DescriptorData(descriptors, self.clf, self.spatial_rels, self.classes)

        # ## Genetic algorithm
        popultaion = Population(self.size, self.crossover, self.mutation, desc_data)
        popultaion.init_population()
        counter = 0
        while counter < self.generations:
            print "Generation: {}".format(counter)
            popultaion = popultaion.generate_population()
            print "________________________________________________________________"
            counter += 1
        descriptions = [self.id2name[x] for x in popultaion.species[0].classify]
        plot(image_rgb, descriptors, descriptions)

    @staticmethod
    def prepare_svm(path):
        model_path = os.path.join(path, "low_desc_svm.pkl")
        clf = joblib.load(model_path)
        return clf

    @staticmethod
    def prepare_spatial_matrix(path):
        rels_path = os.path.join(path, "spatial_relations.npz")
        with np.load(rels_path) as X:
            name2id = X["name2id"]
            rel_mat = X["spatial_relations"]
        return name2id, rel_mat

if __name__ == '__main__':
    # define folder destination
    test_image = cv2.imread('../data/road_2.jpg', 1)
    op = OptiGen(100)
    op.predict(test_image)
