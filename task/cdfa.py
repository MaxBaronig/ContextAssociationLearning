import itertools
import time
from xml.sax.handler import feature_external_ges
import numpy as np
from numpy.random import default_rng, Generator
from scipy.special import comb
import sklearn
import tensorflow as tf
from tensorflow._api.v2 import data, random
from tensorflow.python.eager.context import num_gpus
from tensorflow.python.framework.tensor_spec import TensorSpec
from tensorflow.python.ops.gen_array_ops import shape
# from utils.misc import slice_dataset
import plotly.express as px
import plotly.io as pio   
pio.kaleido.scope.mathjax = None


def slice_dataset(dataset, num_samples, test_frac, val_frac,  **kwargs):
    test_count = int(num_samples * test_frac)
    val_count = int(num_samples * val_frac)

    test = dataset.take(test_count) 
    val = dataset.skip(test_count).take(val_count)
    train = dataset.skip(test_count + val_count)
    return train, val, test
    
def generate_features(num_features, num_neurons_active_per_feature, num_neurons_per_feature, num_cases_per_feature, rng : Generator):
    features = []
    for idx in range(num_features):
        cases = []
        while len(cases) < num_cases_per_feature:
            feature = np.zeros(num_neurons_per_feature)
            indices = rng.choice(np.arange(feature.size), size=num_neurons_active_per_feature, replace=False)
            feature[indices] = 1.0
            if not arreq_in_list(feature, cases):
                cases.append(feature)
        features.append(cases)        
    return features


#test for exact equality
def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)


def generate_class_indices(num_features, num_classes, num_features_per_class, num_cases_per_feature, rng):
    classes = []
    max_combinations = comb(num_features * num_cases_per_feature, num_features_per_class, exact=True)
    if num_classes > max_combinations:
        raise Exception("Task cannot be created with this number of classes and features per class!")
    while len(classes) < num_classes:
        feature_indices = np.sort(rng.choice(np.arange(num_features), size=num_features_per_class, replace=False))
        case_indices = rng.choice(np.arange(num_cases_per_feature), size=num_features_per_class, replace=True)
        indices = np.column_stack((feature_indices, case_indices))
        for d in classes:
            if np.array_equal(indices, d):
                break
        else:
            classes.append(indices)
    return classes

def get_class_vectors(features, class_indices):
    classes = []
    for c in class_indices:
        class_vector = []
        for idx, feature in enumerate(features):
            if idx in c:
                class_vector.append(feature)
            else:
                class_vector.append(np.full(len(feature), -1))
        classes.append(np.array(class_vector).flatten)
    return classes

def replace_indices_with_features(indices, features):
    return np.array([features[idx][val] for idx, val in enumerate(indices)])

def label_combinations(class_indices_list, count, all_combinations):
    combinations_hashmap = dict()
    for combination in all_combinations[:count]:
        labels = get_class_labels(combination, class_indices_list)
        for label in np.where(labels==1)[0]:
            if label not in combinations_hashmap:
                combinations_hashmap[label] = []
            combinations_hashmap[label].append(combination)
    return combinations_hashmap, all_combinations[count:]

def get_combination(label, combinations_hashmap):
    if not label in combinations_hashmap or not combinations_hashmap[label]:
        return []
    combination = combinations_hashmap[label][0]
    for l in combinations_hashmap:
        if combinations_hashmap[l] and np.array_equal(combination, combinations_hashmap[l][0]):
            combinations_hashmap[l].remove(combination)
    return combination

def generate_samples(num_samples, class_indices, features, multitask, random_counterexamples_per_task, num_cases_per_feature,  rng : Generator, autoencoding_pretraining):
    class_indices = np.array(class_indices)
    num_features = features.shape[0]
    fits_exactly = num_samples % class_indices.shape[0] == 0
    if not autoencoding_pretraining:
        if fits_exactly:
            num_samples_per_class = int(num_samples / len(class_indices))
            if num_samples_per_class % 2 != 0:
                raise Exception("Number of samples per class must be even to ensure balanced dataset!")
        else:  
            raise Exception("Number of samples must be divisible by number of classes!")
    X = []
    Y = []
    positive_class_cardinalities = np.zeros(len(class_indices))
    negative_class_cardinalities = np.zeros(len(class_indices))
    pos_ratio = 0.0
    number_of_all_combinations = num_cases_per_feature ** num_features
    assert(number_of_all_combinations < 10000000 and "Number of combinations too large to sample all of them!\
                     Implement a different method.")

    all_combinations = list(itertools.product(*[np.arange(num_cases_per_feature) for _ in range(num_features)]))
    rng.shuffle(all_combinations)
    all_combinations = np.array(all_combinations)
    if not autoencoding_pretraining:
        combinations_hashmap, all_combinations = label_combinations(class_indices, num_samples, all_combinations)
    n = 0
    while len(Y) < num_samples: 
        # if int(int(num_samples/(1 + random_counterexamples_per_task)) / n) * 100 % 10 == 0:
        # lowest_positive_class = np.argmin(positive_class_cardinalities)
        # lowest_negative_class = np.argmin(negative_class_cardinalities)
        lowest_positive_class = rng.choice(np.where(positive_class_cardinalities == np.min(positive_class_cardinalities))[0])
        lowest_negative_class = rng.choice(np.where(negative_class_cardinalities == np.min(negative_class_cardinalities))[0])
        if autoencoding_pretraining:
            x_indices_filled = all_combinations[n]
        else:
            x_indices_filled = []
            while len(x_indices_filled) == 0:
                x_indices_filled = get_combination(lowest_positive_class, combinations_hashmap)
                if len(x_indices_filled) == 0:
                    combinations_hashmap, all_combinations = label_combinations(class_indices, num_samples, all_combinations)
        x = replace_indices_with_features(x_indices_filled, features)
        labels = get_class_labels(x_indices_filled, class_indices)

        if not autoencoding_pretraining:
            positive_task_cue_oh = one_hot(lowest_positive_class, np.size(labels))
            potential_negative = [ind for ind, ele in enumerate(labels) if ele == 0]
            assert(lowest_positive_class not in potential_negative and "Something wrong!!")
            negative_task_cues = []
            while len(negative_task_cues) < random_counterexamples_per_task:
                assert(len(potential_negative) > 0)
                if lowest_negative_class in potential_negative:
                    neg = lowest_negative_class
                    potential_negative.remove(lowest_negative_class)
                else:
                    neg = rng.choice(potential_negative)
                    potential_negative.remove(neg)
                negative_task_cues.append( one_hot(neg, np.size(labels)))
            if positive_class_cardinalities[lowest_positive_class] < int(num_samples_per_class/2):
                X.append((x.flatten(), positive_task_cue_oh))
                Y.append([1.0])
                positive_class_cardinalities += positive_task_cue_oh


            for neg in negative_task_cues:
                if negative_class_cardinalities[np.argmax(neg)] < int(num_samples_per_class/2):
                    X.append((x.flatten(), neg))
                    Y.append([0.0])
                    negative_class_cardinalities += neg

        else:
            target_with_all_features = np.zeros((num_features, num_cases_per_feature))
            target_with_all_features[np.arange(len(x_indices_filled)),x_indices_filled] = 1.0
            target = target_with_all_features.flatten()
            Y.append(target)
            X.append((x.flatten(), np.zeros(np.size(labels))))            
        n += 1
    return X, np.array(Y)



def get_class_labels(x, class_indices_list):
    labels = np.zeros(len(class_indices_list), dtype=np.float32)
    for class_id, class_indices in enumerate(class_indices_list):
        if contains_class(x, class_indices):
            labels[class_id] = 1.0
    return labels

def contains_class(x, class_indices):
    return np.all(np.array_equal(x[class_indices[:,0]], class_indices[:,1]))

def one_hot(idx, size):
    b = np.zeros((size))
    b[idx] = 1
    idx_onehot = b
    return idx_onehot


def reformat_x(X):
    return (np.array([x[0] for x in X]), np.array([x[1] for x in X]))

def set_class_features(init, class_indices):
    return [class_indices[np.where(class_indices[:,0]==i)][0,1] if i in class_indices[:, 0] else init[i] for i, val in enumerate(init)]


def generate_acdc_dataset(num_samples=1000, num_classes=100, num_features=8, 
                    num_neurons_active_per_feature=3, num_neurons_per_feature=6, num_features_per_class=3, num_cases_per_feature=6, random_seed=69, multitask=True, random_counterexamples_per_task=1, autoencoding_pretraining=False, **kwargs):
    rng = default_rng(random_seed)
    features = generate_features(num_features, num_neurons_active_per_feature, num_neurons_per_feature, num_cases_per_feature, rng)
    class_indices = generate_class_indices(num_features, num_classes, num_features_per_class, num_cases_per_feature, rng)
    X, Y = generate_samples(num_samples, class_indices, np.array(features), multitask, random_counterexamples_per_task, num_cases_per_feature, rng, autoencoding_pretraining=autoencoding_pretraining)
    shuffler = rng.permutation(len(X))
    X = [X[i] for i in shuffler]
    Y = [Y[i] for i in shuffler]
    if multitask:
        X = reformat_x(X)
    
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    
    (train, val, test) = slice_dataset(dataset, num_samples, **kwargs)

    if multitask:
        input_shape = (dataset.element_spec[0][0].shape.as_list()[0], dataset.element_spec[0][1].shape.as_list()[0])
        output_shape = 1
    else:
        raise NotImplementedError()
    
    additional_material = (features, class_indices)
    return train, val, test, input_shape, output_shape, additional_material




if __name__ == "__main__":
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly as plt
    import plotly.io as pio


    pio.templates.default = "simple_white"
    tf.random.set_seed(69)
    num_classes = 10
    train, val, test, input_shape, output_shape, additional_material = generate_acdc_dataset(num_samples=20, num_features=5, num_cases_per_feature=3, num_neurons_per_feature=4, num_neurons_active_per_feature=2, num_features_per_class=2, num_classes=2, test_frac=0.1, val_frac=0.1, autoencoding_pretraining=False)
    fig = plot_additional_material(additional_material)
    fig.write_image("tasks/ACDC/images/additional_material.pdf", width=1200, height=300)
    sample = 0
    for x in train.skip(1).take(4):
        sample += 1
        fig = plot_sample(x, additional_material, True)
        fig.write_image("tasks/ACDC/images/sample{}.pdf".format(sample), width=240, height=300)

    exit()
