import os
import numpy as np
import tensorflow as tf


def load_data(dir):
    TRAIN_DATASET_NAME = "train_dataset"
    VAL_DATASET_NAME = "val_dataset"
    TEST_DATASET_NAME = "test_dataset"
    INPUT_SHAPE_NAME = "input_shape.npy"
    OUTPUT_SHAPE_NAME = "output_shape.npy"
    ADDITIONAL_INFO_FILE_NAME = "additional_info.npy"
    train = tf.data.experimental.load(os.path.join(dir, TRAIN_DATASET_NAME), compression="GZIP")
    val = tf.data.experimental.load(os.path.join(dir, VAL_DATASET_NAME), compression="GZIP")
    test = tf.data.experimental.load(os.path.join(dir, TEST_DATASET_NAME), compression="GZIP")
    input_shape = np.load(os.path.join(dir, INPUT_SHAPE_NAME), allow_pickle=True)
    output_shape = np.load(os.path.join(dir, OUTPUT_SHAPE_NAME), allow_pickle=True)
    additional_info = None
    if os.path.isfile(os.path.join(dir, ADDITIONAL_INFO_FILE_NAME)):
        additional_info = np.load(os.path.join(dir, ADDITIONAL_INFO_FILE_NAME), allow_pickle=True)[0]
    return train, val, test, input_shape, output_shape, additional_info