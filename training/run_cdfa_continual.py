import logging
import traceback
import tensorflow as tf
from task.utils import load_data
from hydra.utils import instantiate
from hydra.core.utils import _save_config
import os
import numpy as np
# import mlflow
import aim
from aim import Run
import argparse
from model.network import Network
from config.cdfa_multitask_config import cdfa_multitask_config
from run_cdfa_multitask import run_cdfa_experiment

""" This code runs the CDFA experiment for both the multi-task (Section 2.5) and the continual learning setting (Section 2.6)."""
def run_cdfa_continual(experiment, data_base_path, random_seed, config):  
    # initialize random state
    tf.keras.backend.clear_session()  
    tf.random.set_seed(random_seed)  
    tf.random.set_global_generator(tf.random.Generator.from_seed(random_seed))

    # set paths
    pretraining_dataset_path = os.path.join(data_base_path, "pretraining")
    training_dataset_path = os.path.join(data_base_path, experiment)

    # Construct Model
    model = Network(random_seed=random_seed, **config.model)
    
    # Construct Dataset for unsupervised pretraining of basal weights
    pretraining_dataset, _, _, _, _, additional_task_info = load_data(
        pretraining_dataset_path
    )

    # Basal weight pretraining
    unsupervised_basal_weight_pretraining(
        model,
        pretraining_dataset,
        **config.pretraining_parameters,
    )

    batch_size = config.training_parameters.batch_size

    # Construct Dataset
    # if we do continual learning, we have a separate training and test set for each task
    all_tasks_training_set, all_tasks_testing_set, input_shape, output_shape, _ = load_data(
        training_dataset_path
    )
    all_tasks_testing_set_batched = all_tasks_testing_set.batch(batch_size).cache().prefetch(100 * batch_size)

    # split up the training and test set into separate sets for each task
    training_sets, testing_sets, input_shape, target_shape = split_by_context(
        (all_tasks_training_set, all_tasks_testing_set), 
        config.task_incremental_training_parameters.classes_for_first_task
    )
    
    # in total we have num_tasks tasks. We consider the first task as one, even though it contains multiple classes
    num_tasks = len(training_sets)
    
    test_results_per_task =[]
    for task_id in range(num_tasks):

        # get the training and test set for the current task
        training_set = training_sets[task_id]
        testing_set = testing_sets[task_id]
        
        # batch the datasets and preload
        batched_train = training_set.batch(batch_size).cache().prefetch(100 * batch_size)
        batched_test = testing_set.batch(batch_size).cache().prefetch(100 * batch_size)

        # only if it is the first task ('pre' in Figure 7A) we train the readout neuron
        # disable gradient descent for the readout neuron for all other tasks
        if task_id > 0:
            model.optimizer.learning_rate=0.0
            model.optimizer.lr=0.0
        
        # fit calls model.train_step
        model.fit(
            x=batched_train, 
            epochs=config.training_parameters.epochs, 
            verbose=2, 
            use_multiprocessing=True
        )

        # evaluate model on each of the tasks, even the ones not trained on yet and the global test set
        test_result_per_class = []
        for testing_set in testing_sets:
            test_result_per_class.append(
                model.evaluate(
                    x=testing_set.batch(batch_size).cache().prefetch(100 * batch_size), 
                    verbose=2, 
                    return_dict=True
                )
            )
            
        print("test accuracy on class-specific test sets: {}".format([test_result["accuracy"] for test_result in test_result_per_class]))
        
        test_result_on_all_tasks = model.evaluate(
            x=all_tasks_testing_set_batched, 
            verbose=2, 
            return_dict=True
        )
        print("test result on global test set: {}".format(test_result_on_all_tasks))
        test_results_per_task.append((
            test_result_per_class,
            test_result_on_all_tasks
        ))

    return test_results_per_task


def split_by_context(datasets, contexts_for_pretraining):
    training_set, testing_set, input_shape, target_shape = datasets
    return split_ds_by_context(training_set, contexts_for_pretraining), split_ds_by_context(testing_set, contexts_for_pretraining), input_shape, target_shape


def split_ds_by_context(ds, contexts_for_pretraining):
    new_dss = []
    contexts = np.unique([x[1] for x, y in ds], axis=0)
    if len(contexts) == 0:
        return None

    pretrain_contexts = contexts[:contexts_for_pretraining]
    new_dss.append(ds.filter(lambda x, y: tf.reduce_any(tf.reduce_all(tf.equal(x[1], pretrain_contexts), axis=-1))))

    for c in contexts[contexts_for_pretraining:]:
        new_dss.append(ds.filter(lambda x, y: tf.reduce_all(tf.equal(x[1], c))))
    return new_dss