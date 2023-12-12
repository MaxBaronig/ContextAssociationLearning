import tensorflow as tf
from utils import load_data
import os
from model.network import Network
from training.unsupervised_basal_pretraining import unsupervised_basal_weight_pretraining

""" This code runs the CDFA experiment for both the multi-task (Section 2.5) and the continual learning setting (Section 2.6)."""
def run_cdfa_multitask(experiment, data_base_path, random_seed, config):  
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

    # Construct Dataset
    training_set, testing_set, input_shape, output_shape, _ = load_data(
        training_dataset_path
    )
    
    # Tile into batches
    batch_size = config.training_parameters.batch_size
    batched_train = training_set.batch(batch_size).cache().prefetch(100 * batch_size)
    batched_test = testing_set.batch(batch_size).cache().prefetch(100 * batch_size)

    # run training of apical weights AND readout threshold
    model.fit(
        x=batched_train, 
        epochs=config.training_parameters.epochs, 
        verbose=2, 
        use_multiprocessing=True
    )
    
    # evaluate model
    test_result = model.evaluate(
        x=batched_test, 
        verbose=2, 
        return_dict=True
    )
    return test_result