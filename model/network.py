from re import L
import tensorflow as tf
from tensorflow.keras.activations import softmax, sigmoid
from tensorflow.keras.layers import Dense, Conv2D, Flatten, LayerNormalization, Softmax, Activation, Dropout
from tensorflow.keras.regularizers import l1_l2, l1, l2
from model.layers.context_embedding import ContextEmbedding
from tensorflow.keras.losses import KLD, CategoricalCrossentropy
from model.layers.pyramidal import Pyramidal
import logging

class Network(tf.keras.Model):
    def __init__(self,
                 random_seed,
                 num_pyramidal_neurons,
                 pattern_properties,
                 pyramidal_layer_kwargs,
                 **kwargs) -> None:
        super(Network, self).__init__(**kwargs)
        self.random_seed = random_seed
        self.pattern_properties = pattern_properties
        self.pyramidal_layer_kwargs = pyramidal_layer_kwargs
        self.num_pyramidal_neurons = num_pyramidal_neurons


    def build(self, input_shape):      
        self.flatten = Flatten()
        self.embedding_layer = ContextEmbedding(
            pattern_cache_root="data/patterns",
            random_seed=self.random_seed,
            **self.pattern_properties
            )
        self.pyramidal_layer = Pyramidal(
            num_neurons=self.num_pyramidal_neurons, 
            random_seed=self.random_seed, 
            **self.pyramidal_layer_kwargs
            )
        self.out_threshold = self.add_weight(
            "threshold", 
            initializer=tf.keras.initializers.constant(-20), 
            trainable=True)
        

    # inputs: (basal_input, apical_input)
    def call(self, inputs, training=False, **kwargs):
        # basal_input shape: (batch_size, basal_input_dimension)
        # apical_input shape: (batch_size, apical_input_dimension)
        basal_input, apical_input = inputs

        # apical input is one-hot. Embedding it into a context vector.
        apical_dense = self.embedding_layer(apical_input)
        #  = tf.expand_dims(self.emb(inputs[1]), axis=-2)
        pyramidal_output = self.pyramidal_layer([basal_input, apical_dense], **kwargs)

        network_output = sigmoid(tf.reduce_sum(pyramidal_output, axis=-1, keepdims=True) + self.out_threshold)
        return network_output


    @tf.function
    def train_step(self, data):
        x, y = data

        # self.out_threshold is trained by gradient descent
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True, apical_training=False)  # Forward pass
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, y_pred)

        # apply apical training
        self(x, training=False, apical_training=True)

        return {m.name: m.result() for m in self.metrics}
