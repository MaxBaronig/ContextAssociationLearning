import logging
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Activation

from model.layers.apical_compartment import ApicalCompartment
from model.layers.basal_compartment import BasalCompartment
from tensorflow.keras.regularizers import l1_l2
import plotly.express as px

DEAD_NEURON_THRESHOLD = 1e-4


class Pyramidal(Layer):
    def __init__(self,
                num_neurons,
                basal_compartment_kwargs,
                apical_compartment_kwargs,
                alpha,
                random_seed,
                **kwargs):
        self.basal_compartment_kwargs = basal_compartment_kwargs
        self.apical_compartment_kwargs = apical_compartment_kwargs
        self.random_seed = random_seed
        self.num_neurons = num_neurons
        self.alpha = alpha
        super(Pyramidal, self).__init__(**kwargs)


    def build(self, input_shape):
        super(Pyramidal, self).build(input_shape)
        self.apical_compartment = ApicalCompartment(
            random_seed=self.random_seed,
            **self.apical_compartment_kwargs
            )
        self.basal_compartment = BasalCompartment(
            random_seed=self.random_seed, 
            output_shape=self.num_neurons,
            **self.basal_compartment_kwargs
            )
        self.alpha = self.add_weight(
            name="alpha", 
            shape=(1,),  
            initializer=tf.keras.initializers.constant(self.alpha), 
            trainable=False
            )
            
    # inputs: (basal_input, apical_input)
    def call(self, inputs, training=False, apical_training=False, *args, **kwargs):
        basal_input, apical_input = inputs

        # get basal activations
        u_b = self.basal_compartment(basal_input)
        
        # get ca spike
        ca_spike = self.apical_compartment([apical_input, u_b])

        # get output rate
        somatic_output = u_b + self.alpha * ca_spike

        # train apical weights
        if apical_training:
            u_BP = u_b   
            self.apical_compartment(
                inputs=[apical_input, u_BP],
                training=False, 
                apical_training=True
                )

        return somatic_output
