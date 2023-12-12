import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Softmax, Flatten
import tensorflow_probability as tfp
from tensorflow_probability import stats
import plotly.express as px
import numpy as np


def K_WTA(x, K):
    vals, indices = tf.math.top_k(x, k=K)
    vals_red = tf.reduce_min(vals, axis=-1, keepdims=True)
    y = tf.where(x >= vals_red, 1.0, 0.0)
    return y


class BasalCompartment(tf.keras.layers.Layer):
    def __init__(self, 
                 learning_rule, 
                 lr, 
                 K,
                 output_shape, 
                 random_seed,
                 **kwargs):
        super(BasalCompartment, self).__init__(trainable=True)
        self.learning_rule = learning_rule
        self.random_seed = random_seed
        self.out_shape = output_shape
        self.lr = lr
        self.K = K


    def build(self, input_shape):
        # This is the basal weight matrix, trained by Krotov or Krotov+
        self.flat = Flatten()
        self.kernel = self.add_weight(
            "V",
            (input_shape[-1], self.out_shape),
            regularizer=None,
            initializer=tf.keras.initializers.random_normal(0.0, 1.0, self.random_seed),
            trainable=False
        )
        return super().build(input_shape)


    @tf.function
    def call(self, inputs, training=None, *args, **kwargs):
        inputs = self.flat(inputs)
        u_b = tf.tensordot(inputs, self.kernel, axes=[-1, 0])    

        # apply K-WTA:
        outputs = K_WTA(u_b, self.K)
        return outputs
    

    @tf.function
    def update(self, input, outputs, pretraining):
        input = self.flat(input)
        lr = self.lr
        u_b = tf.tensordot(input, self.kernel, axes=[-1, 0])    
        q = K_WTA(u_b, self.K)

        
        if self.learning_rule == "krotov":
            first_term = tf.transpose(input) @ q
            second_term = tf.reduce_sum(tf.expand_dims(q, axis=-2) * tf.expand_dims(u_b, axis=-2), 0) * self.kernel

        elif self.learning_rule == "krotov_plus":
            first_term = tf.transpose(input) @ (q) 
            second_term = tf.expand_dims(tf.maximum(q, 0), axis=-2) * self.kernel
            second_term = tf.transpose(tf.transpose(q) @ tf.reduce_sum(second_term, axis=-1))


        dw =  first_term -  second_term
        normalized_dw = dw / tf.maximum(tf.reduce_max(tf.abs(dw)), 1e-30)

        if pretraining:  
            self.kernel.assign(self.kernel +  lr * normalized_dw)
        
