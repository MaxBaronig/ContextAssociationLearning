import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow_probability as tfp
import numpy as np
from tensorflow.keras.initializers import random_normal


def thresholded_step(x, th):       
    return tf.cast(tf.math.greater_equal(x, th), tf.float32)


class ApicalCompartment(Layer):
    def __init__(self, 
                 random_seed,
                 num_branches,  
                 weight_init_mean, 
                 weight_init_stddev, 
                 weight_init_sparsity, 
                 learning_rule_params, 
                 apical_weight_learning_rate, 
                 apical_learning_algorithm, 
                 **kwargs):      
        # self.dend_spike_thresholds = np.full(num_branches, dend_spike_threshold)
        self.random_seed = random_seed
        self.num_branches = num_branches
        self.weight_init_mean = weight_init_mean 
        self.weight_init_stddev = weight_init_stddev
        self.weight_init_sparsity = weight_init_sparsity
        self.learning_rule_params = learning_rule_params
        self.apical_weight_learning_rate = apical_weight_learning_rate
        self.apical_learning_algorithm : ApicalLearningAlgorithm = None
        if apical_learning_algorithm == "CAL":
            self.apical_learning_algorithm = ContextAssociationLearning
        elif apical_learning_algorithm == "DLR":
            self.apical_learning_algorithm = DendriticLogisticRegression
        else:
            raise NotImplemented()
        super(ApicalCompartment, self).__init__(**kwargs)


    def build(self, input_shape):
        # basal_input_shape: batch_size x num_neurons x 1
        # apical_input_shape: batch_size x num_neurons x synapses
        basal_input_shape, apical_input_shape = input_shape

        # automatically infer number of neurons and synapses
        self.num_neurons = tf.cast(tf.reduce_prod(basal_input_shape[1]), tf.int32)
        num_synapses = apical_input_shape[-1]

        # shape of W: num_neurons x branches x synapses
        self.W = self.add_weight(name="apical_weight", 
                                 shape=[self.num_neurons, self.num_branches, num_synapses], 
                                 trainable=False, 
                                 initializer=random_normal(
                                    self.weight_init_mean,
                                    self.weight_init_stddev,
                                    seed=self.random_seed
                                    )
                                )
        # sparsity (using numpy for simplicity)
        dropout_mask = np.ones(self.W.shape)

        # we want to set the same number of synapses 0 for each branch
        dropout_mask[:,:,:int(num_synapses*self.weight_init_sparsity)] = 0
        dropout_mask = tf.reshape(dropout_mask, [-1, num_synapses])

        # shuffle per branch
        dropout_mask = tf.scan(lambda x, y: tf.random.shuffle(x), dropout_mask)
        W_new = self.W * tf.cast(tf.reshape(dropout_mask, self.W.shape), dtype=tf.float32)
        self.W.assign(self.apical_learning_algorithm.apply_weight_constraints(W_new, self.p))


    def call(self, input, training=None, apical_training=None):
        # apical input shape: batch_size * 1 * num_synapses
        # u_BP shape: batch_size * num_neurons
        apical_input, u_BP = input

        # convert apical input to batch_size * 1 * 1 * num_synapses
        apical_input = tf.expand_dims(apical_input, axis=-2)

        # convert u_BP to batch_size * num_neurons * 1 * 1
        u_BP = u_BP[:,:,tf.newaxis, tf.newaxis]
    
        # response shape: batch_size * num_neurons * 1 * 1
        response = self.apical_learning_algorithm.call(
            self.W,
            apical_input,
            u_BP,
            self.learning_rule_params,
            training
        )

        if apical_training and self.dendritic_learning_rate > 0.0:
            # compute weight update of learning algorithm
            dw = self.apical_learning_algorithm.train_step(
                self.W,
                apical_input,
                u_BP,
                self.learning_rule_params
            )
            
            # reduce over batch dimension
            batch_update = tf.reduce_mean(dw, axis=0)

            # apply weight update
            W_new = self.W + self.dendritic_learning_rate * batch_update
            W_new = self.apical_learning_algorithm.apply_weight_constraints(W_new, self.p)
            self.W.assign(W_new)
        return tf.squeeze(response, axis=[-1,-2]) # convert response to shape batch_size * num_neurons
    

"""abstract base class for apical learning algorithms"""      
class ApicalLearningAlgorithm():
    def train_step(W, input, bap, p):
        raise NotImplemented()
    
    def call_dend(W, input, p):
        raise NotImplemented()

    def call(W, input, bap, p, training):
        raise NotImplemented()

    def apply_weight_constraints(W, p):
        raise NotImplemented()
    

"""Dendritic Logistic Regression from 
    Rao et al. 2022 
    Title: "Self-supervised learning of probabilistic prediction through synaptic plasticity in apical dendrites: A normative model"
    DOI: https://doi.org/10.1101/2021.03.04.433822 
    bioRxiv preprint"""
class DendriticLogisticRegression(ApicalLearningAlgorithm):
    @staticmethod
    def call(W, input, bap, p, training=None):
        output_per_branch = DendriticLogisticRegression.call_dend(W, input, p)
        # ca_spike = bap * tf.reduce_max(output_per_branch, axis=-2, keepdims=True)
        ca_spike = tf.sigmoid(p["alpha"] * (tf.reduce_sum(output_per_branch, axis=-2, keepdims=True) - p["theta"]))
        return ca_spike

    @staticmethod
    def call_dend(W, input, p):
        u = tf.reduce_sum(input * W, -1, keepdims=True) - p["u0"]
        sig = tf.sigmoid(p["beta"] * u)
        dend_spikes = sig
        return dend_spikes

    @staticmethod
    def train_step(W, input, bap, p):
        psi =  DendriticLogisticRegression.call(W, input, bap, p)
        f_dot = DendriticLogisticRegression.f_dot(W, input, p)
        delta_w  = (bap - psi) * f_dot * input
        return delta_w
        
    
    @staticmethod
    def f_dot(W, input, p):
        u = tf.reduce_sum(input * W, -1, keepdims=True) - p["u0"]
        sig = tf.sigmoid(p["beta"] * u)
        return sig * (1 - sig)

    @staticmethod
    def apply_weight_constraints(W, p):
        return W


"""Context Association Learning from Baronig & Legenstein"""
class ContextAssociationLearning(ApicalLearningAlgorithm):
    @staticmethod
    def call(W, input, bap, p, training=None):
        if "nca" in p:
            nca = p["nca"]
        else:
            nca = 1.0
        output_per_branch = ContextAssociationLearning.call_dend(W, input, p, training)
        if (p["probabilistic"] and training) or nca != 1.0:
            ca_spike = tf.maximum(tf.reduce_sum(output_per_branch, axis=-2, keepdims=True) - nca + 1, 0.0)
        else:
            prod_mat = (1 - output_per_branch) 
            ca_spike = 1-tf.reduce_prod(prod_mat, axis=-2, keepdims=True)
        return tf.clip_by_value(ca_spike, 0.0, 1.0)


    @staticmethod
    def call_dend(W, input, p, training=None):
        u = tf.reduce_sum(input * W, -1, keepdims=True)
        sig = ContextAssociationLearning.sigma(u)
        if (p["probabilistic"] and training) or "nca" in p:
            dend_spikes = tf.cast(tfp.distributions.Bernoulli(probs=sig).sample(), dtype=tf.float32)
        else:
            dend_spikes = sig
        return dend_spikes


    @staticmethod
    def train_step(W, input, bap, p):
        if "nca" in p:
            nca = p["nca"]
        else:
            nca = 1.0
        epsilon = 0.08
        U = tf.reduce_sum(input * W, -1, keepdims=True)
        p_dot = ContextAssociationLearning.sigma_dot(U)
        sig_probs = tf.clip_by_value(ContextAssociationLearning.sigma(U), 0.0, 1.0)

        if p["probabilistic"] or nca != 1.0:
            sig = tf.cast(tfp.distributions.Bernoulli(probs=sig_probs).sample(), dtype=tf.float32)
            increase_term = -1 * tf.cast(tf.less(tf.reduce_sum(sig, axis=-2, keepdims=True), nca), dtype=tf.float32)
        else:
            sig = sig_probs
            prod_mat = (1 - sig) 
            prod = tf.reduce_prod(prod_mat, axis=-2, keepdims=True)
            increase_term = prod * -1

        reg_term = p["lambda_reg"] * sig *  W * (tf.reduce_sum(W, axis=-1, keepdims=True)-input)

        # gradient of loss
        dE_dW = bap * ((p_dot + epsilon) * input * ( increase_term ) \
                        + input * p_dot * p["lambda_"] * (1 - 2 *  sig) \
                        + reg_term) \
                + p["kappa"] * (1 - bap) * p_dot * input 
        
        # weight dependent learning rate (see Methods)
        eta =  p["wmax"] * ((tf.maximum(W,0)**2 * tf.minimum(W-p["wmax"], 0) ** 2) / ((p["wmax"]/2) ** 4)  + 1.0/40.0 )
        return -1 * eta * dE_dW
    

    @staticmethod
    def apply_weight_constraints(W, p):
        return tf.clip_by_value(W, 0.0, p["wmax"])


    """parameters for generalized sigmoid function"""
    @staticmethod
    def get_sigmoid_params():
        D = 0.7
        B = 20
        C = 1
        d1 = C + tf.exp(-1 * B * (1 - D))
        d2 = 1 - C - tf.exp(B * D)
        A = d1/(d1 + d2 - 1)
        K = A * d2
        return K, D, B, C


    """generalized sigmoid function"""
    @staticmethod
    def sigma(u):
        K, D, B, C = ContextAssociationLearning.get_sigmoid_params()
        d1 = C + tf.exp(-1 * B * (1 - D))
        d2 = 1 - C - tf.exp(B * D)
        A = d1/(d1 + d2 - 1)
        y = A + (K - A) / (C + tf.exp(-1 * B * (u - D)))    
        return tf.clip_by_value(y, 0.0, 1.0)

    """derivative of generalized sigmoid function"""
    @staticmethod
    def sigma_dot(u):
        K, D, B, C = ContextAssociationLearning.get_sigmoid_params()
        d1 = C + tf.exp(-1 * B * (1 - D))
        d2 = 1 - C - tf.exp(B * D)
        A = d1/(d1 + d2 - 1)
        res = (B * (K - A) * tf.exp(-B * (u - D))) / (tf.exp(-B * (u - D)) + C) ** 2
        return res