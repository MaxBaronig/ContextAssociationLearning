import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import zeros, constant
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import json
import hashlib
import os

def jaccard_binary(x,y):
    """A function for finding the similarity between two binary vectors"""
    intersection = np.logical_and(x, y)
    union = np.logical_or(x, y)
    similarity = intersection.sum(axis=-1) / union.sum(axis=-1).astype(float)
    return similarity


class ContextEmbedding(Layer):
    def __init__(self, 
                 pattern_properties,
                 random_seed, 
                 pattern_cache_root = None,
                 **kwargs):
        self.pattern_properties = pattern_properties
        self.pattern_cache_root = pattern_cache_root
        self.random_seed = random_seed
        super(ContextEmbedding, self).__init__(**kwargs)
    

    def build(self, input_shape):
        # generate the context patterns
        num_patterns = input_shape[1]
        patterns = self.generate_patterns(
            random_seed=self.random_seed,
            num_patterns=num_patterns,
            **self.pattern_properties
            )  
        
        # initialize the embedding matrix 
        emb_matrix_values = tf.cast(tf.stack(patterns), dtype=tf.float32)
        self.embedding_matrix = self.add_weight(
            name="embedding_matrix", 
            shape=emb_matrix_values.shape, 
            trainable=False, 
            dtype=tf.float32
            )
        self.embedding_matrix.assign(emb_matrix_values)


    def call(self, input, training=None):
        return tf.clip_by_value(tf.matmul(input, self.embedding_matrix), 0.0, 1.0)


    """Generate patterns or retrieve from pattern cache for the context embedding"""
    def generate_patterns(self, num_patterns, pattern_length, active_neurons, max_overlap, random_seed):
        cache_key = ContextEmbedding.compute_key(
            dict(num_patterns=num_patterns, 
                 pattern_length=pattern_length, 
                 active_neurons=active_neurons, 
                 random_seed=random_seed)
                 )

        if os.path.isfile(os.path.join(self.pattern_cache_root, cache_key + ".npy")):
            patterns = np.load(
                os.path.join(self.pattern_cache_root, cache_key + ".npy"), 
                allow_pickle=True
                )
            print("pattern cache hit")
        else:
            print("pattern cache miss")
            mother_pattern = np.zeros(pattern_length)
            mother_pattern[:active_neurons] = 1.0

            rng : np.random.Generator = np.random.default_rng(self.random_seed)

            patterns = []

            while len(patterns) < num_patterns:
                perm = rng.permutation(mother_pattern)
                if patterns and np.max(cosine_similarity(perm.reshape(1, -1), patterns)) > max_overlap:
                    continue
                else:                    
                    patterns.append(perm)
                
            np.save(os.path.join(self.pattern_cache_root, cache_key + ".npy"), np.array(patterns), allow_pickle=True)

        # report the pairwise overlap
        pairwise_patterns = cosine_similarity(patterns, patterns)
        np.fill_diagonal(pairwise_patterns, 0.0)
        max_pattern_similarity = np.max(pairwise_patterns)
        print("Context Embedding: maximum pairwise cosine similarity: {}".format(max_pattern_similarity))
        jac_pairwise = np.array([jaccard_binary(p.reshape(1, -1), patterns) for p in patterns])
        np.fill_diagonal(jac_pairwise, 0.0)
        max_pattern_similarity = np.max(jac_pairwise)
        print("Context Embedding: maximum pairwise jaccard similarity: {}".format(max_pattern_similarity))
        return patterns

    @staticmethod
    def compute_key(c):
        s = json.dumps(c, sort_keys=True)
        return hashlib.sha224(s.encode('utf-8')).hexdigest()