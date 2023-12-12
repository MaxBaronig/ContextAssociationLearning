from model.layers.basal_compartment import BasalCompartment
import tensorflow as tf
from task.utils import load_data
import numpy as np

def unsupervised_basal_weight_pretraining(
        model, 
        epochs, 
        lr0,
        pretrain_set):
    
    print("Starting unsupervised pretraining of basal weights")
    basal_compartment: BasalCompartment = model.pyramidal_layer.basal_compartment
    for epoch in range(epochs):
        print(f"Pretraining Epoch {epoch}/{epochs}")
        lr_ = lr0 * (1 - epoch/epochs)
        basal_compartment.lr = lr_
        for b in pretrain_set:
            outputs = basal_compartment.call(b[0][0])
            basal_compartment.update(b[0][0], outputs, pretraining=True)
