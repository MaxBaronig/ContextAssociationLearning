cdfa_multitask_config = {
    "model": {
        "num_pyramidal_neurons": 60,
            "pattern_properties": {
                "pattern_length": 60,
                "active_neurons": 9,
                "max_overlap": 0.4
        },
        "pyramidal_layer_kwargs": {
            "basal_compartment_kwargs": {
                "learning_rule": "krotov",
                "lr": 0.1,
                "K": 6,
            },
            "apical_compartment_kwargs": {
                "num_branches": 10,  
                "weight_init_mean" : 0.016, 
                "weight_init_stddev":  0.03, 
                "weight_init_sparsity": 0.0, 
                "learning_rule_params": {
                    "wmax": 1.0/9.0,
                    "lambda_": 0.33,
                    "kappa": 0.3,
                    "lambda_reg": 18,
                    "nca": 1,
                    "probabilistic": True,
                },
                "apical_weight_learning_rate": 0.08, 
                "apical_learning_algorithm": "CAL", 
            },
            "alpha": 0.5
        }
    },
    "training_method": "multitask",
    "training_parameters": {
        "batch_size": 64,
        "epochs": 100,
        "learning_rate": 0.1,
    },
    "unsup_basal_pretraining_parameters": {
        "batch_size": 16,
        "epochs": 80,
        "lr0": 0.1,
    },
}