"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import numpy as np
import torch as th
from models.encoders import MultimodalEncoder
from models.context_model import ContextModel
from training.dataset import DataReader
from training.forwarder import CategoricalAutoregressiveForwarder
from training.trainer import Trainer


config = {
    "artifacts_dir": "artifacts_dir",
    "vertex_mean": "assets/face_mean.npy",
    "vertex_std": "assets/face_std.npy",
    "dataloader_workers": 4,
    "expression_space": {
        "classes": 128,
        "heads": 16,
    },
    "train": {
        "learning_rate": 0.0001,
        "batch_size": 8,
        "segment_length": 64,
        "max_iterations": 15_000,
        "decrease_lr_iters": [10_000],
        "save_frequency": 5_000,
        "loss_terms": ["cross_entropy"],
        "loss_weights": {"cross_entropy": 1.0},
    },
}

os.system(f"mkdir -p {config['artifacts_dir']}")

# load mean and stddev
mean = th.from_numpy(np.load(config["vertex_mean"]))
stddev = th.from_numpy(np.load(config["vertex_std"]))

# define train and validation dataset
train_dataset = DataReader(segment_length=64)
val_dataset = DataReader(segment_length=64)

encoder = MultimodalEncoder(classes=config["expression_space"]["classes"],
                            heads=config["expression_space"]["heads"],
                            expression_dim=128,
                            audio_dim=128,
                            n_vertices=train_dataset.n_vertices,
                            mean=mean,
                            stddev=stddev
                            )
encoder.load(config["artifacts_dir"])
context_model = ContextModel(classes=config["expression_space"]["classes"],
                             heads=config["expression_space"]["heads"],
                             audio_dim=128
                             )
print(f"context_model: {context_model.num_trainable_parameters()} trainable parameters")

# train
forwarder = CategoricalAutoregressiveForwarder(config, encoder, context_model)
trainer = Trainer(config, forwarder, train_dataset, val_dataset)
trainer.train()
