"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch as th
import torch.nn.functional as F

from utils.helpers import Net


class VertexUnet(Net):
    def __init__(self, classes: int = 128, heads: int = 64, n_vertices: int = 6172, mean: th.Tensor = None,
                 stddev: th.Tensor = None, model_name: str = 'vertex_unet'):
        """
        VertexUnet consumes a neutral template mesh and an expression encoding and produces an animated face mesh
        :param classes: number of classes for the categorical latent embedding
        :param heads: number of heads for the categorical latent embedding
        :param n_vertices: number of vertices in the face mesh
        :param mean: mean position of each vertex
        :param stddev: standard deviation of each vertex position
        :param model_name: name of the model, used to load and save the model
        """
        super().__init__(model_name)

        self.classes = classes
        self.heads = heads
        self.n_vertices = n_vertices

        shape = (1, 1, n_vertices, 3)
        self.register_buffer("mean", th.zeros(shape) if mean is None else mean.view(shape))
        self.register_buffer("stddev", th.ones(shape) if stddev is None else stddev.view(shape))

        # encoder layers
        self.encoder = th.nn.ModuleList([
            th.nn.Linear(n_vertices*3, 512),
            th.nn.Linear(512, 256),
            th.nn.Linear(256, 128)
        ])

        # multimodal fusion
        self.fusion = th.nn.Linear(heads * classes + 128, 128)

        # decoder layers
        self.temporal = th.nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True)
        self.decoder = th.nn.ModuleList([
            th.nn.Linear(128, 256),
            th.nn.Linear(256, 512),
            th.nn.Linear(512, n_vertices*3)
        ])
        self.vertex_bias = th.nn.Parameter(th.zeros(n_vertices * 3))

    def _encode(self, x: th.Tensor):
        """
        encode the neutral face mesh through the UNet
        :param x: B x T x n_vertices * 3 Tensor containing vertices of neutral face mesh
        :return: x: B x T x 128 Tensor containing a 128-dim encoding of each input mesh
                 skips: outputs after each of the UNet encoder layers
        """
        skips = []
        for i, layer in enumerate(self.encoder):
            skips = [x] + skips
            x = F.leaky_relu(layer(x), 0.2)
        return x, skips

    def _fuse(self, geom_encoding: th.Tensor, expression_encoding: th.Tensor):
        """
        :param geom_encoding: B x T x 128 Tensor containing the encoding of the neutral face meshes
        :param expression_encoding: B x T x heads x classes Tensor containing the one hot expression encodings
        :return: B x T x 128 Tensor containing a latent representation of the animated face
        """
        expression_encoding = expression_encoding.view(
            expression_encoding.shape[0], expression_encoding.shape[1], self.heads * self.classes
        )
        x = self.fusion(th.cat([geom_encoding, expression_encoding], dim=-1))
        x = F.leaky_relu(x, 0.2)
        return x

    def _decode(self, x: th.Tensor, skips: th.Tensor):
        """
        :param x: B x T x 128 Tensor containing a latent representation of the animated face
        :param skips: outputs of each UNet encoder layer to be added back to the decoder layers
        :return: B x T x n_vertices * 3 Tensor containing the animated face meshes
        """
        x, _ = self.temporal(x)
        for i, layer in enumerate(self.decoder):
            x = skips[i] + F.leaky_relu(layer(x), 0.2)
        x = x + self.vertex_bias.view(1, 1, -1)
        return x

    def forward(self, geom: th.Tensor, expression_encoding: th.Tensor):
        """
        :param geom: B x T x n_vertices x 3 Tensor containing template face meshes
        :param expression_encoding: B x T x heads x classes Tensor containing one hot expression encodings
        :return: geom: B x T x n_vertices x 3 Tensor containing predicted face meshes
        """
        x = (geom - self.mean) / self.stddev
        x = x.view(x.shape[0], x.shape[1], self.n_vertices*3)

        geom_encoding, skips = self._encode(x)
        x = self._fuse(geom_encoding, expression_encoding)
        x = self._decode(x, skips)

        x = x.view(x.shape[0], x.shape[1], self.n_vertices, 3)
        geom = x * self.stddev + self.mean
        return {"geom": geom}
