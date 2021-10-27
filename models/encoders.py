"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch as th
import torch.nn.functional as F
import torchaudio as ta

from utils.helpers import Net


class AudioEncoder(Net):
    def __init__(self, latent_dim: int = 128, model_name: str = 'audio_encoder'):
        """
        :param latent_dim: size of the latent audio embedding
        :param model_name: name of the model, used to load and save the model
        """
        super().__init__(model_name)

        self.melspec = ta.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=2048, win_length=800, hop_length=160, n_mels=80
        )

        conv_len = 5
        self.convert_dimensions = th.nn.Conv1d(80, 128, kernel_size=conv_len)
        self.weights_init(self.convert_dimensions)
        self.receptive_field = conv_len

        convs = []
        for i in range(6):
            dilation = 2 * (i % 3 + 1)
            self.receptive_field += (conv_len - 1) * dilation
            convs += [th.nn.Conv1d(128, 128, kernel_size=conv_len, dilation=dilation)]
            self.weights_init(convs[-1])
        self.convs = th.nn.ModuleList(convs)
        self.code = th.nn.Linear(128, latent_dim)

        self.apply(lambda x: self.weights_init(x))

    def weights_init(self, m):
        if isinstance(m, th.nn.Conv1d):
            th.nn.init.xavier_uniform_(m.weight)
            try:
                th.nn.init.constant_(m.bias, .01)
            except:
                pass

    def forward(self, audio: th.Tensor):
        """
        :param audio: B x T x 16000 Tensor containing 1 sec of audio centered around the current time frame
        :return: code: B x T x latent_dim Tensor containing a latent audio code/embedding
        """
        B, T = audio.shape[0], audio.shape[1]
        x = self.melspec(audio).squeeze(1)
        x = th.log(x.clamp(min=1e-10, max=None))
        if T == 1:
            x = x.unsqueeze(1)

        # Convert to the right dimensionality
        x = x.view(-1, x.shape[2], x.shape[3])
        x = F.leaky_relu(self.convert_dimensions(x), .2)

        # Process stacks
        for conv in self.convs:
            x_ = F.leaky_relu(conv(x), .2)
            if self.training:
                x_ = F.dropout(x_, .2)
            l = (x.shape[2] - x_.shape[2]) // 2
            x = (x[:, :, l:-l] + x_) / 2

        x = th.mean(x, dim=-1)
        x = x.view(B, T, x.shape[-1])
        x = self.code(x)

        return {"code": x}


class ExpressionEncoder(Net):
    def __init__(self, latent_dim: int = 128, n_vertices: int = 6172, mean: th.Tensor = None, stddev: th.Tensor = None,
                 model_name: str = 'expression_encoder'):
        """
        :param latent_dim: size of the latent expression embedding before quantization through Gumbel softmax
        :param n_vertices: number of face mesh vertices
        :param mean: mean position of each vertex
        :param stddev: standard deviation of each vertex position
        :param model_name: name of the model, used to load and save the model
        """
        super().__init__(model_name)

        self.n_vertices = n_vertices

        shape = (1, 1, n_vertices, 3)
        self.register_buffer("mean", th.zeros(shape) if mean is None else mean.view(shape))
        self.register_buffer("stddev", th.ones(shape) if stddev is None else stddev.view(shape))

        self.layers = th.nn.ModuleList([
            th.nn.Linear(self.n_vertices * 3, 256),
            th.nn.Linear(256, 128),
        ])
        self.lstm = th.nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.code = th.nn.Linear(128, latent_dim)

    def forward(self, geom: th.Tensor):
        """
        :param geom: B x T x n_vertices x 3 Tensor containing face geometries
        :return: code: B x T x heads x classes Tensor containing a latent expression code/embedding
        """
        x = (geom - self.mean) / self.stddev
        x = x.view(x.shape[0], x.shape[1], self.n_vertices*3)

        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.2)
        x, _ = self.lstm(x)
        x = self.code(x)

        return {"code": x}


class FusionMlp(Net):
    def __init__(self, classes: int = 128, heads: int = 64, expression_dim: int = 128, audio_dim: int = 128,
                 model_name: str = 'fusion_model'):
        """
        :param classes: number of classes for the categorical latent embedding
        :param heads: number of heads for the categorical latent embedding
        :param expression_dim: size of the latent expression embedding before quantization through Gumbel softmax
        :param audio_dim: size of the latent audio embedding
        :param model_name: name of the model, used to load and save the model
        """
        super().__init__(model_name)
        self.classes = classes
        self.heads = heads

        latent_dim = 256
        self.mlp = th.nn.Sequential(
            th.nn.Linear(expression_dim + audio_dim, latent_dim),
            th.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            th.nn.Linear(latent_dim, latent_dim),
            th.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            th.nn.Linear(latent_dim, heads * classes)
        )

    def forward(self, expression_code: th.Tensor, audio_code: th.Tensor):
        """
        :param expression_code: B x T x expression_dim Tensor containing the expression encodings
        :param audio_code: B x T x audio_dim Tensor containing the audio encodings
        :return: logprobs: B x T x heads x classes Tensor containing logprobs for each categorical head
        """
        x = th.cat([expression_code, audio_code], dim=-1)
        x = self.mlp(x).view(x.shape[0], x.shape[1], self.heads, self.classes)
        logprobs = F.log_softmax(x, dim=-1)
        return {"logprobs": logprobs}


class MultimodalEncoder(Net):
    def __init__(self,
                 classes: int = 128,
                 heads: int = 64,
                 expression_dim: int = 128,
                 audio_dim: int = 128,
                 n_vertices: int = 6172,
                 mean: th.Tensor = None,
                 stddev: th.Tensor = None,
                 model_name: str = "encoder"
                 ):
        """
        :param classes: number of classes for the categorical latent embedding
        :param heads: number of heads for the categorical latent embedding
        :param expression_dim: size of the latent expression embedding before quantization through Gumbel softmax
        :param audio_dim: size of the latent audio embedding
        :param n_vertices: number of vertices in the face mesh
        :param mean: mean position of each vertex
        :param stddev: standard deviation of each vertex position
        :param model_name: name of the model, used to load and save the model
        """
        super().__init__(model_name)
        self.audio_encoder = AudioEncoder(audio_dim)
        self.expression_encoder = ExpressionEncoder(expression_dim, n_vertices, mean, stddev)
        self.fusion_model = FusionMlp(classes, heads, expression_dim, audio_dim)

    def fuse(self, expression_code: th.Tensor, audio_code: th.Tensor):
        """
        :param expression_code: B x T x expression_dim Tensor containing the expression encodings
        :param audio_code: B x T x audio_dim Tensor containing the audio encodings
        :return: logprobs: B x T x heads x classes Tensor containing logprobs for each categorical head
        """
        logprobs = self.fusion_model(expression_code, audio_code)["logprobs"]
        return {"logprobs": logprobs}

    def encode(self, geom: th.Tensor, audio: th.Tensor):
        """
        :param geom: B x T x n_vertices x 3 Tensor containing face geometries (expressions)
        :param audio: B x T x 1 x 16000 Tensor containing one second of 16kHz audio centered around each frame 1,...,T
        :return: expression_code: B x T x expression_dim Tensor containing expression encoding
                 audio_code: B x T x audio_dim Tensor containing audio encoding
        """
        expression_code = self.expression_encoder(geom)["code"]
        audio_code = self.audio_encoder(audio)["code"]
        return {"expression_code": expression_code, "audio_code": audio_code}

    def forward(self, geom: th.Tensor, audio: th.Tensor):
        """
        :param geom: B x T x n_vertices x 3 Tensor containing face geometries (expressions)
        :param audio: B x T x 1 x 16000 Tensor containing one second of 16kHz audio centered around each frame 1,...,T
        :return: logprobs: B x T x heads x classes Tensor containing logprobs for each categorical head
                 expression_code: B x T x expression_dim Tensor containing expression encoding
                 audio_code: B x T x audio_dim Tensor containing audio encoding
        """
        codes = self.encode(geom, audio)
        logprobs = self.fuse(codes["expression_code"], codes["audio_code"])["logprobs"]

        return {"logprobs": logprobs, "expression_code": codes["expression_code"], "audio_code": codes["audio_code"]}
