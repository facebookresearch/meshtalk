"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch as th
import torch.nn.functional as F

from utils.helpers import Net

class MaskedContextConvolution(th.nn.Module):
    def __init__(self, ch_in: int, ch_out: int, heads: int, audio_dim: int, kernel_size: int = 1, dilation: int = 1):
        """
        :param ch_in: number of input channels to the layer
        :param ch_out: number of output channels to the layer
        :param heads: number of heads
        :param audio_dim: size of the latent audio embedding
        :param kernel_size: kernel size of the convolution
        :param dilation: dilation used in the convolution
        """
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.heads = heads
        self.audio_dim = audio_dim
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.unmasked_linear = th.nn.Conv1d(audio_dim, ch_out * heads, kernel_size=1)

        self.masked_linear = th.nn.Conv1d(ch_in * heads, ch_out * heads, kernel_size=1)
        mask = th.ones(ch_out * heads, ch_in * heads, 1)
        for i in range(heads):
            mask[ch_out * i:ch_out * (i+1), ch_in * i:, :] = 0
        self.register_buffer("mask", mask)

        if kernel_size > 0:
            self.historic = th.nn.Conv1d(ch_in * heads, ch_out * heads, kernel_size=kernel_size, dilation=dilation)

        self.reset()

    def receptive_field(self):
        """
        :return: receptive field of the layer
        """
        if self.kernel_size == 0:
            return 1
        else:
            return self.dilation * (self.kernel_size - 1) + 2

    def reset(self):
        """
        reset buffer before animating a new sequence
        """
        self.buffer = th.zeros(1, self.heads * self.ch_out, 0)
        self.historic_t = -1

    def forward_inference(self, t: int, h: int, context: th.Tensor, audio: th.Tensor = None):
        """
        :param t: current time step
        :param h: current head
        :param context: B x T x heads x ch_in Tensor
        :param audio: B x T x audio_dim Tensor
        :return: B x T x heads x ch_out Tensor
        """
        B, T = context.shape[0], context.shape[1]
        context = context.view(B, T, -1).permute(0, 2, 1).contiguous()

        if self.historic_t < t:
            self.buffer = self.buffer.to(context.device)
            self.buffer = th.cat([self.buffer, th.zeros(1, self.buffer.shape[1], 1, device=self.buffer.device)], dim=-1)

        # next head from previous head predictions
        y_masked = self.masked_linear.bias[h*self.ch_out:(h+1)*self.ch_out].view(1, -1, 1).clone()
        if h > 0:
            y_masked += F.conv1d(context[:, :h*self.ch_in, -1:],
                                 self.masked_linear.weight[h*self.ch_out:(h+1)*self.ch_out, :h*self.ch_in, :])
        self.buffer[:, h*self.ch_out:(h+1)*self.ch_out, -1:] += y_masked

        # next head from audio
        if audio is not None:
            audio = audio[:, -1:, :]
            audio = audio.permute(0, 2, 1).contiguous()
            y_audio = F.conv1d(audio[:, :, -1:],
                               self.unmasked_linear.weight[h*self.ch_out:(h+1)*self.ch_out, :, :],
                               bias=self.unmasked_linear.bias[h*self.ch_out:(h+1)*self.ch_out])
            self.buffer[:, h*self.ch_out:(h+1)*self.ch_out, -1:] += y_audio

        # historic time steps
        if self.kernel_size > 0 and self.historic_t < t:
            h = context[:, :, -self.receptive_field():-1]
            if h.shape[-1] < self.receptive_field() - 1:
                h = F.pad(h, pad=[self.receptive_field() - 1 - h.shape[-1], 0])
            h = self.historic(h)
            self.buffer[:, :, -1:] += h

        self.historic_t = t

        return self.buffer.permute(0, 2, 1).contiguous().view(1, -1, self.heads, self.ch_out)


    def forward(self, context: th.Tensor, audio: th.Tensor = None):
        """
        :param context: B x T x heads x ch_in Tensor
        :param audio: B x T x audio_dim Tensor
        :return: B x T x heads x ch_out Tensor
        """
        B, T = context.shape[0], context.shape[1]
        context = context.view(B, T, -1).permute(0, 2, 1).contiguous()

        # current context time step: masked along head axis
        y = F.conv1d(context, self.masked_linear.weight * self.mask, bias=self.masked_linear.bias)

        # current audio time step: no masking
        if audio is not None:
            audio = audio.permute(0, 2, 1).contiguous()
            audio = self.unmasked_linear(audio)
            y = y + audio

        # historic time steps
        if self.kernel_size > 0:
            h = F.pad(context[:, :, :-1], [self.dilation * (self.kernel_size - 1) + 1, 0])
            y = y + self.historic(h)

        y = y.permute(0, 2, 1).contiguous().view(B, T, self.heads, self.ch_out)

        return y


class ContextModel(Net):
    def __init__(self, classes: int = 128, heads: int = 64, audio_dim: int = 128, model_name: str = "context_model"):
        """
        :param classes: number of classes for the categorical latent embedding
        :param heads: number of heads for the categorical latent embedding
        :param audio_dim: size of the latent audio embedding
        :param model_name: name of the model, used to load and save the model
        """
        super().__init__(model_name)
        self.classes = classes
        self.heads = heads
        self.audio_dim = audio_dim

        hidden = 64
        self.embedding = MaskedContextConvolution(ch_in=classes, ch_out=hidden, heads=heads, audio_dim=audio_dim, kernel_size=0)
        self.context_layers = th.nn.ModuleList([
            MaskedContextConvolution(ch_in=hidden, ch_out=hidden, heads=heads, audio_dim=audio_dim, kernel_size=2, dilation=1),
            MaskedContextConvolution(ch_in=hidden, ch_out=hidden, heads=heads, audio_dim=audio_dim, kernel_size=2, dilation=2),
            MaskedContextConvolution(ch_in=hidden, ch_out=hidden, heads=heads, audio_dim=audio_dim, kernel_size=2, dilation=4),
            MaskedContextConvolution(ch_in=hidden, ch_out=hidden, heads=heads, audio_dim=audio_dim, kernel_size=2, dilation=8),
        ])
        self.logits = MaskedContextConvolution(ch_in=hidden, ch_out=classes, heads=heads, audio_dim=audio_dim, kernel_size=0)

    def receptive_field(self):
        """
        :return: receptive field of the model
        """
        receptive_field = 1
        for layer in self.context_layers:
            receptive_field += layer.receptive_field() - 1
        return receptive_field

    def _reset(self):
        """
        reset buffers in each layer before animating a new sequence
        """
        self.embedding.reset()
        for layer in self.context_layers:
            layer.reset()
        self.logits.reset()

    def _forward_inference(self, t: int, h: int, context: th.Tensor, audio: th.Tensor):
        """
        :param t: current time step
        :param h: current head
        :param context: B x T x heads x classes Tensor
        :param audio: B x T x audio_dim Tensor
        :return: logprobs: B x T x heads x classes Tensor containing log probabilities for each class
                 probs: B x T x heads x classes Tensor containing probabilities for each class
                labels: B x T x heads LongTensor containing discretized class labels
        """
        x = self.embedding.forward_inference(t, h, context)

        for layer in self.context_layers:
            x = layer.forward_inference(t, h, x, audio)
            x = F.leaky_relu(x, 0.2)

        logits = self.logits.forward_inference(t, h, x)
        logprobs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logprobs, dim=-1)
        labels = th.argmax(logprobs, dim=-1)

        return {"logprobs": logprobs, "probs": probs, "labels": labels}


    def sample(self, audio_code: th.Tensor, argmax: bool = False):
        """
        :param audio_code: B x T x audio_dim Tensor containing the encoded audio for the sequence
        :param argmax: if False, sample from Gumbel softmax; if True use classes with highest probabilities
        :return: B x T x heads x classes Tensor containing one-hot representation of latent code
        """
        assert audio_code.shape[0] == 1
        T = audio_code.shape[1]
        one_hot = th.zeros(1, T, self.heads, self.classes, device=audio_code.device)
        self._reset()
        for t in range(T):
            start, end = max(0, t - self.receptive_field()), t + 1
            context = one_hot[:, start:end, :, :]
            audio = audio_code[:, start:end, :]
            for h in range(self.heads):
                # select input for next logprobs
                logprobs = self._forward_inference(t, h, context, audio)["logprobs"][:, -1, h, :]
                # discretize
                if not argmax:
                    g = -th.log(-th.log(th.clamp(th.rand(logprobs.shape, device=logprobs.device), min=1e-10, max=1)))
                    logprobs = logprobs + g
                label_idx = th.argmax(logprobs, dim=-1).squeeze().item()
                one_hot[:, t, h, label_idx] = 1
        return {"one_hot": one_hot}

    def forward(self, expression_one_hot: th.Tensor, audio_code: th.Tensor):
        """
        :param expression_one_hot: B x T x heads x classes Tensor containing one hot representation of previous labels
               audio_code: B x T x audio_dim Tensor containing the audio embedding
        :return: logprobs: B x T x heads x C Tensor containing log probabilities for each class
                 probs:  B x T x heads x C Tensor containing probabilities for each class
                 labels: B x T x heads LongTensor containing label indices
        """

        x = self.embedding(expression_one_hot)

        for layer in self.context_layers:
            x = layer(x, audio_code)
            x = F.leaky_relu(x, 0.2)

        logits = self.logits(x)
        logprobs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logprobs, dim=-1)
        labels = th.argmax(logprobs, dim=-1)

        return {"logprobs": logprobs, "probs": probs, "labels": labels}
