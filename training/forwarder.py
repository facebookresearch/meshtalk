"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import random
import torch as th


def _gumbel_softmax(logprobs, tau=1.0, argmax=False):
    if argmax:
        logits = logprobs / tau
    else:
        g = -th.log(-th.log(th.clamp(th.rand(logprobs.shape, device=logprobs.device), min=1e-10, max=1)))
        logits = (g + logprobs) / tau
    soft_labels = th.softmax(logits, dim=-1)
    labels = soft_labels.detach().argmax(dim=-1, keepdim=True)
    hard_labels = th.zeros(logits.shape, device=logits.device)
    hard_labels = hard_labels.scatter(-1, labels, 1.0)
    one_hot = hard_labels.detach() - soft_labels.detach() + soft_labels
    return one_hot, labels.squeeze(-1)


def quantize(logprobs, argmax=False):
    """
    :param logprobs: B x T x heads x classes
           argmax: use argmax instead of sampling gumble noise if True
    :return: one_hot: B x T x heads x classes, where each class column is a one hot vector
             labels: B x T x heads (LongTensor), where each element is a class label
    """
    one_hot, labels = _gumbel_softmax(logprobs, argmax=argmax)
    return {
        "one_hot": one_hot,
        "labels": labels
    }


class DiscreteExpressionForwarder:
    """ this is the class to train the expression codes (train_step1.py) """
    def __init__(self, config, geom_unet, encoder, mouth_mask, eye_mask, landmarks):
        self.config = config
        self.mouth_mask = th.from_numpy(mouth_mask).type(th.float32).cuda()
        self.eye_mask = th.from_numpy(eye_mask).type(th.float32).cuda()
        self.landmarks = th.from_numpy(landmarks).type(th.float32).cuda()
        self.geom_unet = geom_unet.cuda()
        self.encoder = encoder.cuda()

    def models(self):
        return {
            "geom_unet": self.geom_unet,
            "encoder": self.encoder,
        }

    def train_args(self, model_name):
        if model_name == "geom_unet":
            return {"weight_decay": self.config["train"]["weight_decay"]}
        else:
            return {}

    def load(self, model_dir, suffix=""):
        self.geom_unet.load(model_dir, suffix)
        self.encoder.load(model_dir, suffix)

    def random_shift(self, size):
        return (th.arange(size) + random.randint(1, size-1)) % size

    def _reconstruct(self, template_geom, expression_code, audio_code):
        logprobs = self.encoder.fuse(expression_code, audio_code)["logprobs"]
        z = quantize(logprobs)["one_hot"]
        recon = self.geom_unet(
            template_geom.unsqueeze(1).expand(-1, z.shape[1], -1, -1).contiguous(), z
        )["geom"]
        return recon

    def forward(self, data):
        template = data["template"].cuda()
        geom = data["geom"].cuda()
        audio = data["audio"].cuda()
        B, T = geom.shape[0], geom.shape[1]
        loss_dict = {}

        enc = self.encoder.encode(geom, audio)
        recon = self._reconstruct(template, enc["expression_code"], enc["audio_code"])

        # compute losses
        if "recon" in self.config["train"]["loss_terms"]:
            l2_loss = th.mean((recon - geom) ** 2)
            loss_dict.update({"recon": l2_loss})

        if "landmarks" in self.config["train"]["loss_terms"]:
            lmk_loss = th.sum(((recon - geom) ** 2) * self.landmarks[None, None, :, None]) / \
                       (B * T * th.sum(self.landmarks) * 3)
            loss_dict.update({"landmarks": lmk_loss})

        if "modality_crossing" in self.config["train"]["loss_terms"]:
            # keep audio, switch expression
            cross_recon = self._reconstruct(template,
                                            enc["expression_code"][self.random_shift(B), :, :],
                                            enc["audio_code"]
                                            )
            audio_consistency_loss = th.sum(((cross_recon - geom) ** 2) * self.mouth_mask[None, None, :, None]) / \
                                     (B * T * th.sum(self.mouth_mask) * 3)
            # keep expression, switch audio
            cross_recon = self._reconstruct(template,
                                            enc["expression_code"],
                                            enc["audio_code"][self.random_shift(B), :, :]
                                            )
            expression_consistency_loss = th.sum(((cross_recon - geom) ** 2) * self.eye_mask[None, None, :, None]) / \
                                          (B * T * th.sum(self.eye_mask) * 3)
            # add modality crossing loss to loss_dict
            loss_dict.update({"modality_crossing": audio_consistency_loss + expression_consistency_loss})

        return loss_dict


class CategoricalAutoregressiveForwarder:
    """ this is the class to train the autoregressive model (train_step2.py) """
    def __init__(self, config, encoder, context_model):
        self.config = config
        self.context_model = context_model.cuda()
        # the encoder is a pre-trained model and is expected to be passed with loaded weights
        # the encoder is also hidden from the trainer
        self.encoder = encoder.cuda().eval()

    def models(self):
        return {
            "context_model": self.context_model
        }

    def train_args(self, model_name):
        return {}

    def load(self, model_dir, suffix=""):
        self.context_model.load(model_dir, suffix)

    def forward(self, data):
        geom = data["geom"].cuda()
        audio = data["audio"].cuda()

        enc = self.encoder(geom, audio)
        quantized = quantize(enc["logprobs"], argmax=True)
        one_hot = quantized["one_hot"].contiguous().detach()
        target_labels = quantized["labels"].contiguous().detach()

        logprobs = self.context_model(one_hot, enc["audio_code"])["logprobs"]

        loss_dict = {}

        if "cross_entropy" in self.config["train"]["loss_terms"]:
            ce_loss = th.nn.functional.nll_loss(logprobs.view(-1, logprobs.shape[-1]), target_labels.view(-1))
            loss_dict.update({"cross_entropy": ce_loss})

        return loss_dict
