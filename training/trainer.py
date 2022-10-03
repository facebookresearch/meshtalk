"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import time
import tqdm
import torch as th
import torch.optim as optim

from torch.utils.data import DataLoader


def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


class Trainer:
    def __init__(self, config, forwarder, train_dataset, val_dataset):
        self.config = config
        self.forwarder = forwarder
        self.val_dataset = val_dataset
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=self.config["train"]["batch_size"],
                                           shuffle=True,
                                           drop_last=True,
                                           num_workers=self.config["dataloader_workers"])
        self.train_dataloader = cycle(self.train_dataloader)

        self.optimizer = {}
        for model_name, model in forwarder.models().items():
            self.optimizer[model_name] = optim.Adam(
                filter(lambda x: x.requires_grad, model.parameters()),
                lr=self.config["train"]["learning_rate"],
                **(self.forwarder.train_args(model_name))
            )

    def save(self, suffix=""):
        for model in self.forwarder.models().values():
            model.save(self.config["artifacts_dir"], suffix)

    def train(self):
        t = time.time()
        for iters in tqdm.tqdm(range(1, self.config["train"]["max_iterations"] + 1)):
            # training
            data = next(self.train_dataloader)
            loss_dict = self.training_step(data)

            # decrease learning rate if requested in config
            if iters in self.config["train"]["decrease_lr_iters"]:
                for opt in self.optimizer.values():
                    for param_group in opt.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.1

            if iters % 1000 == 0:
                # process random batch from validation set
                loss_val_dict = self.validation_step()
                # log some information about this iteration
                iter_str = "iter: {:6d}".format(iters)
                loss_str = ", ".join(["{}: {:.3f}".format(k, v.item()) for k, v in loss_dict.items()])
                loss_val_str = ", ".join(["{}: {:.3f}".format(k, v.item()) for k, v in loss_val_dict.items()])
                t = time.localtime(time.time() - t)
                duration = f"time: {t.tm_min:02d}:{t.tm_sec:02d}"
                print(", ".join([iter_str, loss_str, loss_val_str, duration]))
                t = time.time()

        # save model
        self.save()

    def training_step(self, data):
        # phase -> train
        for model in self.forwarder.models().values():
            model.train()

        # train generator
        for opt in self.optimizer.values():
            opt.zero_grad()

        loss_dict = self.forwarder.forward(data)
        loss = [loss_dict[k] * self.config["train"]["loss_weights"].get(k, 1.0) for k in self.config["train"]["loss_terms"]]
        loss = th.sum(th.stack(loss))
        loss.backward()

        for opt in self.optimizer.values():
            opt.step()

        return loss_dict

    def validation_step(self):
        # phase -> eval
        for model in self.forwarder.models().values():
            model.eval()

        val_dataloader = DataLoader(self.val_dataset,
                                    batch_size=self.config["train"]["batch_size"],
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=1)
        val_dataloader = cycle(val_dataloader)
        data = next(val_dataloader)
        loss_dict = self.forwarder.forward(data)
        loss_dict = {k + "_val": v for k, v in loss_dict.items()}

        return loss_dict
