"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch as th


class DataReader:
    def __init__(
            self,
            segment_length: int = 32,
            n_vertices: int = 6172,
            audio_length: int = 16000
    ):
        """
        TODO: this is only a placeholder loading zeros. Implement the dataloader for your dataset here.
        :param segment_length: length of an animated segment used for training. MeshTalk used 32 frames long segments.
        :param n_vertices: number of vertices in mesh
        :param audio_length: number of samples in provided audio. Note that audio is assumed to be at 16kHz
        """
        self.segment_length = segment_length
        self.n_vertices = n_vertices
        self.audio_length = audio_length
        self.dataset_size = 100  # placeholder for dataset size
        self.current_idx = 0

    def __len__(self):
        return self.dataset_size

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        """
        :param idx: pointer to dataset element to be read
        :return: template: V x 3 tensor containing neutral face template
                 geom: segment_length x V x 3 tensor containing animated segment for same subject as template mesh
                 audio: segment_length x audio_length tensor containing the input audio for each frame of the segment
        """
        template = th.zeros(self.n_vertices, 3)
        geom = th.zeros(self.segment_length, self.n_vertices, 3)
        audio = th.zeros(self.segment_length, self.audio_length)
        return {
            "template": template,
            "geom": geom,
            "audio": audio
        }
