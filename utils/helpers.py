"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch as th
import torchaudio as ta
from pytorch3d.io import load_obj


def load_mask(mask_file: str, dtype = bool):
    """
    :param mask_file: filename of mask to load
    :param dtype: python type, bool for binary masks, np.float32 for float masks
    :return: np.array containing the loaded mask of type dtype
    """
    return np.loadtxt(mask_file).astype(dtype).flatten()


def load_audio(wave_file: str):
    """
    :param wave_file: .wav file containing the audio input
    :return: 1 x T tensor containing input audio resampled to 16kHz
    """
    audio, sr = ta.load(wave_file)
    if not sr == 16000:
        audio = ta.transforms.Resample(sr, 16000)(audio)
    if audio.shape[0] > 1:
        audio = th.mean(audio, dim=0, keepdim=True)
    # normalize such that energy matches average energy of audio used in training
    audio = 0.01 * audio / th.mean(th.abs(audio))
    return audio


def audio_chunking(audio: th.Tensor, frame_rate: int = 30, chunk_size: int = 16000):
    """
    :param audio: 1 x T tensor containing a 16kHz audio signal
    :param frame_rate: frame rate for video (we need one audio chunk per video frame)
    :param chunk_size: number of audio samples per chunk
    :return: num_chunks x chunk_size tensor containing sliced audio
    """
    samples_per_frame = 16000 // frame_rate
    padding = (chunk_size - samples_per_frame) // 2
    audio = th.nn.functional.pad(audio.unsqueeze(0), pad=[padding, padding]).squeeze(0)
    anchor_points = list(range(chunk_size//2, audio.shape[-1]-chunk_size//2, samples_per_frame))
    audio = th.cat([audio[:, i-chunk_size//2:i+chunk_size//2] for i in anchor_points], dim=0)
    return audio


def smooth_geom(geom, mask: th.Tensor = None, filter_size: int = 9, sigma: float = 2.0):
    """
    :param geom: T x V x 3 tensor containing a temporal sequence of length T with V vertices in each frame
    :param mask: V-dimensional Tensor containing a mask with vertices to be smoothed
    :param filter_size: size of the Gaussian filter
    :param sigma: standard deviation of the Gaussian filter
    :return: T x V x 3 tensor containing smoothed geometry (i.e., smoothed in the area indicated by the mask)
    """
    assert filter_size % 2 == 1, f"filter size must be odd but is {filter_size}"
    # Gaussian smoothing (low-pass filtering)
    fltr = np.arange(-(filter_size // 2), filter_size // 2 + 1)
    fltr = np.exp(-0.5 * fltr ** 2 / sigma ** 2)
    fltr = th.Tensor(fltr) / np.sum(fltr)
    # apply fltr
    fltr = fltr.view(1, 1, -1).to(device=geom.device)
    T, V = geom.shape[0], geom.shape[1]
    g = th.nn.functional.pad(
        geom.permute(1, 2, 0).view(V * 3, 1, T),
        pad=[filter_size // 2, filter_size // 2], mode='replicate'
    )
    g = th.nn.functional.conv1d(g, fltr).view(V, 3, T)
    smoothed = g.permute(2, 0, 1).contiguous()
    # blend smoothed signal with original signal
    if mask is None:
        return smoothed
    else:
        return smoothed * mask[None, :, None] + geom * (-mask[None, :, None] + 1)


def get_template_verts(template_mesh: str):
    """
    :param template_mesh: .obj file containing the neutral face template mesh
    :return: V x 3 tensor containing the template vertices
    """
    verts, _, _ = load_obj(template_mesh)
    return verts


class Net(th.nn.Module):

    def __init__(self, model_name: str = "network"):
        """
        :param model_name: name of the model
        """
        super().__init__()
        self.model_name = model_name

    def save(self, model_dir: str, suffix: str = ''):
        """
        :param model_dir: directory where the model should be stored
        :param suffix: option suffix to append to the network name
        """
        self.cpu()
        if suffix == "":
            fname = f"{model_dir}/{self.model_name}.pkl"
        else:
            fname = f"{model_dir}/{self.model_name}.{suffix}.pkl"
        th.save(self.state_dict(), fname)
        self.cuda()
        return self

    def load(self, model_dir: str, suffix: str = ''):
        """
        :param model_dir: directory where the model is be stored
        :param suffix: optional suffix to append to the network name
        """
        self.cpu()
        if suffix == "":
            fname = f"{model_dir}/{self.model_name}.pkl"
        else:
            fname = f"{model_dir}/{self.model_name}.{suffix}.pkl"
        states = th.load(fname)
        self.load_state_dict(states)
        self.cuda()
        print("Loaded:", fname)
        return self

    def num_trainable_parameters(self):
        """
        :return: number of trainable parameters in the model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
