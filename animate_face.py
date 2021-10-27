"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import numpy as np
import torch as th

from utils.renderer import Renderer
from utils.helpers import smooth_geom, load_mask, get_template_verts, load_audio, audio_chunking
from models.vertex_unet import VertexUnet
from models.context_model import ContextModel
from models.encoders import MultimodalEncoder


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",
                    type=str,
                    default="pretrained_models",
                    help="directory containing the models to load")
parser.add_argument("--audio_file",
                    type=str,
                    default="assets/example_sentence.wav",
                    help="wave file to use for face animation"
                    )
parser.add_argument("--face_template",
                    type=str,
                    default="assets/face_template.obj",
                    help=".obj file containing neutral template mesh"
                    )
parser.add_argument("--output",
                    type=str,
                    default="video.mp4",
                    help="video output file"
                    )
args = parser.parse_args()

"""
load assets
"""
print("load assets...")
template_verts = get_template_verts(args.face_template)
audio = load_audio(args.audio_file)
mean = th.from_numpy(np.load("assets/face_mean.npy"))
stddev = th.from_numpy(np.load("assets/face_std.npy"))
forehead_mask = th.from_numpy(load_mask("assets/forehead_mask.txt", dtype=np.float32)).cuda()
neck_mask = th.from_numpy(load_mask("assets/neck_mask.txt", dtype=np.float32)).cuda()

renderer = Renderer("assets/face_template.obj")

"""
load models
"""
print("load models...")
geom_unet = VertexUnet(classes=128,
                       heads=16,
                       n_vertices=6172,
                       mean=mean,
                       stddev=stddev,
                       )
geom_unet.load(args.model_dir)
geom_unet.cuda().eval()
context_model = ContextModel(classes=128,
                             heads=16,
                             audio_dim=128
                             )
context_model.load(args.model_dir)
context_model.cuda().eval()
encoder = MultimodalEncoder(classes=128,
                            heads=16,
                            expression_dim=128,
                            audio_dim=128,
                            n_vertices=6172,
                            mean=mean,
                            stddev=stddev,
                            )
encoder.load(args.model_dir)
encoder.cuda().eval()

"""
generate and render sequence
"""
print("animate face mesh...")
# run template mesh and audio through networks
audio = audio_chunking(audio, frame_rate=30, chunk_size=16000)
with th.no_grad():
    audio_enc = encoder.audio_encoder(audio.cuda().unsqueeze(0))["code"]
    one_hot = context_model.sample(audio_enc, argmax=False)["one_hot"]
    T = one_hot.shape[1]
    geom = template_verts.cuda().view(1, 1, 6172, 3).expand(-1, T, -1, -1).contiguous()
    result = geom_unet(geom, one_hot)["geom"].squeeze(0)
# smooth results
result = smooth_geom(result, forehead_mask)
result = smooth_geom(result, neck_mask)
# render sequence
print("render...")
renderer.to_video(result, args.audio_file, args.output)
print("done")
