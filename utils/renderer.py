"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import ffmpeg
import numpy as np
import torch as th
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    Textures,
)


class Renderer:
    def __init__(self, face_topo_file: str):
        """
        :param face_topo_file: .obj file containing face topology
        """
        if th.cuda.is_available():
            self.device = th.device("cuda:0")
            th.cuda.set_device(self.device)
        else:
            self.device = th.device("cpu")
        verts, faces_idx, _ = load_obj(face_topo_file)
        self.verts_rgb = th.ones_like(verts)[None] * th.Tensor([0.529, 0.807, 0.980])[None, None, :]  # (1, V, 3)
        self.verts_rgb = self.verts_rgb.to(self.device)
        self.faces = faces_idx.verts_idx.to(self.device)[None, :, :]

    def render(self, verts: th.Tensor):
        """
        :param verts: B x V x 3 tensor containing a batch of face vertex positions to be rendered
        :return: B x 640 x 480 x 4 tensor containing the rendered images
        """
        R, T = look_at_view_transform(7.5, 0, 20)
        focal = th.tensor([5.0], dtype=th.float32).to(self.device)
        princpt = th.tensor([0.1, 0.1], dtype=th.float32).to(self.device).unsqueeze(0)

        cameras = PerspectiveCameras(device=self.device, focal_length=focal, R=R, T=T, principal_point=princpt)

        raster_settings = RasterizationSettings(
            image_size=[640, 480],
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        lights = PointLights(device=self.device, location=[[0.0, 0.0, 10.0]])

        verts = verts * 0.01
        textures = Textures(verts_rgb=self.verts_rgb.expand(verts.shape[0], -1, -1))
        mesh = Meshes(
            verts=verts.to(self.device),
            faces=self.faces.expand(verts.shape[0], -1, -1),
            textures=textures
        )

        with th.no_grad():
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=SoftPhongShader(
                    device=self.device,
                    cameras=cameras,
                    lights=lights
                )
            )
            images = renderer(mesh)

        return images

    def to_video(self, verts: th.Tensor, audio_file: str, video_output: str, fps: int = 30, batch_size: int = 30):
        """
        :param verts: B x V x 3 tensor containing a batch of face vertex positions to be rendered
        :param audio_file: filename of the audio input file
        :param video_output: filename of the output video file
        :param fps: frame rate of output video
        :param batch_size: number of frames to render simultaneously in one batch
        """
        if not video_output[-4:] == '.mp4':
            video_output = video_output + '.mp4'

        images = th.cat([self.render(v).cpu() for v in th.split(verts, batch_size)], dim=0)
        images = 255 * images[:, :, :, :3].contiguous().numpy()
        images = images.astype(np.uint8)

        video_stream = ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", s="480x640", r=fps)
        audio_stream = ffmpeg.input(filename=audio_file)
        streams = [video_stream, audio_stream]
        output_args = {
            "format": "mp4",
            "pix_fmt": "yuv420p",
            "vcodec": "libx264",
            "movflags": "frag_keyframe+empty_moov+faststart"
        }
        proc = (
            ffmpeg
            .output(*streams, video_output, **output_args)
            .overwrite_output()
            .global_args("-loglevel", "fatal")
            .run_async(pipe_stdin=True, pipe_stdout=False)
        )

        proc.communicate(input=images.tobytes())
