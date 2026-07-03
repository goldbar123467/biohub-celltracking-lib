from __future__ import annotations


class UNet3DNotImplemented(RuntimeError):
    pass


def build_unet3d_detector(*args, **kwargs):
    raise UNet3DNotImplemented(
        "3D U-Net training is scaffolded but not implemented in the dependency-light baseline"
    )

