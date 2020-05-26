
"""Dataset class for GQN."""

from typing import Tuple

import gzip
import pathlib

import torch
from torch import Tensor


class SceneDataset(torch.utils.data.Dataset):
    """SceneDataset class for GQN.

    SceneDataset class loads data files at each time accessed by index.

    Args:
        root_dir (str): Path to root directory
    """

    def __init__(self, root_dir: str):
        super().__init__()

        self.root_dir = pathlib.Path(root_dir)

    def __len__(self) -> int:
        """Returns number of files and directories in root dir.

        Returns:
            len (int): Number of objects in root dir.
        """

        return len(list(self.root_dir.iterdir()))

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        """Load and get data with specified index.

        This method read `<index>.pt.gz` file, which includes list of tuples,
        `(images, viewpoints)`.

        Args:
            index (int): Index number.

        Returns:
            images (torch.Tensor): Image tensor, size
                `(batch, seqlen, 3, 64, 64)`.
            viewpoints (torch.Tensor): View points, size `(batch, seqlen, 7)`.
        """

        filename = f"{index}.pt.gz"
        with gzip.open(self.root_dir / filename, "rb") as f:
            dataset = torch.load(f)

        images = torch.stack([torch.from_numpy(data[0]) for data in dataset])
        viewpoints = torch.stack(
            [torch.from_numpy(data[1]) for data in dataset])

        # Convert data size: BNHWC -> BNCHW
        images = images.permute(0, 1, 4, 2, 3)

        # Transform viewpoints
        viewpoints = transform_viewpoint(viewpoints)

        return images, viewpoints


def transform_viewpoint(viewpoints: Tensor) -> Tensor:
    """Transforms viewpoints.

    (x, y, z, yaw, pitch)
        -> (x, y, z, cos(yaw), sin(yaw), cos(pitch), sin(pitch))

    Args:
        viewpoints (torch.Tensor): Input viewpoints, size `(*, 5)`.

    Returns:
        converted (torch.Tensor): Transformed viewpoints, size `(*, 7)`.
    """

    pos, tmp = torch.split(viewpoints, 3, dim=-1)
    yaw, pitch = torch.split(tmp, 1, dim=-1)

    view = [pos, torch.cos(yaw), torch.sin(yaw),
            torch.cos(pitch), torch.sin(pitch)]
    view = torch.cat(view, dim=-1)
    return view
