from __future__ import annotations

import dataclasses
import os

import numpy as np
import pandas as pd
import numpy.typing as npt

from pathlib import Path
from skimage.measure import regionprops_table
from typing import Dict, List, Optional, Tuple


# try to use dask, fall back to imageio
try:
    from dask.array.image import imread
except ImportError:
    from skimage.io import imread


@dataclasses.dataclass
class CTCDataset:
    """Cell Tracking Challenge dataset object."""
    name: str 
    path: os.PathLike 
    experiment: str
    scale: Tuple[float] = dataclasses.field(default_factory=tuple)
    _segmentation: Optional[npt.ArrayLike] = None
    _images: Optional[npt.ArrayLike] = None
    _nodes: Optional[pd.DataFrame] = None

    @property 
    def images(self) -> npt.ArrayLike:
        if self._images is None: 
            images_filepattern = str(Path(self.path) / f"{self.experiment}" / "*.tif")
            self._images = imread(images_filepattern)
        return self._images

    @property 
    def segmentation(self) -> npt.ArrayLike:
        if self._segmentation is None:
            filepath = Path(self.path) / f"{self.experiment}_GT/TRA" 
            segmentation_filepattern = str(filepath / "*.tif")
            self._segmentation = imread(segmentation_filepattern)
        return self._segmentation

    @property 
    def nodes(self) -> pd.DataFrame:
        if self._nodes is None:
            self._nodes = _nodes_from_stack(self.segmentation)
        return self._nodes
    
    @property 
    def edges(self):
        raise NotImplementedError

    @property 
    def graph(self) -> Dict[int, List[int]]:
        """Return the lineage graph from the file."""
        filepath = Path(self.path) / f"{self.experiment}_GT/TRA"
        lbep = np.loadtxt(filepath / "man_track.txt", dtype=np.uint)
        full_graph = dict(lbep[:, [0, 3]])
        graph = {k: v for k, v in full_graph.items() if v != 0}
        return graph
    
    @property 
    def volume(self) -> tuple:
        dims = self.segmentation.shape[1:]
        ndim = len(dims)
        scaled_dims = [dims[idx]*self.scale[idx] for idx in range(ndim)]
        return tuple(zip([0]*ndim, scaled_dims))


def _nodes_from_image(stack: np.ndarray, idx: int) -> pd.DataFrame:
    """Return the unique track label, centroid and time for each track vertex.

    Parameters
    ----------
    stack : array
        Segmentation stack.
    idx : int
        Index of the image to calculate the centroids and track labels.

    Returns
    -------
    data_df : pd.DataFrame
       The dataframe of track data for one time step (specified by idx).
    """
    frame = np.asarray(stack[idx, ...])
    props = regionprops_table(frame, properties=("label", "centroid"))
    props["t"] = np.full(props["label"].shape, idx)
    return pd.DataFrame(props)


def _nodes_from_stack(stack: npt.Array) -> pd.DataFrame:
    """Return detections from a stack.
    
    Parameters
    ----------
    stack : array
        Segmentation stack.

    Returns
    -------
    data_df : pd.DataFrame
       The dataframe containing the track data.
    """
    data_df_raw = pd.concat(
        [_nodes_from_image(stack, idx) for idx in range(stack.shape[0])]
    ).reset_index(drop=True)
    # sort the data lexicographically by track_id and time
    data_df = data_df_raw.sort_values(["label", "t"], ignore_index=True)
    rename_map={
        "label": "GT_ID", 
        "centroid-2": "x", 
        "centroid-1": "y", 
        "centroid-0": "z",
    }
    data_df = data_df.rename(columns=rename_map)

    # create the final data array: track_id, T, Z, Y, X
    keys = ["GT_ID", "t", "z", "y", "x"] 
    if "z" not in data_df.keys():
        keys.remove("z")

    data = data_df.loc[:, keys]
    return data


def load(
    path: os.PathLike, **kwargs,
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Load a Cell Tracking Challenge dataset.

    Parameters 
    ----------
    path :
        The path to the root of the dataset.
    experiment : str, default is "01"
        The particular experiment within the dataset.

    Returns
    -------
    dataset :
        An instance of a CTCDataSet containing
    Usage
    -----
    >>> dataset = load_ctc(PATH, experiment=EXPERIMENT)
    """
    dataset_path = Path(path)
    name = dataset_path.stem
    return CTCDataset(name=name, path=dataset_path, **kwargs)
