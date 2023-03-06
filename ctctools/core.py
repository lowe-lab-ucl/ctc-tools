
import os

import numpy as np
import pandas as pd

from pathlib import Path
from skimage.measure import regionprops_table
from typing import Dict, Union

# try to use dask, fall back to imageio
try:
    from dask_image.imread import imread
except ImportError:
    from skimage.io import imread


def _detections_from_image(stack: np.ndarray, idx: int) -> pd.DataFrame:
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
    props = regionprops_table(
        np.asarray(stack[idx, ...]), 
        properties=("label", "centroid")
    )
    props["time"] = np.full(props["label"].shape, idx)
    return pd.DataFrame(props)


def detections_from_stack(stack: np.ndarray) -> pd.DataFrame:
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
        [_detections_from_image(stack, idx) for idx in range(stack.shape[0])]
    ).reset_index(drop=True)
    # sort the data lexicographically by track_id and time
    data_df = data_df_raw.sort_values(["label", "time"], ignore_index=True)
    data_df = data_df.rename(
        columns={
            "label": "ID", 
            "centroid-2": "z", 
            "centroid-1": "y", 
            "centroid-0": "x"
        }
    )

    # create the final data array: track_id, T, Z, Y, X
    keys = ["ID", "time", "x", "y"] 
    if "z" in data_df.keys():
        keys += ["z"]

    data = data_df.loc[:, keys]
    return data


def load_graph(filepath: os.PathLike) -> Dict[int, int]:
    """Return the lineage graph from the file."""
    lbep = np.loadtxt(filepath / "man_track.txt", dtype=np.uint)
    full_graph = dict(lbep[:, [0, 3]])
    graph = {k: v for k, v in full_graph.items() if v != 0}
    return graph


def load_ctc(
    rootpath: os.PathLike, *, experiment: str = "01"
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Load a Cell Tracking Challenge dataset.

    Parameters 
    ----------
    root : path 
        The path to the root of the dataset.
    experiment : str, default is "01"
        The particular experiment within the dataset.

    Returns
    -------
    detections : dataframe 
        A dataframe containing the ground truth centroids for each detection.
    graph : dict
        A dictionary containing the parent-child relationships derived from
        the LBEP table. For example: {2: 1, 3: 1} indicates that tracks 2 and 3 
        are children of track 1.

    Usage
    -----
    >>> detections, graph = load_ctc(PATH, experiment=EXPERIMENT)
    """
    filepath = Path(rootpath) / f"{experiment}_GT/TRA"

    segmentation_filepattern = filepath / "man_track*.tif"
    stack = imread(segmentation_filepattern)

    detections = detections_from_stack(stack)
    graph = load_graph(filepath)
    return detections, graph
