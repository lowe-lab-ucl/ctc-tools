from __future__ import annotations

import dataclasses
import os

import numpy as np
import pandas as pd
import numpy.typing as npt

from pathlib import Path
from scipy.sparse import coo_array
from skimage.measure import regionprops_table
from typing import Dict, List, Optional, Tuple


# try to use dask, fall back to imageio
try:
    from dask.array.image import imread
except ImportError:
    from skimage.io import imread


def _nodes_to_edge_list(nodes: pd.Series) -> list[tuple[int]]:
    """Take a list of nodes and convert to an edge list."""
    return [(i, j) for i, j in zip(nodes, nodes[1:])]


@dataclasses.dataclass
class CellTrackingChallengeDataset:
    """Cell Tracking Challenge dataset object."""
    name: str 
    path: os.PathLike 
    experiment: str
    scale: Tuple[float] = dataclasses.field(default_factory=tuple)
    _segmentation: Optional[npt.ArrayLike] = None
    _images: Optional[npt.ArrayLike] = None
    _nodes: Optional[pd.DataFrame] = None
    _edges: Optional[list] = None

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
        """Nodes in the graph."""
        if self._nodes is None:
            self._nodes = _nodes_from_stack(self.segmentation)
        return self._nodes
    
    @property 
    def edges(self) -> list[tuple[int, int]]:
        """Edges in the graph."""

        if self._edges is not None:
            return self._edges

        tracks = self.nodes.groupby("GT_ID").apply(lambda x: x.index)
        self._edges = []

        for track in tracks:
            track_edges = _nodes_to_edge_list(track)
            self._edges += track_edges

        # now calculate the hypoeredges to add
        for idj, idi in self.graph.items():
            j = tracks.loc[idj].tolist()[0]
            i = tracks.loc[idi].tolist()[-1]
            self._edges.append((i, j))
        
        return self._edges

    def adjacency_matrix(self) -> coo_array:
        """Return an adjacency matrix of the edges of the graph.
        
        Returns
        -------
        adj_matrix : array
            Returns the tracks as an adjacency matrix of dimension N, M. Where 
            N is the number of nodes and M is the number of edges/hyperedges.

        Notes
        -----
        coo_array((data, (i, j)), [shape=(M, N)])

        to construct from three arrays:
            data[:] the entries of the matrix, in any order
            i[:] the row indices of the matrix entries
            j[:] the column indices of the matrix entries

        Where A[i[k], j[k]] = data[k]. When shape is not specified, it is 
        inferred from the index arrays.
        """

        rows, columns = zip(*self.edges)
        N = len(self.nodes)

        adj_matrix = coo_array(
            (
                [True] * len(rows),
                (list(rows), list(columns)),
            ),
            shape=(N, N),
            dtype=bool,
        )

        return adj_matrix

    @property 
    def graph(self) -> Dict[int, List[int]]:
        """Return the lineage graph from the file in a `napari` format."""
        filepath = Path(self.path) / f"{self.experiment}_GT/TRA"
        lbep = np.loadtxt(filepath / "man_track.txt", dtype=np.uint)
        full_graph = dict(lbep[:, [0, 3]])
        graph = {k: v for k, v in full_graph.items() if v != 0}
        return graph
    
    @property 
    def volume(self) -> tuple:
        """The image volume based on the segmentation shape."""
        dims = self.segmentation.shape[1:]
        ndim = len(dims)
        scaled_dims = [dims[idx]*self.scale[idx] for idx in range(ndim)]
        return tuple(zip([0]*ndim, scaled_dims))
    
    @staticmethod 
    def load(filepath: Path, **kwargs) -> CellTrackingChallengeDataset:
        """Load a CTC dataset"""
        dataset_path = Path(filepath)
        name = dataset_path.stem
        return CellTrackingChallengeDataset(name=name, path=dataset_path, **kwargs)


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

    dim_names = "xyz"[:stack.ndim-1][::-1]
    rename_map = {"label": "GT_ID"}
    for dim_idx, dim_name in enumerate(dim_names):
        rename_map.update({f"centroid-{dim_idx}": dim_name})

    data_df = data_df.rename(columns=rename_map)

    # create the final data array: track_id, T, Z, Y, X
    keys = ["GT_ID", "t", "z", "y", "x"] 
    if "z" not in data_df.keys():
        keys.remove("z")

    data = data_df.loc[:, keys]
    return data


def load(
    path: os.PathLike, **kwargs,
) -> CellTrackingChallengeDataset:
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
        An instance of a CellTrackingChallengeDataset containing the data.

    Usage
    -----
    >>> dataset = load_ctc(PATH, experiment=EXPERIMENT)
    """
    return CellTrackingChallengeDataset.load(path, **kwargs)
