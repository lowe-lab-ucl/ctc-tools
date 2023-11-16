# ctc-tools

Python tools to help import [Cell Tracking Challenge](http://celltrackingchallenge.net/) datasets.


## Usage

```python
import ctctools

# note: `scale` allows representation of anisotropy in the image data
dataset = ctctools.load("/path/to/data", experiment="01", scale=(1., 1., 1.))
```

and to visualize in napari:
```python
viewer = napari.Viewer()
viewer.add_image(
    dataset.images, 
    name=dataset.name
)
viewer.add_tracks(
    dataset.nodes.to_numpy(), 
    graph=dataset.graph, 
    name="GT tracks"
)
```

## Usage in Google Colab

An example colab notebook is provided below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hB04DwJZJzT9i_yi_mm7p8vkgWPzbfUA?usp=sharing)
