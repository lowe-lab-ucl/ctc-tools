# ctc-tools

Tools to help import CTC datasets.

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
