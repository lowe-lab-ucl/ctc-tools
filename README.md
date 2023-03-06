# ctc-tools
CTC tools

```python
from ctctools import load_ctc

detections, graph = load_ctc("/path/to/data", experiment="01)
```

and to visualize in napari:
```python
viewer = napari.Viewer()
viewer.add_tracks(detections.to_numpy(), graph=graph, name="Tracks")
```
