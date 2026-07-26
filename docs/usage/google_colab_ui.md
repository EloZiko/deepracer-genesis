# Google Colab (UI / notebooks)

The Colab notebooks are **code-first, not widget-first**: there are no ipywidgets,
forms, or sliders. You configure everything by editing Python cells (config-as-code)
and re-running — the same declarative pattern as local training.

> Mental model in one sentence: "UI" here means the notebook cells you edit —
> experiment classes, track waypoints, or an Optuna search space — not a GUI.

---

## Notebooks

| Notebook | What you edit | Flow |
|----------|---------------|------|
| `deepracer_genesis_colab.ipynb` | an `Experiment` class | edit attributes → `run()` → `rollout_video()` → save to Drive (see [Colab CLI](google_colab_cli.md)) |
| `track_designer.ipynb` | a list of waypoints | `route_from_waypoints()` → matplotlib plot → `install_track(name, route)` → sanity-drive → train |
| `hpo_cnn.ipynb` | an Optuna search space | `sample_spec(trial)` → `study.optimize()` → plot importances/history → export best to ONNX |

## Track designer

```python
route = route_from_waypoints([(0, 0), (2, 0), (3, 1.5), ...])
# plot to check the centerline/edges with matplotlib, then:
install_track("my_track", route)
# reference it later:
FeatureEnvironment(tracks=("my_track",)) >> VectorPolicy()
```

There are no drawing widgets — you edit the waypoint tuples and re-run the cell to
re-plot. See [Tracks](../concepts/tracks.md) for the underlying geometry.

## HPO notebook

`hpo_cnn.ipynb` mirrors `experiments/hpo_optuna.py` but searches CNN architecture
(channels, kernels, activation) for a camera policy, then visualizes parameter
importances and the optimization history with matplotlib. See the
[HPO guide](../guides/hpo.md) for the study/search-space mechanics.

---

Because every workflow is edit-a-cell-and-rerun, anything you can express in the
[experiment DSL](../concepts/experiments.md) works identically in Colab and locally.
