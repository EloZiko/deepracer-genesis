# `Experiment` (authoring)

Subclass `Experiment` to author an experiment: training config as class attributes,
the `>>` chain in `pipeline()`, then `run(MyExperiment)` (or `MyExperiment().run()`).
See [Experiments & the `>>` DSL](../../concepts/experiments.md) for the walkthrough.

::: deepracer_genesis.experiment.Experiment
