# `deepracer_genesis.algorithms`

The built-in PPO runs through rsl-rl's `OnPolicyRunner`; a custom algorithm
(`Algo(cls=...)`) plugs into the **same** runner by implementing the
`RslAlgorithm` interface (see the [custom-algorithms guide](../guides/custom-algorithms.md)).

::: deepracer_genesis.algorithms.protocol

::: deepracer_genesis.algorithms.rsl_rl
