# Custom algorithms

Algorithms are passed as parameters, not registered. Implement the `Algorithm`
protocol and hand your class to the `Algo` stage — no decorator, no global registry.

> Mental model in one sentence: an algorithm is a class the `Trainer` calls —
> `setup(builder)` once, then `train_on_batch(data)` per collector yield — and you
> select it with `>> Algo(cls=MyAlgo)`.

---

## The protocol

`deepracer_genesis/algorithms/protocol.py` defines a `runtime_checkable` `Algorithm`
Protocol:

```python
class Algorithm(Protocol):
    requires_cost: ClassVar[bool] = False   # True → needs a cost stream (safe RL)

    def setup(self, builder: Builder) -> None: ...        # build nets/losses/optim
    @property
    def collect_policy(self): ...                          # module the collector runs
    @property
    def eval_actor(self): ...                              # deterministic eval actor
    def train_on_batch(self, data) -> dict[str, float]: .. # one batch → scalar logs
    def observe_env_logs(self, logs: dict) -> None: ...    # per-iteration sim stats
    def checkpoint(self) -> dict: ...                      # extra state to persist
```

The `Builder` (`experiment/builder.py`) hands you the pieces so you don't reimplement
wiring: `builder.actor()`, `builder.critic(out_key=...)`, `builder.gae(critic)`,
`builder.loss(actor, critic)`, `builder.collector(env, policy)`, `builder.buffer()`,
`builder.optimizer(loss_module)`.

## Built-ins

- **`PPO`** (`algorithms/ppo.py`, `requires_cost = False`) — clipped PPO; reads its
  hyperparameters from `spec.algorithm.ppo`.
- **`PPOLagrangian`** (`algorithms/lagrangian.py`, `requires_cost = True`) — extends
  PPO with a second **cost critic** and a `PIDLagrangian` controller that adjusts a
  multiplier λ from the mean episode cost (via `observe_env_logs`), reweighting the
  advantage as `(A − λ·A_cost)/(1 + λ)`.

`requires_cost = True` makes the trainer enforce that the env emits a cost stream and
a budget is set (see the safe-RL envs in [Experiments](../concepts/experiments.md)).

## Adding one

```python
from typing import Any
from tensordict import TensorDictBase

class MyAlgo:
    requires_cost = False

    def setup(self, builder) -> None:
        self.actor  = builder.actor()
        self.critic = builder.critic()
        self.gae    = builder.gae(self.critic)
        self.loss   = builder.loss(self.actor, self.critic)
        self.optim  = builder.optimizer(self.loss)

    @property
    def collect_policy(self): return self.actor
    @property
    def eval_actor(self):     return self.actor

    def train_on_batch(self, data: TensorDictBase) -> dict[str, float]:
        data = self.gae(data)
        # ... your update ...
        return {"Loss/mine": 0.0}

    def observe_env_logs(self, logs: dict[str, Any]) -> None: ...
    def checkpoint(self) -> dict[str, Any]:
        return {"actor": self.actor.state_dict(),
                "critic": self.critic.state_dict()}
```

Select it in a chain:

```python
spec = (
    FeatureEnvironment(num_envs=512)
    >> VectorPolicy(keys=("state",))
    >> Algo(cls=MyAlgo, params={"learning_rate": 1e-3})
).build(seed=0)
```

For a **safe** algorithm, set `requires_cost = True`, read `budget`/`pid` from
`builder.spec.algorithm.lagrangian`, build a cost critic with
`builder.critic(out_key="cost_value")`, and pair it with a `SafeRLFeatureEnvironment`
(or `SafeRLCameraEnvironment`). `notebooks/custom_algorithm_reinforce.ipynb` shows a
full from-scratch REINFORCE.
