"""Frozen ExperimentSpec value the `>>` DSL builds, hashed by content into a
run id (identical configs share a run dir; runs retrain and overwrite)."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from ..envs.rewards import RewardFn


def _json_default(o):
    """Record a pluggable callable/class by its name so the spec dump stays
    JSON and the run-dir id stays stable."""
    return getattr(o, "__qualname__", None) or getattr(o, "__name__", None) or repr(o)


class SpecError(ValueError):
    """A structurally or semantically invalid experiment declaration."""


VALID_COST_FNS = ("offtrack", "offtrack_or_overspeed", "crash")


@dataclass(frozen=True)
class EnvSpec:
    """The simulator slice of the spec (filled by an Environment stage).

    Attributes:
        modality: whether the env yields camera images or feature vectors.
        render: which renderer backs a camera env (or none for feature envs).
        resolution: rendered image width/height.
        fov: camera field of view in degrees.
        lookahead_k: number of upcoming waypoints exposed to the policy.
        feature_set: which state-vector assembler to use ("classic"/"perception").
        feature_params: tuning knobs (horizons, history lengths) for the feature set.
        tracks: track names the env trains across.
        num_envs: parallel environment count.
        random_start: randomize spawn waypoint and lateral/yaw offset per episode.
        random_direction: flip driving direction (CW/CCW) per episode.
        reward: reward callable, or None for the built-in default.
        reward_scales: per-term scale overrides for the reward.
        emits_cost: whether the env produces a cost signal for SafeRL.
        cost_fn: which cost function to emit.
        cost_budget: per-episode cost budget.
    """

    modality: Literal["camera", "feature"]
    render: Literal["madrona", "nyx", "none"] = "none"
    resolution: tuple[int, int] = (160, 120)
    fov: float = 90.0
    lookahead_k: int = 10
    # which "state" vector the env assembles (envs/features.py):
    # "classic" (waypoint lookahead) or "perception" (CNN targets + error
    # channels for sim2real); feature_params tunes horizons/history lengths
    feature_set: str = "classic"
    feature_params: dict = field(default_factory=dict)
    tracks: tuple[str, ...] = ("reinvent_base",)
    num_envs: int = 512
    # spawn randomization: every episode starts at a random waypoint with
    # lateral/yaw noise; laps are measured by cumulative progress from the
    # spawn, so a "completed lap" ends back at that same random location
    random_start: bool = True
    # coin-flip the driving direction (CW vs CCW) each episode; heading /
    # progress / lookahead observations follow the chosen direction
    random_direction: bool = False
    # reward: a reward CALLABLE (envs/rewards.py: env -> {term: (N,) tensor}) +
    # scale overrides. None keeps the built-in `deepracer` default. The fn's
    # NAME is recorded in the run-dir id (not its body); runs always retrain.
    reward: "RewardFn | None" = None
    reward_scales: dict = field(default_factory=dict)
    emits_cost: bool = False
    cost_fn: Optional[str] = None
    cost_budget: Optional[float] = None


@dataclass(frozen=True)
class ObsDRSpec:
    """Observation-side domain randomization applied at reset/render.

    Attributes:
        image_aug: image augmentation settings for the rendered observation.
        camera_jitter: per-render camera pose/intrinsics jitter.
        physics: physics parameter overrides applied env-side at reset.
        appearance: per-env, per-episode color remap of the rendered scene.
    """

    image_aug: dict = field(default_factory=dict)
    camera_jitter: dict = field(default_factory=dict)
    physics: dict = field(default_factory=dict)   # applied env-side at reset
    # per-env, per-episode color remap of the rendered observation
    # ({"world_color": strength}); see DomainRandomizationTrackAppearance
    appearance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EncoderSpec:
    """Optional feature-encoder stage inserted between env and policy.

    Attributes:
        kind: which encoder to apply (none or a frozen CNN).
        checkpoint: path to the pretrained encoder weights.
        output_dim: dimensionality of the encoded feature vector.
        layer: which network layer to tap for features.
        out_key: observation key under which encoded features are exposed.
    """

    kind: Literal["none", "frozen_cnn"] = "none"
    checkpoint: Optional[str] = None
    output_dim: Optional[int] = None
    layer: Optional[str] = None
    out_key: str = "encoded"


@dataclass(frozen=True)
class PolicySpec:
    """The policy/network slice of the spec (filled by a Policy stage).

    Attributes:
        actor_keys: observation keys the actor consumes.
        critic_keys: observation keys the critic consumes.
        cnn: CNN config, or None for a pure vector policy.
        mlp: MLP head configuration.
        actions: discrete (steer, speed) action list, or None for continuous.
    """

    actor_keys: tuple[str, ...]
    critic_keys: tuple[str, ...]
    cnn: Optional[dict] = None               # None => pure vector policy
    mlp: dict = field(default_factory=dict)
    # None => continuous TanhNormal over [steer, speed] in [-1, 1]^2 (default).
    # A tuple of (steer, speed) pairs => DISCRETE Categorical policy over that
    # action list — the original AWS DeepRacer action-space style.
    actions: Optional[tuple] = None


@dataclass(frozen=True)
class ActionDRSpec:
    """Action-side domain randomization for continuous policies.

    Attributes:
        steer_noise: magnitude of noise added to the steering action.
        speed_noise: magnitude of noise added to the speed action.
        delay_steps: number of steps to delay applied actions.
    """

    steer_noise: float = 0.0
    speed_noise: float = 0.0
    delay_steps: int = 0


@dataclass(frozen=True)
class AlgorithmSpec:
    """Training-algorithm slice; `kind` selects a registered Algorithm
    implementation (e.g. "ppo", "ppo_lagrangian", or a custom kind).

    Attributes:
        kind: which registered algorithm implementation to use.
        ppo: PPO hyperparameters.
        lagrangian: Lagrangian settings (budget, PID gains, ...).
        params: free-form parameters for custom algorithm kinds.
    """

    kind: str = "ppo"
    ppo: dict = field(default_factory=dict)
    lagrangian: dict = field(default_factory=dict)  # budget, pid=(kp,ki,kd), ...
    params: dict = field(default_factory=dict)      # free-form for custom kinds


@dataclass(frozen=True)
class ExperimentSpec:
    """Frozen, content-hashed experiment config assembled by the `>>` DSL.

    Attributes:
        env: the environment/simulator slice.
        obs_dr: observation-side domain randomization.
        encoder: optional feature-encoder stage.
        policy: the policy/network slice.
        action_dr: action-side domain randomization.
        algorithm: the training-algorithm slice.
        total_env_steps: total environment steps to train for.
        eval_every_steps: eval interval in env-steps (0 = final eval only).
        seed: random seed.
        ablation_group: bookkeeping tag grouping related runs.
        variant: bookkeeping tag naming this run within its group.
    """

    env: Optional[EnvSpec] = None
    obs_dr: ObsDRSpec = field(default_factory=ObsDRSpec)
    encoder: EncoderSpec = field(default_factory=EncoderSpec)
    policy: Optional[PolicySpec] = None
    action_dr: ActionDRSpec = field(default_factory=ActionDRSpec)
    algorithm: Optional[AlgorithmSpec] = None
    total_env_steps: int = 5_000_000
    eval_every_steps: int = 0        # 0 = final eval only; N = also every N env-steps
    seed: int = 0
    ablation_group: Optional[str] = None
    variant: Optional[str] = None

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """One-way dump (tuples -> lists, callables -> their name) for the
        hashing + run records."""
        return json.loads(json.dumps(asdict(self), default=_json_default))

    def id(self) -> str:
        """Content-hash identity (sha1 of the config JSON, excluding the
        ablation_group/variant tags) so equal configs share a run dir."""
        # sha1, NOT built-in hash(): identity must be stable across processes.
        # ablation_group/variant are bookkeeping tags, not configuration —
        # the same training config keeps one id however it is tagged.
        payload = {k: v for k, v in self.to_dict().items()
                   if k not in ("ablation_group", "variant")}
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def run_dir(self, root: str = "runs") -> str:
        group = self.ablation_group or "default"
        variant = self.variant or "run"
        return f"{root}/{group}/{variant}-{self.seed}-{self.id()}"

    # ------------------------------------------------------------------
    def available_keys(self) -> tuple[str, ...]:
        """Observation keys the env (+ encoder) makes visible to policies."""
        keys = ["state"]
        if self.env is not None and self.env.modality == "camera":
            keys.append("camera")
        if self.encoder.kind != "none":
            keys.append(self.encoder.out_key)
        return tuple(keys)

    def validate(self) -> "ExperimentSpec":
        """Check the spec is structurally and semantically coherent.

        Returns:
            The spec itself, so ``validate()`` can be chained.

        Raises:
            SpecError: If the env, policy, encoder, obs/action DR, or algorithm
                slice is missing or incoherent.
        """
        if self.env is None:
            raise SpecError("pipeline must start with an Environment stage")
        if self.policy is None:
            raise SpecError("pipeline must include exactly one Policy stage")
        self._validate_environment()
        self._validate_key_routing()
        self._validate_encoder()
        self._validate_obs_dr()
        self._validate_action_dr()
        self._validate_algorithm()
        return self

    def _validate_environment(self) -> None:
        """Check env modality/render coherence and the cost signal.

        Raises:
            SpecError: On a modality/render mismatch, nyx multi-track, or a
                bad cost_fn/cost_budget.
        """
        env = self.env
        match env.modality:
            case "feature" if env.render != "none":
                raise SpecError("feature envs do not render; got render=%r" % env.render)
            case "camera" if env.render not in ("madrona", "nyx"):
                raise SpecError("camera envs need render='madrona'|'nyx'; got %r" % env.render)
        if env.render == "nyx" and len(env.tracks) > 1:
            raise SpecError(
                "heterogeneous tracks are Madrona-only (repo constraint); "
                "render='nyx' with tracks=%r" % (env.tracks,))
        if env.emits_cost:
            if env.cost_fn not in VALID_COST_FNS:
                raise SpecError("cost_fn must be one of %s; got %r" % (VALID_COST_FNS, env.cost_fn))
            if env.cost_budget is None or env.cost_budget <= 0:
                raise SpecError("cost-emitting env needs a positive cost_budget")

    def _validate_key_routing(self) -> None:
        """Check actor/critic obs keys, discrete actions, and camera routing.

        Raises:
            SpecError: On empty/undefined keys, an asymmetric critic missing
                actor keys, a malformed discrete action space, or a
                camera-key/policy mismatch.
        """
        policy = self.policy
        avail = set(self.available_keys())
        a_keys, c_keys = set(policy.actor_keys), set(policy.critic_keys)
        if not policy.actor_keys:
            raise SpecError("policy actor_keys may not be empty")
        if not a_keys <= avail:
            raise SpecError("actor_keys %s not produced by env/encoder (available: %s)"
                            % (sorted(a_keys - avail), sorted(avail)))
        if not c_keys <= avail:
            raise SpecError("critic_keys %s not produced by env/encoder (available: %s)"
                            % (sorted(c_keys - avail), sorted(avail)))
        if not a_keys <= c_keys:
            raise SpecError("asymmetric policies require critic_keys ⊇ actor_keys; "
                            "actor has %s the critic lacks" % sorted(a_keys - c_keys))
        if policy.actions is not None:
            if len(policy.actions) < 2:
                raise SpecError("a discrete action space needs >= 2 actions")
            for a in policy.actions:
                if len(a) != 2 or not all(-1.0 <= float(x) <= 1.0 for x in a):
                    raise SpecError(
                        f"discrete actions are (steer, speed) pairs in [-1, 1]; got {a}")
            if (self.action_dr.steer_noise or self.action_dr.speed_noise
                    or self.action_dr.delay_steps):
                raise SpecError("action DR (noise/delay) operates on continuous "
                                "actions; not compatible with a discrete policy")
        if policy.cnn is None and "camera" in (a_keys | c_keys):
            raise SpecError("a vector policy cannot consume the raw 'camera' key; "
                            "add an encoder stage or use a camera policy")
        if policy.cnn is not None and "camera" not in a_keys:
            raise SpecError("a camera policy's actor must read the 'camera' key")

    def _validate_encoder(self) -> None:
        """Check the frozen-CNN encoder's env/checkpoint/policy requirements.

        Raises:
            SpecError: If a frozen_cnn encoder lacks a camera env, a checkpoint,
                or a downstream vector policy.
        """
        match self.encoder.kind:
            case "frozen_cnn":
                if self.env.modality != "camera":
                    raise SpecError("FrozenCNNToFeatureVector requires an upstream camera env")
                if self.encoder.checkpoint is None:
                    raise SpecError("FrozenCNNToFeatureVector requires a checkpoint path")
                if self.policy.cnn is not None:
                    raise SpecError("FrozenCNNToFeatureVector requires a downstream vector "
                                    "policy (VectorPolicy/AsymmetricVectorPolicy)")

    def _validate_obs_dr(self) -> None:
        """Check observation-DR stages are paired with a camera env.

        Raises:
            SpecError: If appearance/image-aug/camera-jitter DR is used without a
                camera env, or multi-track camera training is requested.
        """
        env, obs_dr = self.env, self.obs_dr
        if obs_dr.appearance and env.modality != "camera":
            raise SpecError("appearance DR recolors the rendered observation; "
                            "it needs a camera env")
        if env.modality == "camera" and len(env.tracks) > 1:
            raise SpecError(
                "multi-track camera training is unsound under the batch "
                "renderer (per-env variant visibility is not implemented in "
                "genesis 1.2.1 — all tracks render superimposed); "
                "multi-track works for feature envs")
        if (obs_dr.image_aug or obs_dr.camera_jitter) and env.modality != "camera":
            raise SpecError("DomainRandomizationCamera requires a camera env")

    def _validate_action_dr(self) -> None:
        """Check action-DR magnitudes are non-negative.

        Raises:
            SpecError: If delay_steps or a noise magnitude is negative.
        """
        if self.action_dr.delay_steps < 0:
            raise SpecError("delay_steps must be >= 0")
        if self.action_dr.steer_noise < 0 or self.action_dr.speed_noise < 0:
            raise SpecError("action noise magnitudes must be >= 0")

    def _validate_algorithm(self) -> None:
        """Check the algorithm matches the env's cost signal and budget.

        Raises:
            SpecError: If the algorithm is missing, PPO-Lagrangian lacks a cost
                env/budget, or the env and algorithm budgets conflict.
        """
        env, algo = self.env, self.algorithm
        if algo is None:
            raise SpecError("algorithm missing: build() must run _infer_algorithm")
        match algo.kind:
            case "ppo" if env.emits_cost:
                warnings.warn(
                    "cost-emitting env trained with plain PPO: the cost stream is "
                    "collected but unconstrained (was this intentional?)",
                    stacklevel=2)
            case "ppo_lagrangian":
                if not env.emits_cost:
                    raise SpecError("PPOLagrangian requires a SafeRL* env that emits a cost signal")
                if not algo.lagrangian.get("budget"):
                    raise SpecError("PPOLagrangian needs a budget (explicit or from the env stage)")
                if (env.cost_budget is not None
                        and algo.lagrangian.get("budget") not in (None, env.cost_budget)):
                    raise SpecError(
                        "conflicting budgets: env.cost_budget=%r vs algorithm.lagrangian"
                        "['budget']=%r — sweep 'env.cost_budget' (ablation.override keeps "
                        "them in sync)" % (env.cost_budget, algo.lagrangian.get("budget")))
