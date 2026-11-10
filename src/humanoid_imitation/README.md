# Source guide

The final evaluation starts at `scripts/run_guidance_comparison_v2.py`:

```text
actual state + history + goal
  → state-action sequence
  → obstacle guidance / constrained output optimization
  → fixed future-root-conditioned decoder
  → physical action and state feedback
```

| Module | Role |
| --- | --- |
| `planning/state_action_sequence.py` | Sequence representation and diffusion sampling |
| `planning/feasible_root_optimization.py` | Feasible root parameterization and MLP output optimization |
| `planning/guidance_only.py` | Cost-guided diffusion |
| `planning/future_conditioned_action.py` | Root features and the shared action decoder |
| `planning/obstacle_geometry.py` | Collision geometry and costs |
| `planning/arrival_success.py` | Arrival and stationary-hold evaluation |
| `envs/goal_amp_env.py` | Goal-conditioned teacher environment |
| `envs/command_amp_env.py` | Command-conditioned AMP environment |
| `data/goal_dataset.py` | Data representation helpers used by retained dependencies |

Other modules are retained imports of the final pipeline. Historical filenames are preserved to avoid changing frozen experiment code during packaging.

[Project overview](../../README.md) · [Reproduction](../../docs/reproduction.md) · [Development notes](../../docs/development_notes.md)
