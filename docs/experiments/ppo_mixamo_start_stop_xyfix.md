# Generic humanoid PPO tracking: Mixamo start and stop (`xyfix`)

Date: 2026-09-28. Simulator: MimicKit Isaac Lab backend / Isaac Sim. Robot: MimicKit generic humanoid. These are **two separate single-motion PPO trackers**, not the final project's shared tracker.

## Setup

- References: `data/reference_motions/mixamo_start_xyfix.pkl` (89 frames, 2.93 s) and `mixamo_stop_xyfix.pkl` (59 frames, 1.93 s).
- Environment: project copies of MimicKit `deepmimic_humanoid_env.yaml`, with the corresponding motion file and `rand_reset: true` during training. It retains target reference observations, imitation rewards and early termination.
- Agent: MimicKit `data/agents/deepmimic_humanoid_ppo_agent.yaml`, used without hyperparameter changes.
- Each run: seed 0, 1024 environments, 32 rollout steps per iteration, 500 PPO iterations = 16,384,000 environment transitions. Headless Isaac Lab. No prior G1 or generic checkpoint was loaded.
- Output: `outputs/ppo_mixamo_start_xyfix_seed0/` and `outputs/ppo_mixamo_stop_xyfix_seed0/`. Each has `model.pt`, `log.txt`, copied configs and intermediate models at iterations 0, 100, 200, 300, 400, 499.

## Results

| Metric | Start | Stop |
| --- | ---: | ---: |
| Final training-log test return | 34.8 | 24.0 |
| Final training-log test episode length | 37.0 steps | 24.9 steps |
| Full clip, 32 parallel copies from first frame | 32/32 SUCC | 32/32 SUCC |
| Full-clip length | 88 steps | 58 steps |
| Full-clip mean root position error | 0.066 m | 0.047 m |
| Full-clip mean body position error | 0.076 m | 0.053 m |
| Random start phase, seed 123, 64 environments | 64/64 SUCC | 64/64 SUCC |
| Random-phase mean body position error | 0.048 m | 0.038 m |

`SUCC` is MimicKit's end-of-motion termination flag, checked with the policy controlling the physical robot. The full-clip copies share the same initial condition and deterministic policy, so 32/32 is a repeatability check, **not 32 independent disturbance trials**. Random-phase evaluation samples states from the *same training clip*; it does not measure unseen-motion generalization. Some random-phase episodes begin near the end and last only one step. The training-log test length is shorter because training config uses random reset phase; it should not be compared directly to the full-clip length.

The earlier default PD position-target controller failed before the end on these clips *and on MimicKit's original humanoid walk*. PPO adds feedback from actual robot and reference observations; it succeeded in these nominal simulations. This does not establish real-robot feasibility, robust recovery or a multi-motion shared tracker. The remaining work is to compare visual quality, quantify contact/slip from physical rollout, and train/evaluate one shared tracker with both motions and other needed references.

## Physical speed profile (one deterministic full-clip rollout)

Measured with `scripts/measure_tracker_speed.py`, seed 0, one environment, Isaac Lab control step 1/30 s. Values are the magnitude of world-frame root XY velocity, averaged over the stated time window; the reference column comes from MimicKit's interpolated reference velocity. This is a nominal tracking diagnostic, not a transition test.

| Motion/window | Physical speed | Reference speed |
| --- | ---: | ---: |
| Start, first 0.5 s | 0.213 m/s | 0.104 m/s |
| Start, last 0.43 s | 0.961 m/s | 1.069 m/s |
| Stop, first 0.5 s | 0.749 m/s | 0.867 m/s |
| Stop, last 0.43 s | 0.100 m/s | 0.109 m/s |

Both rollouts reached `SUCC` (start 88 steps, stop 58). The available `humanoid_walk.pkl` reference averages about 1.27 m/s across its clip, but there is no trained generic-humanoid walk PPO checkpoint in this project yet, so its **physical** speed is unknown. Based on these samples, directly switching start→walk or walk→stop may require speed matching; joint pose, foot phase, heading and body velocity must also be checked. The first/last window averages do not equal exact boundary velocities.

To repeat the measurements:

```bash
/home/an/miniconda3/envs/issaclab/bin/python scripts/measure_tracker_speed.py \
  --env-config configs/envs/deepmimic_mixamo_start_xyfix_eval.yaml \
  --model-file outputs/ppo_mixamo_start_xyfix_seed0/model.pt

/home/an/miniconda3/envs/issaclab/bin/python scripts/measure_tracker_speed.py \
  --env-config configs/envs/deepmimic_mixamo_stop_xyfix_eval.yaml \
  --model-file outputs/ppo_mixamo_stop_xyfix_seed0/model.pt
```

## Reproduce evaluation

Run from `/home/an/Desktop/project_alpha`:

```bash
/home/an/miniconda3/envs/issaclab/bin/python scripts/evaluate_mixamo_tracker.py \
  --env-config configs/envs/deepmimic_mixamo_start_xyfix_eval.yaml \
  --model-file outputs/ppo_mixamo_start_xyfix_seed0/model.pt \
  --episodes 32 --seed 0

/home/an/miniconda3/envs/issaclab/bin/python scripts/evaluate_mixamo_tracker.py \
  --env-config configs/envs/deepmimic_mixamo_stop_xyfix_eval.yaml \
  --model-file outputs/ppo_mixamo_stop_xyfix_seed0/model.pt \
  --episodes 32 --seed 0
```

For a random start phase, use the corresponding training env config in place of the `_eval.yaml` config and `--episodes 64 --seed 123`.

## Visualize trained physical tracking

Run from `/home/an/Desktop/MimicKit-main` and close the window or press Ctrl+C to stop. These commands load PPO and simulate a physical humanoid; they are different from `view_motion` forced-pose playback.

```bash
/home/an/miniconda3/envs/issaclab/bin/python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --mode test --num_envs 1 --visualize true --test_episodes 1000000 \
  --env_config /home/an/Desktop/project_alpha/configs/envs/deepmimic_mixamo_start_xyfix_eval.yaml \
  --model_file /home/an/Desktop/project_alpha/outputs/ppo_mixamo_start_xyfix_seed0/model.pt \
  --out_dir /tmp/ppo_mixamo_start_visual

/home/an/miniconda3/envs/issaclab/bin/python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --mode test --num_envs 1 --visualize true --test_episodes 1000000 \
  --env_config /home/an/Desktop/project_alpha/configs/envs/deepmimic_mixamo_stop_xyfix_eval.yaml \
  --model_file /home/an/Desktop/project_alpha/outputs/ppo_mixamo_stop_xyfix_seed0/model.pt \
  --out_dir /tmp/ppo_mixamo_stop_visual
```

## Training commands

The runs above used MimicKit's PPO agent config unchanged. The commands below show the exact settings; choose a **new** output directory if rerunning, because MimicKit writes `model.pt` there.

```bash
cd /home/an/Desktop/MimicKit-main
/home/an/miniconda3/envs/issaclab/bin/python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --env_config /home/an/Desktop/project_alpha/configs/envs/deepmimic_mixamo_start_xyfix.yaml \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml \
  --num_envs 1024 --visualize false --max_samples 16384000 \
  --save_int_models true --rand_seed 0 \
  --out_dir /home/an/Desktop/project_alpha/outputs/NEW_START_RUN
```

Replace `start` with `stop` in `--env_config` and use a different new `--out_dir` for the second run. `--model_file` was omitted: both models began from random initialization.
