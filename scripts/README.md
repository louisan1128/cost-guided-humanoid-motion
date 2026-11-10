# Scripts

Start with `run_guidance_comparison_v2.py` and the [reproduction instructions](../docs/reproduction.md).

| Script | Purpose |
| --- | --- |
| `train_goal_teacher.py` | Goal-conditioned teacher training |
| `collect_detour_motion_v10.py` | Collect motion variations used during decoder/generator development |
| `repair_detour_decoder_v10.py` | Repair the shared motion-conditioned decoder |
| `build_motion_h60_v10.py` | Build the final 60-frame dataset |
| `train_future_motion_v10.py` | Train the final MLP and diffusion generators |
| `run_guidance_comparison_v2.py` | Physical evaluation with the fixed final models |
| `analyze_guidance_comparison_v2.py` | Aggregate results and diagnose failures |
| `render_guidance_comparison_v2.py` | Render saved evaluation states and diagnostic overlays |
| `run_guidance_heading_demo.py` | Additional heading-change demo; excluded from the main results |
| `capture_saved_episode_isaac.py` | Capture a saved physical episode in Isaac Sim |
| `check_publish.py` | Check publication scope, syntax, links, and local imports without simulation |

Other retained scripts are imports or frozen dependencies of these entry points. Their older filenames do not mean that their original experiments are the final method. Model weights, source rollouts, and external simulator assets are required for training/evaluation; they are not distributed here.
