# Related work

These are design references, not reproduced benchmarks. The project uses a smaller dataset and different evaluation conditions; the results should not be read as comparisons against these papers.

- [MimicKit](https://github.com/xbpeng/MimicKit): the foundation for PPO/AMP and simulator integration. This project adds task environments, data checks, generator/decoder integration, costs, and evaluation tools.
- [BeyondMimic](https://beyondmimic.github.io/): the closest motivation for reusing motion-tracking experience through state-action generation and inference-time task costs. Our final system uses a shared learned root-conditioned decoder and does not reproduce its full system or real-robot results.
- [CLoSD](https://xbpeng.github.io/projects/CLoSD/CLoSD_2025.pdf): an important reference for the earlier generated-reference/physics-tracker stage and actual-state feedback.
- [Diffuser](https://proceedings.mlr.press/v162/janner22a.html): background for trajectory generation with diffusion and inference-time guidance.
- [Diffusion Policy](https://arxiv.org/abs/2303.04137): background for conditional action-sequence generation.

The final research question is whether task costs can adapt learned motion at inference time. The experiment compares raw generation, constrained MLP output optimization, diffusion cost guidance, and analytic root correction under a shared physical executor. It does not establish that diffusion is necessary or universally superior.

[Final results](../reports/final_results.md) · [Development notes](development_notes.md)
