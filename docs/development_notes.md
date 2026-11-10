# Development notes

The final question is whether learned humanoid motion can be adapted to static obstacles through inference-time costs. The project developed in several stages:

1. **Reference tracking.** Trained and checked PPO tracking of walking, starting, stopping, and turning. Motion conversion and clip transitions exposed foot-contact and stopping problems.
2. **Goal teacher and data.** Added a PPO + AMP teacher for commanded locomotion and goal arrival. Collected actual observations and actions, with source-aware splits and checks for physical execution.
3. **Generated references.** Compared retrieval, MLP, and diffusion with a shared tracker. Generated motion and tracker errors interacted, especially during braking and turns; a smooth reference alone did not guarantee successful physical execution.
4. **State-action generation and decoder.** Moved to future state-action prediction and a shared learned decoder conditioned on future root motion. The final prediction horizon is 60 control frames (2 seconds). Data collection, decoder repair, and generator training are retained in the code.
5. **Obstacle costs.** Compared MLP output optimization, diffusion guidance, and analytic root correction. Unbounded output changes could produce unrealistic motion, so the final comparison adds feasibility constraints/costs. Analytic root correction remains a separate baseline.

The final comparison uses fixed checkpoints and reused development sources. Cost-guided diffusion succeeds in 33/48 runs, but larger obstacles remain difficult and compute budgets differ. This is a system experiment, not evidence of general architectural superiority.

[Final results](../reports/final_results.md) · [Reproduction](reproduction.md) · [Provenance](provenance.md)

The full local development diary, intermediate settings, failed runs, and model files are preserved outside the Git publication scope. This summary does not replace their experimental evidence. AI tools assisted coding, debugging, and analysis; I reviewed simulation videos and directed the experiments.
