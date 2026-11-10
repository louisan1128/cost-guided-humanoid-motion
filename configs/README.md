# Configuration

The final evaluation uses `releases/guidance_comparison_v2_manifest.json` and `reports/assets/guidance_comparison_v2/scenes.json`.

- `releases/guidance_comparison_v2_manifest.json`: frozen model, data, and code references
- `releases/guidance_comparison_v2_results.json`: original result hashes and evaluation metadata
- `assets/guidance_pole.usd`: obstacle asset template
- Other retained files: runtime or frozen dependencies of the final code

Manifests describe the full local experiment, including excluded checkpoints and historical artifacts. They are provenance records, not a list of bundled files. Use the [reproduction guide](../docs/reproduction.md) before running an evaluation.
