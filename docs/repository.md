# Repository guide

| Location | Contents |
| --- | --- |
| `src/humanoid_imitation/` | Required model, environment, data, and control modules |
| `scripts/` | Final entry points and their imported helpers |
| `configs/` | Required settings and provenance manifests |
| `reports/final_results.md` | Final comparison and failures |
| `reports/assets/` | Selected results, figures, and videos |
| `output/pdf/` | Two detailed Korean reports |
| `docs/` | Reproduction, design background, and provenance |
| `tests/` | Six core test files for the final implementation |

The publication allowlist is `publication_manifest.json`. Local historical files remain on disk but are ignored by Git. Add a new public file to the manifest and the matching allowlist in `.gitignore`; run `python3 scripts/check_publish.py` before staging it.

Some retained helpers have historical names because the final scripts import them. We kept their paths and contents rather than rewriting frozen experiment code during packaging. Full motion data, checkpoints, and external MimicKit/Isaac Lab are not included. See [reproduction](reproduction.md).
