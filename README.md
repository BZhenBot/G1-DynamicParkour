# G1-DynamicParkour

[简体中文](README.zh-CN.md) · [Reproduction](docs/REPRODUCTION.md) · [Evidence and limitations](docs/RESULTS.md) · [Release audit](audit/PRE_PUBLICATION_AUDIT.md)

A simulation research project for **Unitree G1 parkour across moving platforms**, built with mjlab, MuJoCo Warp, PyTorch, and rsl-rl PPO. Learned jump, recovery, and preparation subpolicies are connected by a deterministic skill coordinator. Independent 200 Hz physical logs verify takeoff, flight, landing, and stability.

**Local release candidate; publication is paused for owner review.** The latest accepted baseline is **V2.2**, not the unfinished V3 research branch. Reproduction outcomes are reported in [the test record](audit/TEST_RESULTS.md). Robot assets and full rendered videos are withheld pending exact asset provenance review. No repository or Release URL exists yet.

![System overview](docs/architecture.svg)

| Accepted experiment | Geometry / protocol | Qualified routes |
|---|---|---:|
| V2 baseline | Two 2 cm gaps; fixed start/middle, laterally moving end | 77/100 |
| V2.1 | Same scene; concurrent preparation and causal command smoothing | 80/100 |
| V2.2 | Two **10 cm** gaps; middle 64 × 100 cm; end 60 × 100 cm | **76/100** |

V2.2 uses lateral motion with amplitude **0.05 m**, frequency **0.10 Hz**. Its 24 nonqualified routes include 17 falls, 4 preparation timeouts, 2 other-support failures, and 1 additional 200 Hz physical failure. V2.1 reduced median middle dwell from 12.66 s to 7.13 s on independently qualified routes, while whole-body tilt RMS increased by 13.2%. Different cohorts are not paired; these results do not establish a significant success-rate improvement.

This release preparation reproduced one qualified route in the clean WSL candidate, but the Windows copy using the same seed failed at second-jump premature target contact (0/1). Source and weight hashes matched. **Repeatability remains unresolved**; exit 0 only establishes execution, not task success.

## Research contribution

The contribution is a reproducible engineering study: PPO subpolicy reuse, coordination using current measured platform/robot state, causal command smoothing, physical event checks, and failure-preserving evaluation. PPO, mjlab, G1 assets, and the simulators are upstream work. The switcher is not learned. Actor inputs exclude future ground-truth trajectories; privileged critic inputs are documented in [the results guide](docs/RESULTS.md).

No real-robot deployment, perception-driven parkour, arbitrary terrain generalization, robust mass adaptation, new RL algorithm, or completed V3 capability is claimed. A single-jump policy trained at 15 cm does not establish a 15 cm two-jump result.

## Reproduce

Target: Ubuntu 22.04 / WSL2, Python 3.10, NVIDIA GPU; accepted runs used an RTX 4060 Laptop GPU (8 GB). CPU-only full-policy evaluation is not supported by these entry points. Use a new checkout and environment; see [the full guide](docs/REPRODUCTION.md) for asset restrictions and exact commands.

```bash
uv venv --python /usr/bin/python3 .venv
uv pip install --python .venv/bin/python --index-url https://pypi.org/simple -r configs/requirements.lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
uv pip check --python .venv/bin/python
# Obtain the reviewed weights archive separately (no upload has been made).
.venv/bin/python scripts/prepare_weights.py --archive g1-dynamicparkour-v2.2-weights.zip
.venv/bin/python scripts/reproduce_course_release.py \
  --python "$PWD/.venv/bin/python" \
  --config configs/v22_course_gap010_limited_middle064_jump015_v1.json \
  --episodes 1 --seed 98000 --video-episodes 0 --label public-smoke
```

The wrapper verifies frozen controller hashes and invokes independent physical auditing. **Process exit 0 is not task success**: inspect `course_independent_audit.json` in the new audit experiment. The accepted historical 100-route result is not a guarantee for a new device or run.

## Repository and assets

- `src/`, `scripts/`: tasks, PPO training/evaluation, coordination, physics audits, and evidence capture.
- `configs/`, `data/`: selected configurations and one simulated landing-reset dataset.
- `results/`: representative acceptance/per-route evidence, with machine-specific paths sanitized.
- `release/`: checkpoint roles, dependency graph, asset SHA256, and distribution decisions.
- `docs/`, `audit/`: reproduction, boundaries, failure analysis, file decisions, and tests.

Ten necessary runtime/baseline checkpoints (about 18 MB uncompressed) are organized as a separate Release asset. No venv, cache, vendor model, operational log, or full experiment archive is tracked. Full V2.1/V2.2 videos are locally organized but **excluded from publishable assets** until exact asset rights are resolved.

## License and citation

Original code and authorized original checkpoint parameters: [MIT](LICENSE), subject to [third-party notices](THIRD_PARTY_NOTICES.md). A dependency license does not establish every robot mesh's provenance. Use [CITATION.cff](CITATION.cff); no paper acceptance, DOI, or repository URL is asserted. Please also acknowledge the upstream frameworks.
