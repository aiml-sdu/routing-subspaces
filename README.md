# Routing Subspaces: Auditing Evaluation-to-Deployment Mismatch in Fine-Tuned Language Models

Phongsakon Mark Konrad<sup>1,*</sup>, Toygar Tanyel<sup>2</sup>, Serkan Ayvaz<sup>1</sup> · NeurIPS 2026 (poster)

<sup>1</sup>Centre for Industrial Software, University of Southern Denmark · <sup>2</sup>Promake · <sup>*</sup>Now at University of Tübingen

[Project page](https://aiml-sdu.github.io/routing-subspaces-page/) · [arXiv](https://arxiv.org/abs/2607.20436)

Fine-tuning can install a gap between how a model behaves on evaluation-style and on deployment-style prompts. The audit fits a paired activation contrast at a path-patching-informed mid-depth attention window, then modifies that coordinate on held-out prompts. Across twelve model and behavior settings it reduces the gap by more than 0.12 in ten and closes it (below 0.06) in four. It is a non-adaptive, white-box diagnostic for controlled fine-tuned checkpoints, not a training-time defense or a guarantee of deployment safety.

## Layout

| Path | Contents |
|---|---|
| `src/scripts/` | Experiment scripts and run orchestration |
| `src/eval_awareness/`, `src/data/`, `src/configs/` | Prompts, held-out splits, training configuration |
| `src/results/` | Stored results ([index](src/results/README.md)) |
| `scripts/` | `make_tables.py` and `make_figures.py` rebuild the appendix tables and figures from the stored results (`numpy`, `matplotlib`) |
| `tools/routing_intervention/` | Packaged intervention pipeline with tests |

Model weights are not included, so the experiment scripts in `src/scripts/` need your own checkpoints (see the paths read from `CHECKPOINT_ROOT`, `VENV`, `TRAINER_ROOT`). The environment is pinned in `src/pyproject.toml` and `src/uv.lock`.

Runs on fresh checkpoints carry manifests with SHA-256 hashes of inputs and outputs. Hashed files are kept byte-identical, so manifests list cluster paths, and two orchestration scripts keep the cluster's directory names.

## License

Code: MIT ([LICENSE](LICENSE)). Authored data and stored results: CC BY 4.0. Third-party data keeps its own license ([THIRD_PARTY.md](THIRD_PARTY.md)).

## Citation

```bibtex
@inproceedings{konrad2026routing,
    title={Routing Subspaces: Auditing Evaluation-to-Deployment Mismatch in Fine-Tuned Language Models},
    author={Konrad, Phongsakon Mark and Tanyel, Toygar and Ayvaz, Serkan},
    booktitle={Advances in Neural Information Processing Systems},
    year={2026},
    eprint={2607.20436},
    archivePrefix={arXiv},
    url={https://arxiv.org/abs/2607.20436}
}
```
