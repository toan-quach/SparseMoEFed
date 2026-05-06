# SparseFedMoE

Communication-efficient federated fine-tuning of Mixture-of-Experts LLMs via activation-aware sparse
communication, implemented on NVIDIA NVFlare. Architecture spec: `../SparseFedMoE_Architecture.md`.

## Layout

```
src/sparsefedmoe/
├── client/   # Forward-hook activation tracker, sparse encoder/decoder filters, OLMoE SFT trainer
├── server/   # Frequency-weighted controller, global floor monitor, router alignment
└── common/   # Per-expert compressor, model-introspection helpers, clusterer, config dataclass
jobs/         # NVFlare job definitions (sparsefedmoe_olmoe, fedavg_olmoe baseline)
scripts/      # Data partitioning + simulator launcher
tests/        # Unit tests for each component
```

## Install

Requires Python 3.11 (NVFlare 2.6 supports 3.9–3.12, but the project pins 3.11 for reproducibility).

```bash
uv venv --python 3.11
uv pip install -e .
```

The editable install places `sparsefedmoe` on the venv's import path, which is how NVFlare's
class-loader resolves `path:` references like `sparsefedmoe.server.freq_weighted_controller.FreqWeightedFedAvg`
in the job configs. Each job's `app/custom/olmoe_sft_trainer.py` is a small shim that the
in-process client API runner executes, which then imports the trainer's `main()` from the package.

## Quickstart

```bash
# 1. Partition data (domain-skewed, 2 clients for local dev)
python scripts/prepare_data.py --num_clients 7 --strategy domain --output_dir ./data

# 2. Smoke test the sparse job with 2 simulated clients
bash scripts/run_simulator.sh jobs/sparsefedmoe_olmoe -n 2

# 3. Compare against FedAvg-Full (all flags disabled, same trainer)
bash scripts/run_simulator.sh jobs/fedavg_olmoe -n 2
```

## Components (arch doc mapping)

| §   | Module                                            |
|-----|---------------------------------------------------|
| 3.1 | `client/activation_tracker.py`                    |
| 3.2 | `client/sparse_moe_encoder.py` + `common/expert_compressor.py` (EF21 buffer) |
| 3.3 | `server/freq_weighted_controller.py`              |
| 3.4 | `server/router_alignment.py`                      |
| 3.5 | `server/global_floor_monitor.py`                  |
| 3.6 | `common/client_clusterer.py` (optional, disabled in default jobs) |

## Tests

```bash
pytest -v
```
