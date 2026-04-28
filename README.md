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

```bash
pip install -e .
```

The `jobs/*/app/custom/sparsefedmoe` entries are symlinks to `src/sparsefedmoe/`, so NVFlare's
`path:` references resolve both via the installed package and via the job's `custom/` directory.

## Quickstart

```bash
# 1. Partition data (domain-skewed, 2 clients for local dev)
python scripts/prepare_data.py --num_clients 2 --strategy domain --output_dir ./data

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
