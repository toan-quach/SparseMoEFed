# SparseFedMoE — Architecture Design Document

**Communication-Efficient Federated Fine-Tuning of MoE LLMs via Activation-Aware Sparse Communication**

Version 1.0 · April 2026

---

## 1. Overview

SparseFedMoE is a communication protocol for federated fine-tuning of Mixture-of-Experts (MoE) large language models. It exploits a fundamental property of MoE architectures under non-IID federated settings: each client's data activates only a subset of experts, creating exploitable sparsity in gradient updates.

The system sits in the communication layer between local training and server aggregation. It is not a training algorithm — it is a transport-level optimization that is orthogonal to and composable with existing federated MoE methods such as FLEx, HFedMoE, and FLEX-MoE.

### 1.1 Design Principles

- **Transparency.** Compression happens entirely in the communication layer. The local training loop and model architecture are untouched. Any MoE fine-tuning strategy (full FFN, LoRA, adapter) works with SparseFedMoE without modification.
- **Composability.** SparseFedMoE composes with expert selection methods (HFedMoE), personalization schemes (FLEx grafting), and load-balancing algorithms (FLEX-MoE). It answers a different question — not *which* experts to train, but *how* to efficiently transmit the results.
- **Adaptivity.** Compression levels are not fixed. They adjust automatically each round based on observed per-expert activation patterns, responding to evolving data distributions and router behavior throughout training.
- **Self-consistency.** The same activation frequency statistics that drive gradient compression also govern aggregation weighting and router averaging, ensuring the entire pipeline uses a unified importance signal.

### 1.2 Target Models

| Model | Total Params | Active Params | MoE Layers | Experts/Layer | Top-k | Expert FFN Size |
|-------|-------------|--------------|------------|---------------|-------|-----------------|
| OLMoE-1B-7B (primary) | 6.9B | 1.3B | 16 | 64 | 8 | ~18M |
| Mixtral-8x7B (scaling) | 46.7B | 12.9B | 32 | 8 | 2 | ~180M |

The communication savings scale with expert count. Fine-grained MoE models (OLMoE with 64 experts, DeepSeek-V3 with 256) benefit the most.

---

## 2. System Architecture

### 2.1 High-Level Data Flow

```
┌───────────────────────────────────────────────────────────────┐
│                        FL Server                              │
│                                                               │
│  ┌──────────────────┐    ┌──────────────────────────────┐     │
│  │  Global Floor    │    │  Frequency-Weighted          │     │
│  │  Monitor         │───▶│  Aggregator                  │     │
│  │  (§3.5)          │    │  (§3.3) + Router Alignment   │     │
│  └──────────────────┘    │  Weighting (§3.4)            │     │
│                          └──────────┬───────────────────┘     │
│                                     │                         │
│              ┌──────────────────────┼──────────────────┐      │
│              │ broadcast            │ broadcast        │      │
│              ▼                      ▼                  ▼      │
│  ┌───────────────┐     ┌───────────────┐    ┌──────────────┐  │
│  │   Client A    │     │   Client B    │    │   Client C   │  │
│  │               │     │               │    │              │  │
│  │ ┌───────────┐ │     │ ┌───────────┐ │    │ ┌──────────┐ │  │
│  │ │ Local     │ │     │ │ Local     │ │    │ │ Local    │ │  │
│  │ │ Training  │ │     │ │ Training  │ │    │ │ Training │ │  │
│  │ └─────┬─────┘ │     │ └─────┬─────┘ │    │ └────┬─────┘ │  │
│  │       │       │     │       │       │    │      │       │  │
│  │ ┌─────▼─────┐ │     │ ┌─────▼─────┐ │    │ ┌────▼─────┐ │  │
│  │ │ Activation│ │     │ │ Activation│ │    │ │Activation│ │  │
│  │ │ Tracker   │ │     │ │ Tracker   │ │    │ │ Tracker  │ │  │
│  │ │ (§3.1)    │ │     │ │ (§3.1)    │ │    │ │ (§3.1)   │ │  │
│  │ └─────┬─────┘ │     │ └─────┬─────┘ │    │ └────┬─────┘ │  │
│  │       │       │     │       │       │    │      │       │  │
│  │ ┌─────▼─────┐ │     │ ┌─────▼─────┐ │    │ ┌────▼─────┐ │  │
│  │ │ Sparse    │ │     │ │ Sparse    │ │    │ │ Sparse   │ │  │
│  │ │ Encoder   │ │     │ │ Encoder   │ │    │ │ Encoder  │ │  │
│  │ │ (§3.2)    │ │     │ │ (§3.2)    │ │    │ │ (§3.2)   │ │  │
│  │ └─────┬─────┘ │     │ └─────┬─────┘ │    │ └────┬─────┘ │  │
│  │       │upload │     │       │upload │    │      │upload │  │
│  └───────┼───────┘     └───────┼───────┘    └──────┼───────┘  │
│          └─────────────────────┼─────────────────────┘        │
│                                ▼                              │
│                     Sparse Decoder → Aggregator               │
└───────────────────────────────────────────────────────────────┘
```

### 2.2 Round Protocol

A single SparseFedMoE round proceeds as follows:

1. **Broadcast.** Server sends global model weights to all participating clients. Optionally includes per-expert floor tier assignments from the Global Floor Monitor.
2. **Local training.** Each client runs E local epochs of fine-tuning on its private data. The Activation Tracker hooks into the MoE router to record per-expert token routing frequencies.
3. **Sparse encoding.** The client computes `Δw = w_local − w_global` and passes it through the Sparse Encoder, which applies per-expert tier-based compression (FP16 / INT8 / skip) based on activation frequencies and any server-mandated floor tiers.
4. **Upload.** Client transmits: (a) compressed expert deltas, (b) full-precision shared parameter deltas (attention, norms, router weights), and (c) activation frequency metadata vector `A_i ∈ ℝ^{L×E}`.
5. **Decode + aggregate.** Server decompresses expert deltas, applies frequency-weighted aggregation for expert parameters and alignment-weighted aggregation for router parameters, and produces the updated global model.
6. **Floor monitoring.** Server inspects global activation statistics to identify underutilized experts and prepares floor assignments for the next round.

---

## 3. Core Components

### 3.1 Component 1: Activation Tracker

**Purpose.** Record per-expert activation frequencies during local training to produce a compact importance signal that drives all downstream decisions.

**Mechanism.** A PyTorch forward hook is registered on each `SparseMoeBlock` (or equivalent MoE routing layer). On every forward pass, the hook intercepts the router's top-k expert indices and increments per-expert counters using `torch.bincount`.

**Data structure:**

```
freq[layer_l][expert_e] = (tokens routed to expert e in layer l)
                          / (total tokens processed by layer l)
```

For a model with L MoE layers and E experts per layer, this produces a frequency matrix `A_i ∈ ℝ^{L×E}`. Under top-k routing, `Σ_e freq[l][e] = k` for each layer.

**Output.** The activation profile `A_i` is transmitted alongside gradient updates as metadata. For OLMoE (16 layers × 64 experts), this is 1,024 float32 values — approximately 4 KB, negligible relative to model parameters.

**Overhead.** A single `torch.bincount` per layer per forward pass. Measured at less than 0.1% of training wall-clock time.

### 3.2 Component 2: Adaptive Per-Expert Sparse Encoder

**Purpose.** Compress gradient updates at expert granularity, applying heavier compression to less-activated experts and preserving fidelity for heavily-used ones.

**Tier assignment.** Each expert's activation frequency determines its compression tier:

| Tier | Condition | Format | Bytes/Element | Rationale |
|------|-----------|--------|---------------|-----------|
| HIGH | `freq ≥ τ_high` | FP16 | 2 | High-confidence updates; client processed many tokens through this expert |
| MED | `τ_skip ≤ freq < τ_high` | INT8 symmetric quantization | 1 | Moderate confidence; aggressive compression acceptable |
| SKIP | `freq < τ_skip` | Not transmitted | 0 | Expert barely activated; gradient is dominated by noise |

Default thresholds: `τ_high = 0.05`, `τ_skip = 0.005` (tunable per deployment).

**Floor override.** The server may mandate a minimum tier for specific experts via the Global Floor Monitor (§3.5). If an expert is marked as floor-protected, its tier is promoted to at least INT8 regardless of local activation frequency. This prevents expert collapse under severe non-IID conditions.

**Shared parameters.** Attention layers, layer norms, embedding layers, and router/gate weights are always transmitted at full precision (FP32 or FP16 depending on training dtype). These parameters are activated by every token and do not benefit from frequency-based sparsification.

**Error feedback.** When an expert is skipped, the gradient residual is accumulated in a client-side buffer. If the accumulated residual's L2 norm exceeds a configurable threshold, the buffer contents are transmitted as a correction term in the next round. This mechanism, derived from the EF21 framework, ensures that skipped gradients are not permanently lost — only deferred.

**Encode algorithm (pseudocode):**

```
function SparseMoEEncode(delta_w, activation_profile, config, floor_tiers):
    result = {}

    # Shared parameters: always full precision
    result["shared"] = delta_w.shared

    # Per-expert encoding
    for layer_l in moe_layers:
        for expert_e in range(E):
            freq = activation_profile[l][e]
            floor = floor_tiers.get((l, e), None)

            tier = assign_tier(freq, config.tau_skip, config.tau_high)
            if floor and tier_rank(floor) > tier_rank(tier):
                tier = floor  # server-mandated minimum

            if tier == "FP16":
                result[(l, e)] = to_fp16(delta_w.expert[l][e])
            elif tier == "INT8":
                result[(l, e)] = symmetric_int8_quantize(delta_w.expert[l][e])
            else:  # SKIP
                error_buffer[(l, e)] += delta_w.expert[l][e]
                if norm(error_buffer[(l, e)]) > config.ef_threshold:
                    result[(l, e)] = symmetric_int8_quantize(error_buffer[(l, e)])
                    error_buffer[(l, e)] = 0

    result["metadata"] = activation_profile  # ~4KB
    return result
```

### 3.3 Component 3: Frequency-Weighted Aggregation

**Purpose.** Aggregate expert updates using activation frequency as an importance weight rather than the conventional dataset-size weighting.

**Intuition.** A client that routed 40% of its tokens through expert e has a much more reliable gradient estimate for that expert than a client that routed 1%. Activation frequency is a natural importance weight — it directly measures how much data informed each expert's gradient.

**Aggregation rule for expert parameters:**

For each expert e, let `S_e = {i : client i transmitted an update for expert e}`. The server computes:

```
w^{t+1}(expert_e) = w^t(expert_e)
    + Σ_{i ∈ S_e} [ α_i(e) / Σ_{j ∈ S_e} α_j(e) ] × decode(Δw_i(expert_e))
```

where `α_i(e)` is client i's activation frequency for expert e.

If `S_e = ∅` (no client transmitted updates for expert e), the expert retains its current global weights. This preserves pretrained knowledge for experts not relevant to any client's current data distribution.

**Aggregation rule for shared parameters:** Standard FedAvg weighted by dataset size:

```
w^{t+1}(shared) = Σ_i [ |D_i| / Σ_j |D_j| ] × Δw_i(shared)
```

**Variance reduction property.** Under uniform FedAvg, all clients contribute equally to each expert regardless of how many tokens they processed through it. Frequency weighting acts as importance sampling — clients with higher activation frequency (lower per-gradient variance) receive higher weight, reducing the variance of the aggregated update. This is formalized via a variance decomposition showing `Var(freq-weighted) ≤ Var(uniform)` whenever activation frequencies differ across clients.

### 3.4 Component 4: Alignment-Weighted Router Aggregation

**Purpose.** Prevent router corruption by ensuring that clients whose routing preferences align with the federation's global distribution have proportionally more influence on the global gating policy.

**Problem addressed.** SparseFedMoE handles expert FFN weights correctly via frequency-weighted aggregation, but naive FedAvg on the router can corrupt the mechanism that decides which experts to use. A client with unusual routing preferences (e.g., exclusively activating experts that no other client uses) should not have equal influence on the global router.

**Mechanism.** The server reuses the per-expert activation frequencies already collected for compression decisions:

1. **Active expert sets.** For each client c, define `S_c = {e : freq_c(e) > τ_active}`.

2. **Expert popularity.** For each expert e, compute the fraction of clients that actively use it:
   ```
   popularity(e) = |{c : e ∈ S_c}| / N
   ```

3. **Client alignment score.** Sum the popularity of a client's active experts:
   ```
   alignment(c) = Σ_{e ∈ S_c} popularity(e)
   ```
   Clients whose active experts are widely shared across the federation get higher alignment scores.

4. **Router aggregation weights.** Normalize alignment scores:
   ```
   weight(c) = alignment(c) / Σ_{c'} alignment(c')
   ```
   Apply these weights when averaging router parameters instead of uniform or dataset-size weighting.

**Cost.** Four lines of server-side code on top of data already computed. No extra client-server communication. No extra client computation.

**Self-consistency.** This makes the router aggregation consistent with the expert communication scheme: clients whose data aligns with the global distribution have proportionally more influence on both which expert weights get transmitted and what the global routing policy looks like.

### 3.5 Component 5: Global Floor Monitor

**Purpose.** Prevent expert collapse — a failure mode where rarely-activated experts receive too few gradient updates across the federation and effectively freeze at their initialization weights.

**Failure mode.** Under severe non-IID conditions, some experts may be consistently below the skip threshold on every client. After many rounds, these experts accumulate significant drift from the training distribution. Training metrics won't catch this because the training data never activates those experts — the failure surfaces only at deployment on out-of-distribution inputs.

**Mechanism.** After each round, the server computes global per-expert activation frequency by summing across all clients:

```
global_freq(e) = Σ_c freq_c(e)
mean_freq = mean over all experts of global_freq(e)

for each expert e:
    if global_freq(e) < γ × mean_freq:   # e.g., γ = 0.10
        floor_tier(e) = "INT8"
```

The floor tier assignments are broadcast to clients at the start of the next round as a small vector (E values per layer). The Sparse Encoder (§3.2) respects these floors by promoting any expert below the floor to at least INT8 compression.

**Why this approach over alternatives.** Several options were considered:

- *Minimum communication floor (always INT8, no skip):* Too blunt — wastes bandwidth on experts that are locally rare but globally well-covered.
- *Periodic full sync rounds:* Expensive and doesn't target the problem surgically.
- *Router entropy regularization:* Changes training dynamics and fights against the non-IID specialization that SparseFedMoE exploits.
- *Expert dropout at aggregation (freeze underutilized experts):* Prevents corruption but doesn't solve under-training.

The frequency-adaptive global floor is surgical — it only applies the floor to genuinely underutilized experts across the entire federation. It fits naturally into the existing framework since the server already receives activation frequency vectors from all clients.

### 3.6 Component 6: Expert-Affinity Client Clustering (Optional)

**Purpose.** Group clients with similar expert activation patterns for hierarchical aggregation, reducing redundant WAN transmissions.

**Mechanism.** After round 1, the server computes pairwise cosine similarity of client activation profiles. Clients are grouped into K clusters via standard clustering (k-means or agglomerative). Two aggregation modes:

- **Cluster-aware server aggregation (simple).** Clients in the same cluster contribute to joint expert updates with intra-cluster averaging before inter-cluster aggregation. Reduces effective variance.
- **Hierarchical P2P aggregation (advanced).** A cluster head performs intra-cluster aggregation locally; only the aggregated result is sent to the server. Reduces WAN hops from N to K.

Reclustering occurs every R rounds (default R = 10) to adapt to evolving activation patterns.

This component is optional and is positioned as an add-on for large-scale deployments (N > 10 clients).

---

## 4. Communication Analysis

### 4.1 Per-Round Cost

**Baseline FedAvg (dense):**

```
Cost_FedAvg = N × (|w_shared| + E × |w_expert|)    per round
```

**SparseFedMoE:**

```
Cost_Sparse = N × (|w_shared| + k̄ × r̄ × |w_expert| + metadata)
```

where `k̄` is the average number of experts passing the skip threshold per client, and `r̄` is the average compression ratio (FP16 = 0.5, INT8 = 0.25 relative to FP32).

**Savings factor:**

```
savings = 1 − (k̄ × r̄) / E
```

### 4.2 Projected Savings

| Model | E | Top-k | Avg Experts Passing (k̄) | Avg Compression (r̄) | Savings |
|-------|---|-------|--------------------------|---------------------|---------|
| OLMoE-1B-7B | 64 | 8 | ~10 | 0.50 | **92.2%** |
| Mixtral-8x7B | 8 | 2 | ~3 | 0.70 | **73.8%** |

Savings scale with expert count — models with more experts (DeepSeek-V3: 256 experts) will see even greater reductions.

### 4.3 Metadata Overhead

Per client per round: `L × E × 4 bytes` (one float32 per expert per layer).

- OLMoE: 16 × 64 × 4 = 4,096 bytes ≈ 4 KB
- Mixtral: 32 × 8 × 4 = 1,024 bytes ≈ 1 KB

Negligible relative to model parameter transmission.

---

## 5. Convergence Properties

### 5.1 Assumptions

SparseFedMoE's convergence analysis extends the standard federated optimization framework with two MoE-specific terms:

- **(A1) L-smoothness.** Each client's local loss F_i is L-smooth.
- **(A2) Bounded stochastic variance.** `E[‖∇F_i(w) − g_i(w)‖²] ≤ σ²`
- **(A3) Bounded heterogeneity.** `(1/N) Σ_i ‖∇F_i(w) − ∇F(w)‖² ≤ δ²`
- **(A4) Expert sparsification error.** At most k of E experts pass the skip threshold per client. Skipped experts contribute an error term `ε_sparse` bounded by the product of the skip threshold τ and the maximum expert gradient norm.
- **(A5) Compression error.** For compressor C_e with ratio r_e: `E[‖C_e(x) − x‖²] ≤ (1 − r_e)‖x‖²` (standard biased compressor).

### 5.2 Convergence Rate (Informal)

After T rounds with learning rate η:

```
(1/T) Σ_{t=0}^{T−1} E[‖∇F(w^t)‖²]

  ≤  O(1/√T)                         ← standard FL convergence rate
   + O(η²E²σ²)                       ← local SGD variance (same as FedAvg)
   + O(η²E²δ²)                       ← data heterogeneity (same as FedAvg)
   + O(ε_sparse)                      ← expert skipping error (NEW)
   + O((1−r̄) × η²E²σ²)              ← compression error (NEW)
```

The expert skipping error `ε_sparse` is bounded because: (1) experts with low activation frequency tend to have small gradients for the client's data, and (2) error feedback ensures skipped residuals are eventually transmitted. With error feedback, the compression error term vanishes asymptotically (via EF21).

### 5.3 Frequency Weighting Reduces Variance

The variance of frequency-weighted aggregation is strictly lower than uniform FedAvg for expert parameters when activation frequencies differ across clients. This follows from importance sampling: `α_i(e)` acts as a natural proposal distribution that assigns higher weight to clients with lower per-gradient variance for each expert.

---

## 6. Implementation on NVFlare

### 6.1 Component Mapping

| SparseFedMoE Component | NVFlare Abstraction |
|------------------------|---------------------|
| Activation Tracker | Integrated into Client API training script (forward hooks) |
| Sparse Encoder/Decoder | `task_result_filters` (NVFlare Filter mechanism) |
| Frequency-Weighted Aggregator | Custom `ModelController` (`FreqWeightedFedAvg`) |
| Router Alignment Weighting | Integrated into `FreqWeightedFedAvg` controller |
| Global Floor Monitor | Server-side logic within `ModelController` |
| Client Clustering | Optional server-side module |

### 6.2 Key Implementation Files

```
sparsefedmoe/
├── client/
│   ├── activation_tracker.py       # Forward hooks for MoE router
│   ├── sparse_moe_encoder.py       # NVFlare Filter: compress on upload
│   └── sparse_moe_decoder.py       # NVFlare Filter: decompress on receive
├── server/
│   ├── freq_weighted_controller.py # Custom ModelController
│   ├── router_alignment.py         # Alignment-weighted router aggregation
│   └── global_floor_monitor.py     # Under-utilization detection
├── common/
│   ├── compression.py              # FP16/INT8 quantization utilities
│   └── config.py                   # Threshold and hyperparameter definitions
└── jobs/
    └── sparsefedmoe_olmoe/
        ├── meta.json
        └── app/
            ├── config/
            │   ├── config_fed_server.json
            │   └── config_fed_client.json
            └── custom/
                └── (symlink to sparsefedmoe/)
```

### 6.3 Execution Environment

- **Simulation:** NVFlare multi-process simulator on NVIDIA H20 HGX (8×GPU cluster via SLURM on ParaCloud)
- **Conda environment:** `gfq_env`
- **Model path:** Full absolute path to local snapshot directory (cluster has no internet on compute nodes; model transferred via SCP)
- **Submission:** SLURM batch scripts

---

## 7. Experiment Design Summary

### 7.1 Primary Evaluation Matrix

| Dimension | Values |
|-----------|--------|
| Models | OLMoE-1B-7B, Mixtral-8x7B |
| Clients | 5, 10, 20 |
| Non-IID strategy | Domain skew (primary), Dirichlet α ∈ {0.1, 0.5, 1.0} |
| Datasets | dolly-15k, Alpaca-GPT4, Finance-Alpaca, MedAlpaca |
| Fine-tuning mode | Full expert FFN (primary), LoRA (secondary) |

### 7.2 Baselines

| Baseline | Description |
|----------|-------------|
| FedAvg-Full | Naive dense transmission of all parameters |
| FedAvg-LoRA | LoRA adapters without MoE awareness |
| FLEx | Freeze experts, aggregate only shared parameters |
| HFedMoE | Resource-aware expert selection with importance scoring |
| Top-k Sparsification | Generic gradient sparsification (k = 10%) |
| FedSparse | Importance-based sparse updates |
| SharedOnlyCompressor | Transmit only shared params (no expert updates) |
| RandomSkipCompressor | Skip random experts (no frequency awareness) |

The last two baselines (SharedOnly, RandomSkip) are critical controls. SharedOnly isolates the value of transmitting any expert updates at all. RandomSkip isolates the value of frequency-aware selection over naive random dropping.

### 7.3 Metrics

- **Communication:** Total bytes transmitted, bytes to reach target accuracy
- **Quality:** Task accuracy/F1, perplexity, MMLU (knowledge preservation)
- **Efficiency:** Wall-clock time under simulated WAN (100 Mbps, 1 Gbps)
- **System overhead:** Activation tracker latency, encoder/decoder latency, aggregation overhead

### 7.4 Key Ablations

1. **Threshold sensitivity.** Sweep `τ_skip` to plot the Pareto curve of accuracy vs. communication reduction.
2. **Compression strategy.** Compare no compression / uniform INT8 / uniform FP16 / adaptive (ours) at matched communication budgets.
3. **Aggregation weighting.** Compare uniform FedAvg / dataset-size-weighted / frequency-weighted (ours).
4. **Global floor effect.** With vs. without the floor monitor — demonstrate expert collapse on severely skewed data without the floor, and its prevention with.
5. **Error feedback.** With vs. without EF — show accuracy recovery from aggressive compression.
6. **Client clustering.** With vs. without, varying K.

---

## 8. Positioning Relative to Prior Work

SparseFedMoE occupies a distinct niche in the FL + MoE landscape. Existing work focuses on the *algorithmic* question of expert selection and assignment. SparseFedMoE addresses the *systems* question of communication efficiency.

| Method | Focus | Relationship to SparseFedMoE |
|--------|-------|------------------------------|
| FLEx | Personalization via expert grafting | Complementary — grafting + sparse communication compose |
| HFedMoE | Resource-aware expert selection | Complementary — selection decides what to train; we decide how to transmit |
| FLEX-MoE | Load-balanced expert assignment | Orthogonal — they assign experts; we compress transmissions |
| FedMoE-DA | P2P selective expert sync | Architecturally different (P2P vs. star topology), evaluated only at small scale |
| FFT-MoE | MoE adapter training | Orthogonal — they design the training; we design the communication |
| Generic FL compression (Top-k, FedSparse) | Parameter-level sparsification | MoE-unaware; SparseFedMoE uses expert boundaries as natural semantic units |

The key differentiator: SparseFedMoE uses the MoE router as a **free importance oracle** — an existing architectural component that tells us which parameters matter for which data, at zero additional cost. Generic compression methods treat all parameters equally and miss this structural information.
