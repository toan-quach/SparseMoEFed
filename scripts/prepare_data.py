"""Partition instruction data into per-client shards for SparseFedMoE.

Three strategies (arch §7.1):
  - ``domain``: one category of databricks/databricks-dolly-15k per client.
  - ``dirichlet``: Dirichlet(α) mix of dolly categories.
  - ``mixed``: distinct source datasets per client (finance / medical / general)
    to produce the strongest domain signal.

Outputs ``<output_dir>/client_{i}/`` in HuggingFace ``save_to_disk`` format,
loadable by the trainer's ``datasets.load_from_disk``.

Usage:
    python scripts/prepare_data.py --num_clients 2 --strategy domain \\
        --output_dir ./data --samples_per_client 64
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict

import numpy as np
from datasets import Dataset, load_dataset


def _format_example(item: Dict) -> str:
    parts = []
    for field in ("instruction", "input", "question"):
        val = item.get(field)
        if val:
            parts.append(str(val))
    context = item.get("context")
    if context:
        parts.append(str(context))
    for field in ("response", "output", "answer", "text"):
        val = item.get(field)
        # NOTE: bug?
        if val and field != parts:
            parts.append(str(val))
            break
    text = "\n".join(parts).strip()
    return text


def domain_partition(num_clients: int, samples_per_client: int):
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    grouped = defaultdict(list)
    for item in ds:
        cat = item.get("category", "general")
        text = _format_example(item)
        if len(text) > 50:
            grouped[cat].append({"text": text, "category": cat})

    categories = sorted(grouped.keys())
    print(f"Categories: {categories}")

    out = {}
    for i in range(num_clients):
        cat = categories[i % len(categories)]
        samples = list(grouped[cat])
        if len(samples) < samples_per_client:
            samples = samples * (samples_per_client // max(len(samples), 1) + 1)
        samples = samples[:samples_per_client]
        out[i] = {"dataset": Dataset.from_list(samples), "domain": cat}
        print(f"  client_{i}: domain={cat!r} samples={len(samples)}")
    return out


def dirichlet_partition(num_clients: int, alpha: float, samples_per_client: int):
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    grouped = defaultdict(list)
    for item in ds:
        cat = item.get("category", "general")
        text = _format_example(item)
        if len(text) > 50:
            grouped[cat].append({"text": text, "category": cat})

    categories = sorted(grouped.keys())
    proportions = np.random.dirichlet([alpha] * len(categories), size=num_clients)

    out = {}
    for i in range(num_clients):
        samples = []
        for k, cat in enumerate(categories):
            n = int(proportions[i, k] * samples_per_client)
            pool = grouped[cat]
            if not pool:
                continue
            idx = np.random.choice(len(pool), size=min(n, len(pool)), replace=True)
            samples.extend([pool[j] for j in idx])
        np.random.shuffle(samples)
        samples = samples[:samples_per_client]
        dom = categories[int(np.argmax(proportions[i]))]
        out[i] = {"dataset": Dataset.from_list(samples), "domain": f"dirichlet:{dom}"}
        print(f"  client_{i}: dominant={dom!r} samples={len(samples)}")
    return out


def mixed_partition(num_clients: int, samples_per_client: int):
    sources = [
        ("gbharti/finance-alpaca", "finance"),
        ("medalpaca/medical_meadow_medical_flashcards", "medical"),
        ("databricks/databricks-dolly-15k", "general"),
    ]
    domain_data = {}
    for ds_name, dom in sources:
        try:
            ds = load_dataset(ds_name, split="train")
            rows = [
                {"text": _format_example(it), "category": dom}
                for it in ds
                if len(_format_example(it)) > 50
            ]
            domain_data[dom] = rows[: samples_per_client * 2]
            print(f"  loaded {len(domain_data[dom])} from {ds_name}")
        except Exception as e:  # noqa: BLE001
            print(f"  failed to load {ds_name}: {e}")

    doms = list(domain_data.keys())
    out = {}
    for i in range(num_clients):
        dom = doms[i % len(doms)]
        samples = domain_data[dom][:samples_per_client]
        if len(samples) < samples_per_client:
            samples = (samples * (samples_per_client // max(len(samples), 1) + 1))[:samples_per_client]
        out[i] = {"dataset": Dataset.from_list(samples), "domain": dom}
        print(f"  client_{i}: domain={dom!r} samples={len(samples)}")
    return out


def save_partitions(client_datasets, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    info = {}
    for i, data in client_datasets.items():
        path = os.path.join(output_dir, f"client_{i}")
        data["dataset"].save_to_disk(path)
        info[str(i)] = {"domain": data["domain"], "n": len(data["dataset"]), "path": path}
    with open(os.path.join(output_dir, "partition_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nWrote {len(client_datasets)} partitions to {output_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_clients", type=int, default=2)
    p.add_argument("--strategy", choices=["domain", "dirichlet", "mixed"], default="domain")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--samples_per_client", type=int, default=64)
    p.add_argument("--output_dir", default="./data")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    print(f"prepare_data: strategy={args.strategy} clients={args.num_clients}")

    if args.strategy == "domain":
        client_ds = domain_partition(args.num_clients, args.samples_per_client)
    elif args.strategy == "dirichlet":
        client_ds = dirichlet_partition(args.num_clients, args.alpha, args.samples_per_client)
    else:
        client_ds = mixed_partition(args.num_clients, args.samples_per_client)

    save_partitions(client_ds, args.output_dir)


if __name__ == "__main__":
    main()
