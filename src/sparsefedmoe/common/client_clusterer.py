"""ClientClusterer — groups FL clients by expert activation similarity (arch §3.6).

Optional component, not wired in the default jobs. Kept here so it can be
enabled by a future hierarchical-aggregation job without another round of
migration.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


class ClientClusterer:
    def __init__(
        self,
        num_clusters: Optional[int] = None,
        similarity_threshold: float = 0.85,
        recluster_every: int = 10,
    ):
        self.num_clusters = num_clusters
        self.similarity_threshold = similarity_threshold
        self.recluster_every = recluster_every
        self.clusters: Dict[int, List[str]] = {}

    def fit(self, client_profiles: Dict[str, np.ndarray]) -> Dict[int, List[str]]:
        names = sorted(client_profiles.keys())
        if len(names) <= 1:
            self.clusters = {0: names}
            return self.clusters

        vectors = np.array([client_profiles[n].flatten() for n in names])
        norms = np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-10)
        normalized = vectors / norms
        sim = normalized @ normalized.T

        if self.num_clusters is not None:
            labels = self._kmeans(vectors, self.num_clusters)
        else:
            labels = self._threshold_cluster(sim, self.similarity_threshold)

        grouped: Dict[int, List[str]] = defaultdict(list)
        for i, label in enumerate(labels):
            grouped[label].append(names[i])
        self.clusters = dict(grouped)
        return self.clusters

    def should_recluster(self, current_round: int) -> bool:
        return current_round % self.recluster_every == 0

    def get_cluster_for_client(self, client_name: str) -> Optional[int]:
        for cid, members in self.clusters.items():
            if client_name in members:
                return cid
        return None

    @staticmethod
    def _kmeans(vectors: np.ndarray, k: int, max_iter: int = 50) -> List[int]:
        n = vectors.shape[0]
        if k >= n:
            return list(range(n))
        rng = np.random.default_rng(0)
        centroids = vectors[rng.choice(n, k, replace=False)].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            dists = np.linalg.norm(vectors[:, None] - centroids[None, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                mask = labels == c
                if mask.sum() > 0:
                    centroids[c] = vectors[mask].mean(axis=0)
        return labels.tolist()

    @staticmethod
    def _threshold_cluster(sim: np.ndarray, threshold: float) -> List[int]:
        n = sim.shape[0]
        labels = [-1] * n
        current = 0
        for i in range(n):
            if labels[i] >= 0:
                continue
            labels[i] = current
            for j in range(i + 1, n):
                if labels[j] < 0 and sim[i, j] >= threshold:
                    labels[j] = current
            current += 1
        return labels
