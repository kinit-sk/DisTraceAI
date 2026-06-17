"""Periodic re-clustering sweep (README §6).

Incremental assignment is path-dependent: early articles can lock in cluster
boundaries later ones would not have produced. The sweep finds low-cohesion
narratives (mean pairwise cosine of member central claims below a threshold),
splits them with k-means (k=2), removes the degraded narrative, and re-assigns
its members through the assigner — which merges or re-seeds them coherently.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ReclusteringSweep:
    def __init__(self, kb, assigner, embedder, *,
                 min_size: int = 5, cohesion_threshold: float = 0.55) -> None:
        # Knowledge base containing narratives and sub-narratives.
        self.kb = kb

        # Narrative assignment component responsible for assigning
        # sub-narratives to existing narratives or creating new ones.
        self.assigner = assigner

        # Embedding model used to encode central claims into vectors.
        self.embedder = embedder

        # Minimum number of members required before a narrative is
        # considered for re-clustering.
        self.min_size = min_size

        # Narratives with cohesion below this threshold will be split.
        self.cohesion_threshold = cohesion_threshold

    def _cohesion(self, claims: list[str]) -> float:
        """Compute mean pairwise cosine similarity between claims."""

        # Single-member narratives are considered perfectly cohesive.
        if len(claims) <= 1:
            return 1.0

        # Generate embeddings for all claims.
        embs = np.asarray(self.embedder.encode(claims), dtype=np.float32)

        # L2-normalize embeddings so dot product equals cosine similarity.
        embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

        # Pairwise cosine similarity matrix.
        sim = embs @ embs.T

        # Extract only the upper triangle (excluding diagonal) to avoid
        # self-similarities and duplicate pairs.
        n = len(claims)
        pairs = [sim[i, j] for i in range(n) for j in range(i + 1, n)]

        # Return mean pairwise similarity.
        return float(np.mean(pairs)) if pairs else 1.0

    def _split(self, members: list) -> tuple[list, list]:
        """Split a narrative into two groups using k-means clustering."""
        claims = [m.central_claim for m in members]

        try:
            from sklearn.cluster import KMeans

            # Embed each member's central claim.
            embs = np.asarray(self.embedder.encode(claims), dtype=np.float32)

            # Partition members into two clusters.
            labels = KMeans(
                n_clusters=2,
                n_init=10,
                random_state=42
            ).fit_predict(embs)

            # Separate members according to cluster labels.
            a = [m for m, l in zip(members, labels) if l == 0]
            b = [m for m, l in zip(members, labels) if l == 1]

            # Use the k-means result only if both groups are non-empty.
            if a and b:
                return a, b

        except Exception as exc:
            logger.warning(
                "[recluster] k-means split failed (%s); using 50/50.",
                exc
            )

        # Fallback: split members evenly if clustering fails.
        mid = len(members) // 2
        return members[:mid], members[mid:]

    def run(self) -> dict:
        """Run a full re-clustering sweep across this dataset/backend's narratives."""
        dataset = self.assigner.dataset
        detector = self.assigner.detector

        # Re-use the assigner's in-memory SN index when populated;
        # fall back to KB only for any IDs it doesn't know about.
        base_index = getattr(self.assigner, "_sn_index", {})
        if base_index:
            sns = dict(base_index)
        else:
            sns = {sn.id: sn
                   for sn in self.kb.sub_narratives(dataset, detector)}

        # Retrieve all narratives managed by the current dataset/backend.
        narratives = self.kb.narratives(dataset, self.assigner.backend.name)

        checked = split = 0

        # Track narrative count before re-clustering.
        before = len(self.assigner.narratives)

        # Iterate over a copy since narratives may be removed during the sweep.
        for nar in list(narratives):

            # Resolve narrative member IDs to actual sub-narrative objects.
            members = [
                sns[sid]
                for sid in nar.sub_narratives
                if sid in sns
            ]

            # Skip narratives that are too small to evaluate.
            if len(members) < self.min_size:
                continue

            checked += 1

            # Skip narratives that are already sufficiently cohesive.
            if (
                self._cohesion([m.central_claim for m in members])
                >= self.cohesion_threshold
            ):
                continue

            logger.info(
                "[recluster] splitting narrative %s (%d members)",
                nar.id,
                len(members)
            )

            # Divide the low-cohesion narrative into two candidate groups.
            group_a, group_b = self._split(members)

            # Remove the original degraded narrative from the registry/KB.
            self.assigner.remove_narrative(nar.id)

            # Reassign all members through the normal assignment pipeline.
            # This allows them to merge into existing narratives or seed
            # new narratives consistently.
            for group in (group_a, group_b):
                for sn in group:
                    self.assigner.assign(sn)

            split += 1

        # Track narrative count after re-clustering.
        after = len(self.assigner.narratives)

        # Summarize sweep results.
        result = {
            "checked": checked,
            "split": split,
            "narrative_delta": after - before,
        }

        logger.info("[recluster] sweep complete: %s", result)
        return result
