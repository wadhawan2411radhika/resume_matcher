"""
Evaluation Metrics for Resume Matching.

Computes ranking quality metrics against ground truth labels.

Metrics:
  - Spearman Rank Correlation: Overall rank order agreement
  - nDCG@K: Normalized Discounted Cumulative Gain — rewards putting best candidates first
  - Precision@K: Fraction of top-K results that are "good matches"
  - Score Correlation: Pearson correlation between predicted and ground truth scores
"""

import json
import math
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    spearman_rho: float
    spearman_pvalue: float
    ndcg_at_3: float
    ndcg_at_5: float
    precision_at_3: float
    precision_at_5: float
    pearson_r: float
    pearson_pvalue: float
    n_candidates: int

    def print_report(self):
        print("\n" + "=" * 55)
        print("  EVALUATION METRICS REPORT")
        print("=" * 55)
        print(f"  Candidates evaluated   : {self.n_candidates}")
        print(f"  Spearman ρ (rank corr) : {self.spearman_rho:+.3f}  (p={self.spearman_pvalue:.3f})")
        print(f"  Pearson r  (score corr): {self.pearson_r:+.3f}  (p={self.pearson_pvalue:.3f})")
        print(f"  nDCG@3                 : {self.ndcg_at_3:.3f}")
        print(f"  nDCG@5                 : {self.ndcg_at_5:.3f}")
        print(f"  Precision@3            : {self.precision_at_3:.3f}")
        print(f"  Precision@5            : {self.precision_at_5:.3f}")
        print("=" * 55)
        print()
        # Interpretation
        if self.spearman_rho >= 0.8:
            print("  ✓ Excellent rank agreement with ground truth")
        elif self.spearman_rho >= 0.6:
            print("  ~ Good rank agreement — some ordering differences")
        elif self.spearman_rho >= 0.4:
            print("  ! Moderate agreement — system needs calibration")
        else:
            print("  ✗ Weak agreement — significant ranking errors")
        print()


def _dcg_at_k(relevances: list[float], k: int) -> float:
    """Compute DCG@K. Relevances are in ranked order (rank 1 first)."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


def _ndcg_at_k(predicted_ranking: list[str], ground_truth: dict[str, float], k: int) -> float:
    """
    Compute nDCG@K.

    Args:
        predicted_ranking: Candidate IDs sorted by predicted score (best first).
        ground_truth: Dict mapping candidate_id -> true relevance score.
        k: Cutoff.
    """
    predicted_relevances = [ground_truth.get(cid, 0.0) for cid in predicted_ranking[:k]]
    ideal_relevances = sorted(ground_truth.values(), reverse=True)

    dcg = _dcg_at_k(predicted_relevances, k)
    idcg = _dcg_at_k(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


def _precision_at_k(predicted_ranking: list[str], ground_truth: dict[str, float], k: int, threshold: float = 0.6) -> float:
    """
    Precision@K: Fraction of top-K results with ground truth relevance >= threshold.
    """
    top_k = predicted_ranking[:k]
    hits = sum(1 for cid in top_k if ground_truth.get(cid, 0.0) >= threshold)
    return hits / k


def compute_metrics(
    results: list,  # List of MatchResult objects from engine
    eval_dataset_path: str = "evaluation/eval_dataset.json",
) -> EvalMetrics:
    """
    Compute evaluation metrics by comparing predicted scores to ground truth.

    Args:
        results: Sorted list of MatchResult objects from MatchingEngine.match()
        eval_dataset_path: Path to eval_dataset.json

    Returns:
        EvalMetrics dataclass with all computed metrics.
    """
    with open(eval_dataset_path) as f:
        eval_data = json.load(f)

    ground_truth = {
        c["id"]: c["ground_truth_label"]
        for c in eval_data["candidates"]
    }

    # Build predicted ranking and score vectors
    predicted_ranking = []
    predicted_scores = []
    true_scores = []

    for result in results:
        cid = result.candidate_name
        if cid in ground_truth:
            predicted_ranking.append(cid)
            predicted_scores.append(result.score)
            true_scores.append(ground_truth[cid])

    if len(predicted_scores) < 2:
        raise ValueError("Need at least 2 matched candidates for metric computation.")

    # Spearman rank correlation
    spearman_rho, spearman_p = stats.spearmanr(predicted_scores, true_scores)

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(predicted_scores, true_scores)

    # Ranking metrics
    ndcg_3 = _ndcg_at_k(predicted_ranking, ground_truth, k=3)
    ndcg_5 = _ndcg_at_k(predicted_ranking, ground_truth, k=5)
    prec_3 = _precision_at_k(predicted_ranking, ground_truth, k=3)
    prec_5 = _precision_at_k(predicted_ranking, ground_truth, k=5)

    return EvalMetrics(
        spearman_rho=round(float(spearman_rho), 4),
        spearman_pvalue=round(float(spearman_p), 4),
        ndcg_at_3=round(ndcg_3, 4),
        ndcg_at_5=round(ndcg_5, 4),
        precision_at_3=round(prec_3, 4),
        precision_at_5=round(prec_5, 4),
        pearson_r=round(float(pearson_r), 4),
        pearson_pvalue=round(float(pearson_p), 4),
        n_candidates=len(predicted_scores),
    )


def print_score_comparison(results: list, eval_dataset_path: str = "evaluation/eval_dataset.json"):
    """Print a side-by-side comparison of predicted vs ground truth scores."""
    with open(eval_dataset_path) as f:
        eval_data = json.load(f)

    ground_truth = {c["id"]: c["ground_truth_label"] for c in eval_data["candidates"]}
    labels = {c["id"]: c["label_string"] for c in eval_data["candidates"]}

    print("\n── SCORE COMPARISON: Predicted vs Ground Truth ──────────────────────")
    print(f"{'RANK':<5} {'CANDIDATE':<25} {'PREDICTED':<12} {'GT SCORE':<12} {'GT LABEL':<25} {'DELTA'}")
    print("-" * 95)

    for rank, result in enumerate(results, 1):
        cid = result.candidate_name
        gt = ground_truth.get(cid, "N/A")
        label = labels.get(cid, "Unknown")
        delta = result.score - gt if isinstance(gt, float) else "—"
        delta_str = f"{delta:+.3f}" if isinstance(delta, float) else delta

        print(
            f"{rank:<5} {cid:<25} {result.score:<12.3f} {str(gt):<12} "
            f"{label:<25} {delta_str}"
        )
    print()
