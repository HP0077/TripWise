# ==============================================================================
# TripWise - Model Evaluation Script
# Trains the same ensemble on a user-level split (no user appears in both train
# and test) and reports ranking metrics against random and popularity baselines.
# Results are written to metrics.json.
# ==============================================================================

import json
import time
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, log_loss,
                             precision_recall_fscore_support, top_k_accuracy_score)

from train_model import build_dataset, build_ensemble

K_VALUES = [1, 5, 10]
SEED = 42

# Feature-column prefixes for each questionnaire question
QUESTION_COLUMNS = {
    'budget': 'form_b', 'season': 'form_c', 'activity': 'form_h', 'safety': 'form_i',
    'popularity': 'form_j', 'experiences': 'form_f_', 'scenery': 'form_g_', 'region': 'form_rr_'
}


def ranking_metrics(score_matrix, liked_sets, classes):
    """
    score_matrix: (n_users, n_classes) scores, higher is better.
    liked_sets: list of sets of destination ids each test user swiped 'yes' on.
    Returns per-user averaged HitRate@k, Precision@k, Recall@k, MRR and NDCG@10.
    """
    order = np.argsort(-score_matrix, axis=1)
    ranked_ids = classes[order]

    results = {f"hit_rate@{k}": [] for k in K_VALUES}
    results.update({f"precision@{k}": [] for k in K_VALUES})
    results.update({f"recall@{k}": [] for k in K_VALUES})
    results.update({f"f1@{k}": [] for k in K_VALUES})
    results["mrr"] = []
    results["ndcg@10"] = []

    discounts = 1.0 / np.log2(np.arange(2, 12))
    for ranking, liked in zip(ranked_ids, liked_sets):
        relevant = np.array([dest in liked for dest in ranking])
        for k in K_VALUES:
            hits = relevant[:k].sum()
            results[f"hit_rate@{k}"].append(float(hits > 0))
            results[f"precision@{k}"].append(hits / k)
            results[f"recall@{k}"].append(hits / len(liked))
            precision, recall = hits / k, hits / len(liked)
            results[f"f1@{k}"].append(2 * precision * recall / (precision + recall) if hits else 0.0)
        first_hit = np.argmax(relevant)
        results["mrr"].append(1.0 / (first_hit + 1) if relevant.any() else 0.0)
        dcg = (relevant[:10] * discounts).sum()
        idcg = discounts[:min(len(liked), 10)].sum()
        results["ndcg@10"].append(dcg / idcg)

    return {name: round(float(np.mean(vals)), 4) for name, vals in results.items()}


def classification_metrics(estimator, X_test, y_test, classes):
    """Standard multi-class metrics on the row-level test set."""
    proba = estimator.predict_proba(X_test)
    y_pred = classes[proba.argmax(axis=1)]
    macro = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    weighted = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    return {
        "top1_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "top5_accuracy": round(float(top_k_accuracy_score(y_test, proba, k=5, labels=classes)), 4),
        "top10_accuracy": round(float(top_k_accuracy_score(y_test, proba, k=10, labels=classes)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
        "precision_macro": round(float(macro[0]), 4),
        "recall_macro": round(float(macro[1]), 4),
        "f1_macro": round(float(macro[2]), 4),
        "precision_weighted": round(float(weighted[0]), 4),
        "recall_weighted": round(float(weighted[1]), 4),
        "f1_weighted": round(float(weighted[2]), 4),
        "log_loss": round(float(log_loss(y_test, proba, labels=classes)), 4),
    }


def main():
    X, y, user_ids, _, _ = build_dataset()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups=user_ids))
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test, test_users = X.iloc[test_idx], y.iloc[test_idx], user_ids.iloc[test_idx]
    print(f"Train rows: {len(train_idx)} | Test rows: {len(test_idx)} | Test users: {test_users.nunique()}")

    print("Training ensemble on training users...")
    start = time.perf_counter()
    model = build_ensemble()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start
    classes = model.classes_

    # Row-level classification metrics (one row per liked destination, 225 classes)
    row_metrics = {name: classification_metrics(estimator, X_test, y_test, classes)
                   for name, estimator in [('tripwise_ensemble', model)] + list(model.named_estimators_.items())}
    row_metrics['rf_only'] = row_metrics.pop('rf')
    row_metrics['et_only'] = row_metrics.pop('et')
    most_popular = y_train.value_counts().idxmax()
    row_metrics['chance_level'] = {"top1_accuracy": round(1 / len(classes), 4),
                                   "always_most_popular_accuracy": round(float((y_test == most_popular).mean()), 4)}

    # User-level ranking metrics: one questionnaire -> one ranked list of 225 destinations
    first_row_per_user = ~test_users.duplicated().values
    user_features = X_test[first_row_per_user]
    user_order = test_users[first_row_per_user].values
    liked_by_user = y_test.groupby(test_users.values).apply(set).to_dict()
    liked_sets = [liked_by_user[u] for u in user_order]

    model_scores = model.predict_proba(user_features)

    popularity = y_train.value_counts().reindex(classes, fill_value=0).values.astype(float)
    popularity_scores = np.tile(popularity, (len(user_order), 1))

    rng = np.random.default_rng(SEED)
    random_scores = rng.random((len(user_order), len(classes)))

    # Each half of the ensemble on its own (their probability columns follow model.classes_)
    individual_models = {
        f"{name}_only": ranking_metrics(estimator.predict_proba(user_features), liked_sets, classes)
        for name, estimator in model.named_estimators_.items()
    }

    # Question importance: scramble one question's answers across travellers and
    # measure how much hit rate @10 drops (grouped permutation importance, 5 repeats)
    baseline_hit10 = ranking_metrics(model_scores, liked_sets, classes)["hit_rate@10"]
    question_importance = {}
    for question, prefix in QUESTION_COLUMNS.items():
        columns = [c for c in X.columns if c.startswith(prefix)]
        drops = []
        for _ in range(5):
            scrambled = user_features.copy()
            scrambled[columns] = scrambled[columns].values[rng.permutation(len(scrambled))]
            hit10 = ranking_metrics(model.predict_proba(scrambled), liked_sets, classes)["hit_rate@10"]
            drops.append(baseline_hit10 - hit10)
        question_importance[question] = {"drop": round(float(np.mean(drops)), 4), "std": round(float(np.std(drops)), 4)}
        print(f"  {question}: hit rate @10 drops by {np.mean(drops):.4f} (std {np.std(drops):.4f})")

    # Inference latency for a single questionnaire (what /predict does)
    sample = user_features.iloc[[0]]
    model.predict_proba(sample)
    timings = []
    for _ in range(50):
        t0 = time.perf_counter()
        model.predict_proba(sample)
        timings.append((time.perf_counter() - t0) * 1000)

    report = {
        "split": "GroupShuffleSplit by user id, 90% train / 10% test",
        "n_destinations": int(len(classes)),
        "n_features": int(X.shape[1]),
        "n_train_rows": int(len(train_idx)),
        "n_test_rows": int(len(test_idx)),
        "n_test_users": int(len(user_order)),
        "train_seconds": round(train_seconds, 1),
        "inference_ms_median": round(float(np.median(timings)), 1),
        "row_level": row_metrics,
        "user_level": {
            "tripwise_ensemble": ranking_metrics(model_scores, liked_sets, classes),
            "popularity_baseline": ranking_metrics(popularity_scores, liked_sets, classes),
            "random_baseline": ranking_metrics(random_scores, liked_sets, classes),
            **individual_models,
        },
        "question_importance": dict(
            sorted(question_importance.items(), key=lambda item: item[1]["drop"], reverse=True)
        ),
    }

    with open('metrics.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
