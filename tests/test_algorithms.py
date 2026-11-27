"""
Unit tests for the mathematical core of each algorithm in this repo.

All implementations are self-contained numpy reproductions of the logic
in the notebooks — no notebook import is needed.
"""

import numpy as np
import pytest
from itertools import permutations


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return max(0.0, -np.sum(p * np.log2(p + 1e-15)))


def _gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p ** 2)


# ---------------------------------------------------------------------------
# Simple Linear Regression
# ---------------------------------------------------------------------------

class TestSimpleLinearRegression:
    def test_slope_intercept_no_noise(self):
        X = np.array([1, 2, 3, 4, 5], dtype=float)
        y = 2 * X + 3
        m_x, m_y = np.mean(X), np.mean(y)
        m = np.sum((X - m_x) * (y - m_y)) / np.sum((X - m_x) ** 2)
        c = m_y - m * m_x
        assert abs(m - 2.0) < 1e-10
        assert abs(c - 3.0) < 1e-10

    def test_perfect_fit_r2(self):
        X = np.array([1, 2, 3, 4, 5], dtype=float)
        y = 3 * X - 1
        m_x, m_y = np.mean(X), np.mean(y)
        m = np.sum((X - m_x) * (y - m_y)) / np.sum((X - m_x) ** 2)
        c = m_y - m * m_x
        y_pred = m * X + c
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        assert abs(1.0 - ss_res / ss_tot - 1.0) < 1e-10

    def test_gradient_descent_converges(self):
        np.random.seed(0)
        X = np.linspace(0, 10, 50)
        y = 2.5 * X + 1.0 + np.random.randn(50) * 0.3
        m, c, lr, n = 0.0, 0.0, 0.001, len(X)
        for _ in range(8000):
            y_p = m * X + c
            m -= lr * (-2 / n) * np.sum(X * (y - y_p))
            c -= lr * (-2 / n) * np.sum(y - y_p)
        assert abs(m - 2.5) < 0.3

    def test_mse_nonnegative(self):
        X = np.array([1, 2, 3], dtype=float)
        y = np.array([2, 4, 5], dtype=float)
        m_x, m_y = np.mean(X), np.mean(y)
        m = np.sum((X - m_x) * (y - m_y)) / np.sum((X - m_x) ** 2)
        c = m_y - m * m_x
        mse = np.mean((y - (m * X + c)) ** 2)
        assert mse >= 0.0

    def test_prediction_on_new_point(self):
        X = np.array([1, 2, 3, 4, 5], dtype=float)
        y = 2 * X + 1
        m_x, m_y = np.mean(X), np.mean(y)
        m = np.sum((X - m_x) * (y - m_y)) / np.sum((X - m_x) ** 2)
        c = m_y - m * m_x
        assert abs(m * 10 + c - 21.0) < 1e-10


# ---------------------------------------------------------------------------
# Multiple Linear Regression (Normal Equation)
# ---------------------------------------------------------------------------

class TestMultipleLinearRegression:
    def test_normal_equation_exact(self):
        np.random.seed(7)
        X_raw = np.random.randn(20, 3)
        true_w = np.array([1.0, -2.0, 3.0])
        y = X_raw @ true_w + 5.0
        X = np.c_[np.ones(20), X_raw]
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        assert abs(beta[0] - 5.0) < 1e-8
        assert np.allclose(beta[1:], true_w, atol=1e-8)

    def test_predictions_match_y(self):
        np.random.seed(3)
        X_raw = np.random.randn(10, 2)
        y = 3 * X_raw[:, 0] - X_raw[:, 1] + 2.0
        X = np.c_[np.ones(10), X_raw]
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        y_pred = X @ beta
        assert np.allclose(y, y_pred, atol=1e-8)


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

class TestLogisticRegression:
    def test_sigmoid_range(self):
        z = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
        out = _sigmoid(z)
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_sigmoid_at_zero(self):
        assert abs(_sigmoid(0.0) - 0.5) < 1e-10

    def test_sigmoid_monotone(self):
        z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        assert np.all(np.diff(_sigmoid(z)) > 0)

    def test_converges_linearly_separable(self):
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(50, 2) + [-3, 0],
            np.random.randn(50, 2) + [3, 0],
        ])
        y = np.array([0] * 50 + [1] * 50, dtype=float)
        n, d = X.shape
        W, b, lr = np.zeros(d), 0.0, 0.5
        for _ in range(1000):
            yhat = _sigmoid(X @ W + b)
            W -= lr * (X.T @ (yhat - y)) / n
            b -= lr * np.mean(yhat - y)
        preds = (_sigmoid(X @ W + b) >= 0.5).astype(int)
        assert np.mean(preds == y.astype(int)) > 0.95

    def test_l2_shrinks_weights(self):
        np.random.seed(1)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        n = len(y)

        def train(l2):
            W, b = np.zeros(5), 0.0
            for _ in range(500):
                yhat = _sigmoid(X @ W + b)
                W -= 0.1 * ((X.T @ (yhat - y)) / n + l2 * W)
                b -= 0.1 * np.mean(yhat - y)
            return np.linalg.norm(W)

        assert train(1.0) < train(0.0)

    def test_binary_cross_entropy_decreases(self):
        np.random.seed(5)
        X = np.random.randn(60, 3)
        y = (X[:, 0] > 0).astype(float)
        n = len(y)
        W, b = np.zeros(3), 0.0
        losses = []
        for _ in range(200):
            yhat = _sigmoid(X @ W + b)
            eps = 1e-15
            yhat_c = np.clip(yhat, eps, 1 - eps)
            losses.append(-np.mean(y * np.log(yhat_c) + (1 - y) * np.log(1 - yhat_c)))
            W -= 0.1 * (X.T @ (yhat - y)) / n
            b -= 0.1 * np.mean(yhat - y)
        assert losses[-1] < losses[0]


# ---------------------------------------------------------------------------
# Distance Metrics
# ---------------------------------------------------------------------------

class TestDistanceMetrics:
    def test_euclidean_3_4_5(self):
        assert abs(np.sqrt(np.sum((np.array([0, 0]) - np.array([3, 4])) ** 2)) - 5.0) < 1e-10

    def test_euclidean_self_zero(self):
        p = np.array([1.0, 2.0, 3.0])
        assert np.sqrt(np.sum((p - p) ** 2)) == 0.0

    def test_manhattan_known(self):
        p1, p2 = np.array([1, 2, 3]), np.array([4, 6, 3])
        assert np.sum(np.abs(p1 - p2)) == 7

    def test_minkowski_p1_is_manhattan(self):
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([4.0, 6.0, 3.0])
        assert abs(np.sum(np.abs(p1 - p2) ** 1) ** 1 - np.sum(np.abs(p1 - p2))) < 1e-10

    def test_minkowski_p2_is_euclidean(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([3.0, 4.0])
        assert abs(np.sum(np.abs(p1 - p2) ** 2) ** 0.5 - 5.0) < 1e-10

    def test_cosine_orthogonal(self):
        p1, p2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        cos = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
        assert abs(cos) < 1e-10

    def test_cosine_parallel(self):
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = 2 * p1
        cos = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
        assert abs(cos - 1.0) < 1e-10

    def test_triangle_inequality_euclidean(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 0.0])
        c = np.array([0.0, 4.0])
        d_ab = np.linalg.norm(a - b)
        d_bc = np.linalg.norm(b - c)
        d_ac = np.linalg.norm(a - c)
        assert d_ab + d_bc >= d_ac


# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------

def _kmeans(X, k, n_iter=200, seed=0):
    np.random.seed(seed)
    centroids = X[np.random.choice(len(X), k, replace=False)].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(n_iter):
        dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centroids])
        new_labels = np.argmin(dists, axis=0)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for j in range(k):
            if (labels == j).sum() > 0:
                centroids[j] = X[labels == j].mean(axis=0)
    return labels, centroids


class TestKMeans:
    def test_separable_clusters(self):
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(50, 2) + [0, 0],
            np.random.randn(50, 2) + [12, 0],
            np.random.randn(50, 2) + [0, 12],
        ])
        true_labels = np.array([0] * 50 + [1] * 50 + [2] * 50)
        # Try a few seeds; well-separated clusters should converge for at least one
        best = 0.0
        for seed in range(5):
            labels, _ = _kmeans(X, k=3, seed=seed)
            for perm in permutations([0, 1, 2]):
                mapped = np.array([perm[int(l)] for l in labels])
                acc = np.mean(mapped == true_labels)
                if acc > best:
                    best = acc
        assert best > 0.95

    def test_centroid_equals_cluster_mean(self):
        np.random.seed(7)
        X = np.vstack([np.random.randn(30, 2), np.random.randn(30, 2) + [8, 8]])
        labels, centroids = _kmeans(X, k=2, seed=0)
        for j in range(2):
            if (labels == j).sum() > 0:
                assert np.allclose(centroids[j], X[labels == j].mean(axis=0), atol=1e-10)

    def test_inertia_never_increases(self):
        np.random.seed(3)
        X = np.random.randn(80, 2)
        np.random.seed(0)
        centroids = X[np.random.choice(80, 3, replace=False)].copy()
        labels = np.zeros(80, dtype=int)
        prev = float('inf')
        for _ in range(30):
            dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centroids])
            labels = np.argmin(dists, axis=0)
            inertia = sum(
                np.sum((X[labels == j] - centroids[j]) ** 2)
                for j in range(3) if (labels == j).sum() > 0
            )
            assert inertia <= prev + 1e-9
            prev = inertia
            for j in range(3):
                if (labels == j).sum() > 0:
                    centroids[j] = X[labels == j].mean(axis=0)

    def test_k1_puts_all_in_one_cluster(self):
        X = np.random.randn(40, 3)
        labels, _ = _kmeans(X, k=1)
        assert np.all(labels == 0)


# ---------------------------------------------------------------------------
# Entropy / Gini / Information Gain (Decision Trees)
# ---------------------------------------------------------------------------

class TestEntropyGini:
    def test_entropy_uniform_binary(self):
        y = np.array([0, 1])
        assert abs(_entropy(y) - 1.0) < 1e-9

    def test_entropy_pure(self):
        y = np.array([0, 0, 0, 0])
        assert _entropy(y) < 1e-9

    def test_entropy_nonnegative(self):
        for y in [np.array([0, 1, 2]), np.array([0, 0, 1]), np.array([1])]:
            assert _entropy(y) >= 0

    def test_entropy_more_classes_higher(self):
        y2 = np.array([0, 0, 1, 1])
        y4 = np.array([0, 1, 2, 3])
        assert _entropy(y4) > _entropy(y2)

    def test_gini_uniform_binary(self):
        y = np.array([0, 1])
        assert abs(_gini(y) - 0.5) < 1e-10

    def test_gini_pure(self):
        y = np.array([1, 1, 1])
        assert _gini(y) < 1e-10

    def test_gini_in_0_1(self):
        for y in [np.array([0, 1, 2]), np.array([0, 0, 1])]:
            g = _gini(y)
            assert 0 <= g <= 1

    def test_information_gain_perfect_split(self):
        parent = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        left   = np.array([0, 0, 0, 0])
        right  = np.array([1, 1, 1, 1])
        n = len(parent)
        ig = _entropy(parent) - (len(left)/n)*_entropy(left) - (len(right)/n)*_entropy(right)
        assert abs(ig - 1.0) < 1e-9

    def test_information_gain_no_split_zero(self):
        parent = np.array([0, 1, 0, 1])
        left   = parent[:2]
        right  = parent[2:]
        n = len(parent)
        ig = _entropy(parent) - (len(left)/n)*_entropy(left) - (len(right)/n)*_entropy(right)
        assert ig < 1e-9


# ---------------------------------------------------------------------------
# Q-Learning
# ---------------------------------------------------------------------------

class TestQLearning:
    def test_update_formula(self):
        Q = np.zeros((4, 2))
        lr, gamma = 0.5, 0.9
        s, a, r, s_ = 0, 1, 1.0, 2
        Q[s, a] += lr * (r + gamma * np.max(Q[s_]) - Q[s, a])
        assert abs(Q[s, a] - 0.5) < 1e-10

    def test_greedy_picks_max(self):
        Q = np.array([[0.1, 0.9], [0.5, 0.3]])
        assert np.argmax(Q[0]) == 1
        assert np.argmax(Q[1]) == 0

    def test_epsilon_zero_always_greedy(self):
        Q = np.array([1.0, 100.0, 0.0])
        np.random.seed(42)
        actions = []
        for _ in range(50):
            eps = 0.0
            a = np.random.randint(len(Q)) if np.random.rand() < eps else int(np.argmax(Q))
            actions.append(a)
        assert all(a == 1 for a in actions)

    def test_q_values_bounded_by_reward_sum(self):
        # With discount < 1 and finite reward, Q values must converge
        Q = np.zeros((3, 2))
        lr, gamma, r = 0.1, 0.9, 1.0
        for step in range(5000):
            s  = step % 3
            a  = np.random.randint(2)
            s_ = (s + 1) % 3
            Q[s, a] += lr * (r + gamma * np.max(Q[s_]) - Q[s, a])
        # Maximum possible Q value for constant reward is r / (1 - gamma) = 10
        assert np.all(Q <= 10.0 + 0.5)

    def test_convergence_on_simple_chain(self):
        # 1-D chain: states 0,1,2 — reward=+1 on reaching state 2
        np.random.seed(0)
        Q = np.zeros((3, 2))  # 2 actions: left=0, right=1
        lr, gamma = 0.5, 0.9
        for _ in range(2000):
            s = np.random.randint(2)  # start from 0 or 1
            a = 1                     # always go right
            s_ = min(s + 1, 2)
            reward = 1.0 if s_ == 2 else 0.0
            Q[s, a] += lr * (reward + gamma * np.max(Q[s_]) - Q[s, a])
        # After convergence, Q[1, right] > Q[0, right]
        assert Q[1, 1] > Q[0, 1]


# ---------------------------------------------------------------------------
# Neural Network activation functions
# ---------------------------------------------------------------------------

class TestActivations:
    def test_sigmoid_output_range(self):
        z = np.random.randn(100) * 10
        out = _sigmoid(z)
        assert np.all(out > 0) and np.all(out < 1)

    def test_relu_zero_for_negatives(self):
        z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        relu = np.maximum(0, z)
        assert np.all(relu[:2] == 0)
        assert np.all(relu[3:] > 0)

    def test_relu_identity_for_positives(self):
        z = np.array([0.5, 1.0, 2.0, 5.0])
        assert np.allclose(np.maximum(0, z), z)

    def test_tanh_range(self):
        z = np.random.randn(100) * 10
        out = np.tanh(z)
        assert np.all(out >= -1) and np.all(out <= 1)

    def test_tanh_odd_symmetry(self):
        z = np.array([1.0, 2.0, 3.0])
        assert np.allclose(np.tanh(-z), -np.tanh(z))

    def test_softmax_sums_to_one(self):
        z = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.5]])
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        out = e / e.sum(axis=1, keepdims=True)
        assert np.allclose(out.sum(axis=1), 1.0)

    def test_softmax_preserves_order(self):
        z = np.array([[1.0, 3.0, 2.0]])
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        out = e / e.sum(axis=1, keepdims=True)
        assert np.argmax(out) == 1


# ---------------------------------------------------------------------------
# Boosting (Gradient Boosting residuals, AdaBoost weight updates)
# ---------------------------------------------------------------------------

class TestBoosting:
    def test_gradient_boosting_reduces_residual_variance(self):
        # A single boosting round should shrink the residual toward zero
        # whenever the weak learner's predictions correlate with it.
        rng = np.random.RandomState(0)
        y = rng.randn(50) * 5
        residual = y - np.mean(y)
        weak_pred = residual * 0.8 + rng.randn(50) * 0.1  # learner tracks residual closely
        lr = 0.1
        new_residual = residual - lr * weak_pred
        assert np.var(new_residual) < np.var(residual)

    def test_gradient_boosting_converges_toward_target(self):
        rng = np.random.RandomState(1)
        y = rng.randn(30) * 3 + 10
        pred = np.full(30, np.mean(y))
        errors = []
        for _ in range(20):
            residual = y - pred
            errors.append(np.mean(residual ** 2))
            pred = pred + 0.3 * residual  # perfect weak learner = identity on residual
        assert errors[-1] < errors[0]

    def test_adaboost_alpha_high_when_error_low(self):
        err_low = 0.05
        err_high = 0.45
        alpha_low = 0.5 * np.log((1 - err_low) / err_low)
        alpha_high = 0.5 * np.log((1 - err_high) / err_high)
        assert alpha_low > alpha_high
        assert alpha_low > 0 and alpha_high > 0

    def test_adaboost_alpha_zero_at_random_guessing(self):
        err = 0.5
        alpha = 0.5 * np.log((1 - err) / err)
        assert np.isclose(alpha, 0.0)

    def test_adaboost_weight_update_increases_misclassified(self):
        n = 10
        w = np.full(n, 1 / n)
        y = np.ones(n)
        pred = np.ones(n)
        pred[:3] = -1  # first 3 samples misclassified
        err = np.sum(w[pred != y])
        alpha = 0.5 * np.log((1 - err) / err)
        w_new = w * np.exp(-alpha * y * pred)
        w_new /= w_new.sum()
        assert w_new[:3].min() > w_new[3:].max()

    def test_adaboost_weights_stay_normalized(self):
        rng = np.random.RandomState(2)
        n = 20
        w = np.full(n, 1 / n)
        y = rng.choice([-1, 1], size=n)
        for _ in range(5):
            pred = rng.choice([-1, 1], size=n)
            err = np.clip(np.sum(w[pred != y]), 1e-10, 1 - 1e-10)
            alpha = 0.5 * np.log((1 - err) / err)
            w = w * np.exp(-alpha * y * pred)
            w = w / w.sum()
            assert np.isclose(w.sum(), 1.0)
            assert np.all(w >= 0)

    def test_adaboost_decision_stump_beats_random_on_separable_data(self):
        # A single-feature threshold should perfectly separate linearly
        # separable 1D data, giving zero weighted error.
        X = np.array([[1.0], [2.0], [3.0], [8.0], [9.0], [10.0]])
        y = np.array([-1, -1, -1, 1, 1, 1])
        w = np.full(6, 1 / 6)
        best_err = np.inf
        for t in [1.5, 3.5, 5.5, 8.5]:
            pred = np.where(X[:, 0] <= t, -1, 1)
            err = np.sum(w[pred != y])
            best_err = min(best_err, err)
        assert best_err == 0.0


# ---------------------------------------------------------------------------
# K-Fold Cross-Validation
# ---------------------------------------------------------------------------

def _k_fold_split(n_samples, k=5, shuffle=True, seed=None):
    idx = np.arange(n_samples)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[: n_samples % k] += 1
    folds = []
    start = 0
    for size in fold_sizes:
        val_idx = idx[start:start + size]
        train_idx = np.concatenate([idx[:start], idx[start + size:]])
        folds.append((train_idx, val_idx))
        start += size
    return folds


class TestKFoldCrossValidation:
    def test_every_sample_validated_exactly_once(self):
        folds = _k_fold_split(97, k=5, seed=0)
        all_val = np.concatenate([val for _, val in folds])
        assert sorted(all_val) == list(range(97))

    def test_train_val_never_overlap(self):
        folds = _k_fold_split(50, k=5, seed=1)
        for train_idx, val_idx in folds:
            assert len(set(train_idx) & set(val_idx)) == 0

    def test_fold_sizes_within_one_of_each_other(self):
        folds = _k_fold_split(23, k=5, seed=2)
        sizes = [len(val) for _, val in folds]
        assert max(sizes) - min(sizes) <= 1
        assert sum(sizes) == 23

    def test_train_plus_val_covers_all_samples(self):
        n = 40
        folds = _k_fold_split(n, k=4, seed=3)
        for train_idx, val_idx in folds:
            assert len(train_idx) + len(val_idx) == n

    def test_k_equals_n_is_leave_one_out(self):
        folds = _k_fold_split(10, k=10, seed=4)
        assert all(len(val_idx) == 1 for _, val_idx in folds)

    def test_no_shuffle_is_deterministic_contiguous_blocks(self):
        folds = _k_fold_split(10, k=5, shuffle=False)
        assert list(folds[0][1]) == [0, 1]
        assert list(folds[-1][1]) == [8, 9]

    def test_cross_val_score_mean_within_fold_range(self):
        rng = np.random.RandomState(5)
        X = rng.randn(60, 2)
        y = (X[:, 0] > 0).astype(float)

        def fit_predict(X_tr, y_tr, X_val):
            return (X_val[:, 0] > 0).astype(float)

        folds = _k_fold_split(len(X), k=5, seed=6)
        scores = []
        for train_idx, val_idx in folds:
            y_pred = fit_predict(X[train_idx], y[train_idx], X[val_idx])
            scores.append(np.mean(y[val_idx] == y_pred))
        scores = np.array(scores)
        assert scores.min() <= scores.mean() <= scores.max()

    def test_kfold_mean_more_stable_than_single_split(self):
        # Averaging over folds should reduce variance relative to any
        # single fold's estimate, since noise partially cancels.
        rng = np.random.RandomState(7)
        y_true = rng.randint(0, 2, size=200)
        y_pred = y_true.copy()
        flip = rng.choice(200, size=40, replace=False)
        y_pred[flip] = 1 - y_pred[flip]

        folds = _k_fold_split(200, k=10, seed=8)
        fold_accs = np.array([np.mean(y_true[val] == y_pred[val]) for _, val in folds])
        overall_acc = np.mean(y_true == y_pred)
        assert abs(fold_accs.mean() - overall_acc) < 1e-9


class TestModelPersistence:
    """Round-trip save/load of a numpy model via .npz + .json."""

    @staticmethod
    def _fit(X, y, n_iter=300, lr=0.1):
        mean, std = X.mean(0), X.std(0) + 1e-12
        Z = (X - mean) / std
        w, b = np.zeros(X.shape[1]), 0.0
        for _ in range(n_iter):
            err = 1 / (1 + np.exp(-(Z @ w + b))) - y
            w -= lr * Z.T @ err / len(y)
            b -= lr * err.mean()
        return w, b, mean, std

    @staticmethod
    def _proba(X, w, b, mean, std):
        return 1 / (1 + np.exp(-(((X - mean) / std) @ w + b)))

    @staticmethod
    def _save(path, w, b, mean, std):
        import json
        np.savez(path + ".npz", w=w, b=np.array(b), mean=mean, std=std)
        with open(path + ".json", "w") as f:
            json.dump({"n_features": int(w.shape[0])}, f)

    @staticmethod
    def _load(path):
        import json
        with open(path + ".json") as f:
            meta = json.load(f)
        with np.load(path + ".npz") as d:
            w, b, mean, std = d["w"], float(d["b"]), d["mean"], d["std"]
        if w.shape[0] != meta["n_features"]:
            raise ValueError("shape mismatch")
        return w, b, mean, std

    def _data(self):
        rng = np.random.RandomState(0)
        X = rng.randn(80, 4) * [1, 5, 10, 0.1] + [0, 3, -2, 1]
        y = (X[:, 0] + 0.1 * X[:, 1] > 0.3).astype(float)
        return X, y

    def test_roundtrip_predictions_identical(self, tmp_path):
        X, y = self._data()
        params = self._fit(X, y)
        p = str(tmp_path / "m")
        self._save(p, *params)
        assert np.array_equal(self._proba(X, *params), self._proba(X, *self._load(p)))

    def test_scaler_statistics_must_be_persisted(self, tmp_path):
        # Reloading weights but re-estimating the scaler on new data changes outputs.
        X, y = self._data()
        w, b, mean, std = self._fit(X, y)
        X_new = X[:10]
        correct = self._proba(X_new, w, b, mean, std)
        wrong = self._proba(X_new, w, b, X_new.mean(0), X_new.std(0) + 1e-12)
        assert not np.allclose(correct, wrong)

    def test_load_rejects_mismatched_metadata(self, tmp_path):
        import json
        X, y = self._data()
        p = str(tmp_path / "m")
        self._save(p, *self._fit(X, y))
        with open(p + ".json", "w") as f:
            json.dump({"n_features": 99}, f)
        with pytest.raises(ValueError):
            self._load(p)

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            self._load(str(tmp_path / "absent"))


# ---------------------------------------------------------------------------
# Stratified K-Fold and Grid Search
# ---------------------------------------------------------------------------

def _stratified_k_fold_split(y, k=5, shuffle=True, seed=None):
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    members = [[] for _ in range(k)]
    offset = 0
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if shuffle:
            rng.shuffle(idx)
        for j, i in enumerate(idx):
            members[(j + offset) % k].append(i)
        offset += len(idx)
    all_idx = np.arange(len(y))
    out = []
    for m in members:
        val = np.sort(np.array(m, dtype=int))
        out.append((np.setdiff1d(all_idx, val), val))
    return out


def _grid_search_cv(X, y, grid, make_fp, k=3, seed=0):
    import itertools
    names = list(grid)
    folds = _stratified_k_fold_split(y, k, seed=seed)
    results = []
    for vals in itertools.product(*(grid[n] for n in names)):
        params = dict(zip(names, vals))
        fp = make_fp(**params)
        s = [np.mean(y[v] == fp(X[t], y[t], X[v])) for t, v in folds]
        results.append((params, float(np.mean(s))))
    return max(results, key=lambda r: r[1]), results


class TestStratifiedGridSearch:
    def test_folds_partition_samples(self):
        y = np.array([0] * 47 + [1] * 13 + [2] * 20)
        folds = _stratified_k_fold_split(y, k=4, seed=1)
        val = np.concatenate([v for _, v in folds])
        assert sorted(val) == list(range(len(y)))
        for tr, v in folds:
            assert len(set(tr) & set(v)) == 0

    def test_class_ratio_preserved(self):
        y = np.array([0] * 90 + [1] * 10)
        for _, v in _stratified_k_fold_split(y, k=5, seed=0):
            assert y[v].sum() == 2

    def test_small_class_per_fold_counts_differ_by_at_most_one(self):
        y = np.array([0] * 50 + [1] * 7)
        counts = [int(y[v].sum()) for _, v in _stratified_k_fold_split(y, k=5, seed=3)]
        assert max(counts) - min(counts) <= 1 and sum(counts) == 7

    def test_seed_reproducible(self):
        y = np.array([0] * 30 + [1] * 30)
        a = _stratified_k_fold_split(y, 3, seed=5)
        b = _stratified_k_fold_split(y, 3, seed=5)
        assert all(np.array_equal(x[1], z[1]) for x, z in zip(a, b))

    def test_grid_search_picks_best_param_and_covers_grid(self):
        rng = np.random.RandomState(0)
        X = np.vstack([rng.randn(40, 2) - 3, rng.randn(40, 2) + 3])
        y = np.array([0] * 40 + [1] * 40)

        def make_fp(threshold):
            # classify by sign of feature sum relative to threshold
            return lambda Xt, yt, Xv: (Xv.sum(axis=1) > threshold).astype(int)

        best, results = _grid_search_cv(X, y, {"threshold": [-50.0, 0.0, 50.0]}, make_fp)
        assert len(results) == 3
        assert best[0]["threshold"] == 0.0 and best[1] > 0.95


# ---- Classification metrics (mirrors Classification Metrics.ipynb) ----
def confusion_matrix(y_true, y_pred, n_classes=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)   # rows = actual, cols = predicted
    return cm

def _safe_div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

def precision_recall_f1(y_true, y_pred, average=None, n_classes=None):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    pred_pos, actual_pos = cm.sum(axis=0), cm.sum(axis=1)
    if average == 'micro':
        p = r = tp.sum() / cm.sum()
        return p, r, _safe_div(2 * p * r, p + r)
    p, r = _safe_div(tp, pred_pos), _safe_div(tp, actual_pos)
    f = _safe_div(2 * p * r, p + r)
    if average is None:
        return p, r, f
    w = np.ones(len(tp)) / len(tp) if average == 'macro' else actual_pos / actual_pos.sum()
    return (p * w).sum(), (r * w).sum(), (f * w).sum()

def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
def _threshold_counts(y_true, scores):
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    order = np.argsort(-scores, kind='mergesort')
    y_s, s_s = y_true[order], scores[order]
    distinct = np.where(np.diff(s_s))[0]
    idx = np.r_[distinct, len(y_s) - 1]       # last index of each distinct score
    tps = np.cumsum(y_s)[idx]
    fps = 1 + idx - tps
    return fps, tps, s_s[idx]

def roc_curve(y_true, scores):
    fps, tps, thr = _threshold_counts(y_true, scores)
    tpr, fpr = tps / tps[-1], fps / fps[-1]
    return np.r_[0, fpr], np.r_[0, tpr], np.r_[np.inf, thr]

def auc(x, y):
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))   # trapezoid rule

def roc_auc_score(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    return auc(fpr, tpr)

def precision_recall_curve(y_true, scores):
    fps, tps, thr = _threshold_counts(y_true, scores)
    precision, recall = tps / (tps + fps), tps / tps[-1]
    return np.r_[precision[::-1], 1], np.r_[recall[::-1], 0], thr[::-1]

def average_precision(y_true, scores):
    p, r, _ = precision_recall_curve(y_true, scores)
    return float(-np.sum(np.diff(r) * p[:-1]))


class TestClassificationMetrics:
    def test_confusion_matrix_layout(self):
        cm = confusion_matrix([0, 0, 1, 1, 1], [0, 1, 1, 1, 0])
        assert cm.tolist() == [[1, 1], [1, 2]]

    def test_prf_binary_known_values(self):
        p, r, f = precision_recall_f1([0, 0, 1, 1, 1], [0, 1, 1, 1, 0])
        assert np.allclose(p[1], 2 / 3) and np.allclose(r[1], 2 / 3) and np.allclose(f[1], 2 / 3)

    def test_zero_division_returns_zero(self):
        p, r, f = precision_recall_f1([0, 0, 1], [0, 0, 0], n_classes=2)
        assert p[1] == 0 and r[1] == 0 and f[1] == 0

    def test_averages_match_sklearn(self):
        from sklearn import metrics as skm
        rng = np.random.RandomState(0)
        yt, yp = rng.randint(0, 3, 200), rng.randint(0, 3, 200)
        for avg in ['micro', 'macro', 'weighted']:
            ref = skm.precision_recall_fscore_support(yt, yp, average=avg)[:3]
            assert np.allclose(precision_recall_f1(yt, yp, average=avg), ref)

    def test_roc_auc_and_ap_match_sklearn_with_ties(self):
        from sklearn import metrics as skm
        rng = np.random.RandomState(1)
        y = rng.randint(0, 2, 300)
        s = np.round(rng.rand(300) + 0.5 * y, 1)   # many tied scores
        assert np.isclose(roc_auc_score(y, s), skm.roc_auc_score(y, s))
        assert np.isclose(average_precision(y, s), skm.average_precision_score(y, s))

    def test_perfect_and_random_auc(self):
        assert roc_auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
        assert roc_auc_score([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
