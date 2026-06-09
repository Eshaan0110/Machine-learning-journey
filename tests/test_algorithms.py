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
