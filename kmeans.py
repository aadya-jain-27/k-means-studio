"""K-means from scratch. This is the whole algorithm, about 60 lines.
Read LEARN.md first, then follow the numbered steps below."""
import numpy as np


def init_random(X, k, rng):
    """Pick k random data points as the starting centres."""
    return X[rng.choice(len(X), k, replace=False)].copy()


def init_plus_plus(X, k, rng):
    """k-means++: first centre random, then each new centre is likely to be FAR from the others.
    This spreads the starting centres out, which usually gives a better final answer."""
    centers = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        # squared distance from every point to its nearest chosen centre
        d2 = np.min([((X - c) ** 2).sum(axis=1) for c in centers], axis=0)
        centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])   # far points = bigger chance
    return np.array(centers)


def assign(X, centers):
    """STEP 1: give every point the label of its nearest centre."""
    # distances has shape (n_points, k): distance from each point to each centre
    distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1), distances.min(axis=1)


def update(X, labels, centers):
    """STEP 2: move every centre to the mean of the points that chose it."""
    new = centers.copy()
    for j in range(len(centers)):
        members = X[labels == j]
        if len(members) > 0:               # an empty cluster keeps its old centre
            new[j] = members.mean(axis=0)
    return new


def kmeans(X, k, init="kmeans++", max_iter=100, seed=0):
    """Returns centers, labels, inertia and a history of every iteration."""
    rng = np.random.default_rng(seed)
    centers = (init_plus_plus if init == "kmeans++" else init_random)(X, k, rng)
    history = []
    for _ in range(max_iter):
        labels, sq_dist = assign(X, centers)                    # step 1
        history.append((centers.copy(), labels, sq_dist.sum()))
        new_centers = update(X, labels, centers)                # step 2
        if np.allclose(new_centers, centers):                   # step 3: stop when nothing moves
            break
        centers = new_centers
    return centers, labels, sq_dist.sum(), history


def best_of(X, k, init="kmeans++", n_init=5):
    """Different starts can end in different answers, so try several and keep the best."""
    runs = [kmeans(X, k, init, seed=s) for s in range(n_init)]
    return min(runs, key=lambda r: r[2])
