"""Run: python -m tests.check   (compares our k-means with scikit-learn)"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from kmeans import kmeans, best_of, assign, update

X, y = make_blobs(600, centers=4, random_state=3)
ours = best_of(X, 4, n_init=10)[2]
theirs = KMeans(4, n_init=10, random_state=0).fit(X).inertia_
print("inertia ours vs sklearn:", round(ours, 3), round(theirs, 3), "PASS" if np.isclose(ours, theirs) else "FAIL")

# hand worked example from LEARN.md
P = np.array([[1., 1.], [2., 1.], [8., 8.], [9., 9.]])
C = np.array([[1., 1.], [9., 9.]])
labels, _ = assign(P, C)
print("labels", labels.tolist(), "PASS" if labels.tolist() == [0, 0, 1, 1] else "FAIL")
print("new centres", update(P, labels, C).tolist(), "PASS" if np.allclose(update(P, labels, C), [[1.5, 1], [8.5, 8.5]]) else "FAIL")

_, _, _, hist = kmeans(X, 4, "random")
inert = [h[2] for h in hist]
print("inertia never rises:", "PASS" if np.all(np.diff(inert) <= 1e-9) else "FAIL", f"({len(inert)} iterations)")
