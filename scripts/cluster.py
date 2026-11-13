"""Group the tracks in data/data.csv into K style clusters with k-means.

The clusters are the hidden state z of the listener HMM: the "style" a listener
wants right now. See README "Dataset" and "AI Formulation".

Inputs:
  data/data.csv       track sample produced by scripts/get_data.py
Outputs:
  data/catalog.csv    data.csv plus a `cluster` column (the frozen experiment catalog)
  data/clusters.csv   one row per cluster: centroid, size, name, nearest neighbor
  figures/clusters.png  PCA scatter and genre-by-cluster heatmap

Usage:
  python scripts/cluster.py            # fit K clusters and write the outputs
  python scripts/cluster.py --sweep    # also print model-selection stats for K = 5..14
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

SEED = 0
K = 8

# Features that describe musical *style*. They are all already in [0, 1].
# Left out on purpose:
#   liveness  - detects an audience in the recording, not the style of the song
#   loudness  - r = 0.83 with energy, so it would count the energy axis twice
CLUSTER_FEATURES = [
    "danceability", "energy", "valence", "acousticness",
    "instrumentalness", "speechiness", "tempo",
]


def fit(X, k, seed=SEED):
    return KMeans(n_clusters=k, n_init=50, random_state=seed).fit(X)


def sweep(X, genres):
    """Print the stats used to choose K."""
    print(f"{'K':>3} {'silhouette':>10} {'stability':>9} {'NMI(genre)':>10}  sizes")
    for k in range(5, 15):
        runs = [KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(X) for s in range(8)]
        stability = np.mean([adjusted_rand_score(runs[i], runs[j])
                             for i in range(len(runs)) for j in range(i + 1, len(runs))])
        sizes = sorted(np.bincount(runs[0]).tolist())
        print(f"{k:>3} {silhouette_score(X, runs[0]):>10.3f} {stability:>9.2f} "
              f"{normalized_mutual_info_score(genres, runs[0]):>10.3f}  {sizes}")


def describe(centroid, overall_mean, overall_std, n=3):
    """Name a cluster by the features where its centroid differs most from average."""
    z = (centroid - overall_mean) / overall_std
    top = np.argsort(-np.abs(z))[:n]
    return ", ".join(f"{'high' if z[i] > 0 else 'low'} {CLUSTER_FEATURES[i]}" for i in top)


def plot(X, labels, centroids, data, path):
    pca = PCA(n_components=2, random_state=SEED).fit(X)
    P, C = pca.transform(X), pca.transform(centroids)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    cmap = plt.get_cmap("tab10")
    for c in range(len(centroids)):
        m = labels == c
        ax1.scatter(P[m, 0], P[m, 1], s=6, alpha=0.5, color=cmap(c), label=f"cluster {c}")
        ax1.annotate(str(c), C[c], fontsize=13, weight="bold", ha="center", va="center",
                     bbox=dict(boxstyle="circle", fc="white", ec=cmap(c)))
    ev = pca.explained_variance_ratio_
    ax1.set(title="Style clusters (PCA of clustering features)",
            xlabel=f"PC1 ({ev[0]:.0%} var)", ylabel=f"PC2 ({ev[1]:.0%} var)")
    ax1.legend(markerscale=3, fontsize=8, loc="best")

    tab = pd.crosstab(data["sample_genre"], labels)
    share = tab.div(tab.sum(axis=1), axis=0)
    im = ax2.imshow(share.values, cmap="Blues", vmin=0, vmax=share.values.max(), aspect="auto")
    ax2.set_xticks(range(tab.shape[1]), [str(c) for c in tab.columns])
    ax2.set_yticks(range(tab.shape[0]), tab.index)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            if v:
                ax2.text(j, i, v, ha="center", va="center", fontsize=7,
                         color="white" if share.values[i, j] > 0.4 else "black")
    ax2.set(title="Tracks per genre in each cluster", xlabel="cluster", ylabel="sample genre")
    fig.colorbar(im, ax=ax2, label="share of genre")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--sweep", action="store_true", help="print stats for K = 5..14")
    args = parser.parse_args()

    data = pd.read_csv(DATA / "data.csv")
    X = data[CLUSTER_FEATURES].to_numpy()

    if args.sweep:
        sweep(X, data["sample_genre"])
        print()

    km = fit(X, args.k)

    # k-means numbers clusters arbitrarily. Renumber them by energy (calm -> intense)
    # so cluster ids stay readable and stable if the fit is rerun.
    order = np.argsort(km.cluster_centers_[:, CLUSTER_FEATURES.index("energy")])
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    labels = remap[km.labels_]
    centroids = km.cluster_centers_[order]

    data["cluster"] = labels
    data.to_csv(DATA / "catalog.csv", index=False)

    mean, std = X.mean(axis=0), X.std(axis=0)
    dist = np.linalg.norm(centroids[:, None] - centroids[None], axis=2)
    np.fill_diagonal(dist, np.inf)
    clusters = pd.DataFrame(centroids, columns=CLUSTER_FEATURES).round(4)
    clusters.insert(0, "cluster", range(len(centroids)))
    clusters["size"] = np.bincount(labels)
    clusters["artists"] = data.groupby("cluster")["main_artist"].nunique().to_numpy()
    clusters["mean_popularity"] = data.groupby("cluster")["popularity"].mean().round(1).to_numpy()
    clusters["top_genres"] = [
        ", ".join(data.loc[labels == c, "sample_genre"].value_counts().index[:3])
        for c in range(len(centroids))
    ]
    clusters["description"] = [describe(c, mean, std) for c in centroids]
    clusters["nearest_cluster"] = dist.argmin(axis=1)   # target of the BRIDGE action
    clusters["nearest_distance"] = dist.min(axis=1).round(4)
    clusters.to_csv(DATA / "clusters.csv", index=False)

    FIGURES.mkdir(exist_ok=True)
    plot(X, labels, centroids, data, FIGURES / "clusters.png")

    print(f"K={args.k}  silhouette={silhouette_score(X, labels):.3f}  "
          f"NMI(genre)={normalized_mutual_info_score(data['sample_genre'], labels):.3f}")
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(clusters[["cluster", "size", "artists", "mean_popularity", "top_genres",
                        "description", "nearest_cluster"]].to_string(index=False))
    print("\nSaved data/catalog.csv, data/clusters.csv, figures/clusters.png")


if __name__ == "__main__":
    main()
