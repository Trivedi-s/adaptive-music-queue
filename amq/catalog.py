"""Read-only view of the frozen experiment catalog (data/catalog.csv + data/clusters.csv).

Tracks are addressed by their row index (0..N-1) everywhere in the simulator and agents;
`track_ids[i]` gives the Spotify id when you need it for display.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class Catalog:
    features: np.ndarray          # (N, F) clustering features, all in [0, 1]
    feature_names: tuple
    cluster: np.ndarray           # (N,) style cluster id of each track
    centroids: np.ndarray         # (K, F) cluster centroids mu_z
    popularity: np.ndarray        # (N,) Spotify popularity in [0, 1]
    artist: np.ndarray            # (N,) integer id of main_artist
    track_ids: tuple
    names: tuple
    artist_names: tuple           # artist_names[artist[i]] is the main artist of track i
    nearest_cluster: np.ndarray   # (K,) nearest other cluster (BRIDGE target)
    centroid_dist: np.ndarray     # (K, K) Euclidean distances between centroids
    members: tuple                # members[z] = array of track indices in cluster z

    @property
    def n_tracks(self):
        return len(self.cluster)

    @property
    def n_clusters(self):
        return len(self.centroids)

    @classmethod
    def load(cls, data_dir=DATA):
        data_dir = Path(data_dir)
        tracks = pd.read_csv(data_dir / "catalog.csv")
        clusters = pd.read_csv(data_dir / "clusters.csv").sort_values("cluster")

        # clusters.csv stores the feature columns between `cluster` and `size`,
        # so the feature list is read from the file instead of duplicated here.
        cols = list(clusters.columns)
        feature_names = tuple(cols[cols.index("cluster") + 1: cols.index("size")])

        centroids = clusters[list(feature_names)].to_numpy(dtype=float)
        dist = np.linalg.norm(centroids[:, None] - centroids[None], axis=2)
        artist_codes, artist_names = pd.factorize(tracks["main_artist"])
        cluster = tracks["cluster"].to_numpy()

        return cls(
            features=tracks[list(feature_names)].to_numpy(dtype=float),
            feature_names=feature_names,
            cluster=cluster,
            centroids=centroids,
            popularity=tracks["popularity"].to_numpy(dtype=float) / 100.0,
            artist=artist_codes,
            track_ids=tuple(tracks["track_id"]),
            names=tuple(tracks["name"]),
            artist_names=tuple(artist_names),
            nearest_cluster=clusters["nearest_cluster"].to_numpy(),
            centroid_dist=dist,
            members=tuple(np.flatnonzero(cluster == z) for z in range(len(centroids))),
        )

    def similarity(self, tracks, z):
        """sim(x, mu_z) = 1 - ||x - mu_z||. About 0.7 for a typical track in z, about 0.1 elsewhere."""
        return 1.0 - np.linalg.norm(self.features[tracks] - self.centroids[z], axis=-1)

    def describe(self, i):
        return f"{self.names[i]} - {self.artist_names[self.artist[i]]} (cluster {self.cluster[i]})"
