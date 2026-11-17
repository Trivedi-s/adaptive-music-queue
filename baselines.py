"""Selection rules for baseline recommenders."""


def choose_track(scores, played_ids):
    """Return the highest-scoring unplayed track ID, or None if none remain."""
    candidates = [track_id for track_id in scores if track_id not in played_ids]
    if not candidates:
        return None
    return max(candidates, key=scores.get)


def squared_euclidean_distance(reference, candidate):
    """Return the sum of squared differences between aligned features."""
    if len(reference) != len(candidate):
        raise ValueError("Track feature vectors must have the same length.")
    return sum((a - b) ** 2 for a, b in zip(reference, candidate))


def choose_similar_track(reference, features_by_id, played_ids):
    """Choose the closest unplayed track to the last completed track."""
    scores = {
        track_id: -squared_euclidean_distance(reference, features)
        for track_id, features in features_by_id.items()
        if track_id not in played_ids
    }
    return choose_track(scores, played_ids)


def content_similarity(env):
    """Recommend an unplayed catalog row near the last positive response."""
    played = set(env.played)
    reference = next(
        (track for track, feedback in reversed(env.history) if feedback.is_positive),
        None,
    )
    features = env.catalog.features
    if reference is None:
        return next((i for i in range(len(features)) if i not in played), None)
    return choose_similar_track(features[reference], dict(enumerate(features)), played)


def skip_streak(env, k=2):
    """Switch style after k consecutive skips within the current style."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    if not env.history:
        return content_similarity(env)

    catalog = env.catalog
    last_track = env.history[-1][0]
    current = catalog.cluster[last_track]
    streak = 0
    for track, feedback in reversed(env.history):
        if not feedback.is_skip or catalog.cluster[track] != current:
            break
        streak += 1

    played = set(env.played)
    switch = streak >= k
    clusters = sorted(
        range(len(catalog.members)), key=lambda z: catalog.centroid_dist[current, z]
    )
    other_clusters = [z for z in clusters if z != current]
    order = other_clusters + [current] if switch else [current] + other_clusters
    for cluster in order:
        candidates = {int(i): catalog.features[i] for i in catalog.members[cluster]}
        choice = choose_similar_track(catalog.features[last_track], candidates, played)
        if choice is not None:
            return choice
    return None
