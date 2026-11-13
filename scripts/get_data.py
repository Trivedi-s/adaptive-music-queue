"""Build the track dataset from the Spotify Tracks Dataset.

Output:
  data/data.csv  2,000-track sample used for experiments (see README "Dataset")
"""
from pathlib import Path

import pandas as pd
from datasets import load_dataset

SEED = 50
OUT = Path(__file__).resolve().parents[1] / "data" / "data.csv"

FEATURES = [
    "danceability", "energy", "valence", "acousticness",
    "instrumentalness", "speechiness", "liveness", "tempo", "loudness",
]
EXTRA = ["album_name", "duration_ms", "explicit", "key", "mode", "time_signature"]

# Genres that are mostly not music, or out of scope for this project
EXCLUDED_GENRES = {"comedy", "sleep", "kids", "children"}

MIN_DURATION_MS = 60_000      # 1 min
MAX_DURATION_MS = 600_000     # 10 min
MAX_SPEECHINESS = 0.66        # above this Spotify says the track is spoken word

SAMPLE_GENRES = [
    "acoustic", "ambient", "classical", "country", "edm", "hip-hop",
    "jazz", "latin", "metal", "pop", "r-n-b", "rock",
    "techno", "trip-hop", "opera",
]
SAMPLE_SIZE = 2000


df = load_dataset("maharshipandya/spotify-tracks-dataset", split="train").to_pandas()
print(f"Raw: {len(df)} rows")

# --- Row-level cleaning -----------------------------------------------------
df = df.dropna(subset=["track_name", "artists", "track_genre"] + FEATURES)
df = df[df["duration_ms"].between(MIN_DURATION_MS, MAX_DURATION_MS)]
df = df[df["tempo"] > 0]                       # tempo 0 = missing value
df = df[df["speechiness"] <= MAX_SPEECHINESS]  # podcasts, stand-up, audiobooks
print(f"After row filters: {len(df)} rows")

# --- Deduplicate ------------------------------------------------------------
# The same song can appear under several track_ids (single vs album release)
# and the same track_id under several genres. Group by normalized name+artist,
# keep the most popular release, and merge every genre label it had.
df["song_key"] = (
    df["track_name"].str.lower().str.strip() + "||" + df["artists"].str.lower().str.strip()
)
genres = df.groupby("song_key")["track_genre"].agg(lambda s: sorted(set(s)))

data = (
    df.sort_values("popularity", ascending=False)
      .drop_duplicates("song_key")
      .set_index("song_key")
)
data["genres"] = genres
data = data.reset_index(drop=True)
print(f"After dedup: {len(data)} unique songs")

# Drop anything tagged with an excluded genre
data = data[data["genres"].apply(lambda g: not EXCLUDED_GENRES & set(g))]
print(f"After genre exclusions: {len(data)} songs")

# --- Columns ----------------------------------------------------------------
data = data.rename(columns={"track_name": "name", "artists": "artist"})
data["main_artist"] = data["artist"].str.split(";").str[0].str.strip()

# Scale tempo and loudness to 0-1 using the 1st-99th percentiles, so a few
# extreme tracks don't squash everyone else into a narrow range.
for col in ["tempo", "loudness"]:
    lo, hi = data[col].quantile([0.01, 0.99])
    data[col] = ((data[col] - lo) / (hi - lo)).clip(0, 1)

data["genres"] = data["genres"].apply("|".join)
data = data[
    ["track_id", "name", "artist", "main_artist", "popularity", "genres"] + FEATURES + EXTRA
].reset_index(drop=True)
print(f"Cleaned: {len(data)} tracks")

# --- Sample -----------------------------------------------------------------
# Split SAMPLE_SIZE across the genres, then within each genre split evenly
# across popularity terciles, so each genre has both familiar and obscure
# tracks. A track that belongs to several sample genres is only picked once.
base, rem = divmod(SAMPLE_SIZE, len(SAMPLE_GENRES))
picked = []
taken = set()
for i, genre in enumerate(SAMPLE_GENRES):
    per_genre = base + (1 if i < rem else 0)
    pool = data[
        data["genres"].str.split("|").apply(lambda g: genre in g)
        & ~data["track_id"].isin(taken)
    ].copy()
    # rank first so ties (e.g. many popularity-0 tracks) still split into 3 bins
    pool["tercile"] = pd.qcut(pool["popularity"].rank(method="first"), 3, labels=False)
    per_bin, extra = divmod(per_genre, 3)
    for t in range(3):
        n = per_bin + (1 if t < extra else 0)
        chosen = pool[pool["tercile"] == t].sample(n=n, random_state=SEED)
        chosen = chosen.drop(columns="tercile").assign(sample_genre=genre)
        picked.append(chosen)
        taken.update(chosen["track_id"])

sample = pd.concat(picked, ignore_index=True)
OUT.parent.mkdir(exist_ok=True)
sample.to_csv(OUT, index=False)
print(f"Saved {OUT}: {len(sample)} tracks across {sample['sample_genre'].nunique()} genres")
