"""
enrich_data.py — Enrich track data with real tags from Last.fm API.

Last.fm is free, open, and has no quota restrictions like Spotify.
This script looks up each track by artist + title, pulls real genre/mood tags,
then re-seeds the synthetic audio features with much more accurate values.

Setup:
  1. Get a free Last.fm API key at https://www.last.fm/api/account/create
  2. Add to your .env:  LASTFM_API_KEY=your_key_here
  3. Run:  python enrich_data.py
  4. Then re-run:  python preprocess.py && python analysis.py

This does NOT require re-running fetch_data.py.
"""

import os
import time
import random
from datetime import datetime

import requests
import pandas as pd
from dotenv import load_dotenv

DATA_DIR = "data"
RAW_CSV = os.path.join(DATA_DIR, "raw_tracks.csv")
ENRICHED_CSV = os.path.join(DATA_DIR, "raw_tracks.csv")  # overwrite in place
CACHE_CSV = os.path.join(DATA_DIR, "lastfm_cache.csv")   # avoid re-fetching
SYNTHETIC_FLAG = os.path.join(DATA_DIR, ".synthetic_features")

LASTFM_API = "https://ws.audioscrobbler.com/2.0/"
REQUEST_DELAY = 0.25  # seconds between requests (Last.fm allows ~5/sec)


def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# ---------------------------------------------------------------------------
# Tag → audio feature mapping
# ---------------------------------------------------------------------------

# These map Last.fm tags (lowercased) to audio feature adjustments.
# Values are applied as overrides to the base genre seed.
TAG_FEATURE_MAP = {
    # Energy modifiers
    "energetic":      dict(energy=+0.20),
    "aggressive":     dict(energy=+0.18, valence=-0.10),
    "intense":        dict(energy=+0.15),
    "calm":           dict(energy=-0.20, acousticness=+0.15),
    "mellow":         dict(energy=-0.15, acousticness=+0.10),
    "soft":           dict(energy=-0.12, acousticness=+0.12),
    "heavy":          dict(energy=+0.15, loudness=+0.10),
    "quiet":          dict(energy=-0.15, loudness=-0.15),
    "loud":           dict(loudness=+0.15),

    # Mood / valence modifiers
    "happy":          dict(valence=+0.20),
    "sad":            dict(valence=-0.20),
    "melancholic":    dict(valence=-0.18, energy=-0.08),
    "dark":           dict(valence=-0.15, energy=-0.05),
    "uplifting":      dict(valence=+0.18),
    "positive":       dict(valence=+0.15),
    "depressing":     dict(valence=-0.20),
    "angry":          dict(valence=-0.15, energy=+0.15),
    "romantic":       dict(valence=+0.10, energy=-0.05),
    "melancholy":     dict(valence=-0.18),
    "heartbreak":     dict(valence=-0.15),
    "bittersweet":    dict(valence=-0.05),

    # Danceability modifiers
    "danceable":      dict(danceability=+0.20),
    "dance":          dict(danceability=+0.18),
    "groovy":         dict(danceability=+0.15),
    "party":          dict(danceability=+0.18, energy=+0.12),
    "club":           dict(danceability=+0.18, energy=+0.15),
    "rhythm and blues": dict(danceability=+0.10),

    # Acousticness modifiers
    "acoustic":       dict(acousticness=+0.30, instrumentalness=-0.05),
    "unplugged":      dict(acousticness=+0.28),
    "live":           dict(liveness=+0.30, acousticness=+0.10),
    "a cappella":     dict(acousticness=+0.25, instrumentalness=-0.10),

    # Instrumentalness modifiers
    "instrumental":   dict(instrumentalness=+0.35, speechiness=-0.05),
    "ambient":        dict(instrumentalness=+0.25, energy=-0.15, acousticness=+0.10),
    "classical":      dict(instrumentalness=+0.40, acousticness=+0.20, energy=-0.10),
    "jazz":           dict(instrumentalness=+0.20, acousticness=+0.15),
    "orchestral":     dict(instrumentalness=+0.35, acousticness=+0.15),

    # Speechiness modifiers
    "rap":            dict(speechiness=+0.25, danceability=+0.10),
    "hip-hop":        dict(speechiness=+0.20, danceability=+0.12),
    "spoken word":    dict(speechiness=+0.35, instrumentalness=-0.10),
    "hip hop":        dict(speechiness=+0.20, danceability=+0.12),

    # Tempo modifiers
    "fast":           dict(tempo=+20),
    "slow":           dict(tempo=-20),
    "upbeat":         dict(tempo=+15, valence=+0.10),
    "downtempo":      dict(tempo=-15, energy=-0.10),

    # Genre seeds (use GENRE_SEEDS from fetch_data but expressed as deltas here)
    "pop":            dict(danceability=+0.05, valence=+0.05),
    "rock":           dict(energy=+0.10),
    "metal":          dict(energy=+0.20, valence=-0.10),
    "punk":           dict(energy=+0.15, tempo=+20),
    "folk":           dict(acousticness=+0.20, energy=-0.10),
    "country":        dict(acousticness=+0.12),
    "blues":          dict(acousticness=+0.10, valence=-0.05),
    "soul":           dict(valence=+0.08, acousticness=+0.08),
    "r&b":            dict(danceability=+0.08, valence=+0.05),
    "electronic":     dict(instrumentalness=+0.15, energy=+0.10),
    "edm":            dict(energy=+0.18, danceability=+0.15),
    "reggae":         dict(valence=+0.12, tempo=-10),
    "latin":          dict(danceability=+0.15, valence=+0.12),
    "k-pop":          dict(danceability=+0.12, energy=+0.10, valence=+0.08),
    "trap":           dict(speechiness=+0.12, tempo=-15),
    "indie":          dict(acousticness=+0.08),
    "alternative":    dict(energy=+0.05),
}

# Base "default" features that tag deltas are applied on top of
BASE_FEATURES = dict(
    danceability=0.60,
    energy=0.60,
    valence=0.55,
    tempo=112,
    acousticness=0.25,
    instrumentalness=0.08,
    speechiness=0.06,
    loudness=0.65,
    liveness=0.13,
)

FEATURE_BOUNDS = dict(
    danceability=(0.0, 1.0),
    energy=(0.0, 1.0),
    valence=(0.0, 1.0),
    tempo=(60, 200),
    acousticness=(0.0, 1.0),
    instrumentalness=(0.0, 1.0),
    speechiness=(0.0, 1.0),
    loudness=(0.0, 1.0),
    liveness=(0.0, 1.0),
)


# ---------------------------------------------------------------------------
# Last.fm API calls
# ---------------------------------------------------------------------------

def get_lastfm_tags(api_key: str, artist: str, title: str) -> list[str]:
    """
    Fetch top tags for a track from Last.fm.
    Returns list of lowercased tag names (up to 10), or [] on failure.
    """
    params = {
        "method": "track.getTopTags",
        "api_key": api_key,
        "artist": artist,
        "track": title,
        "format": "json",
        "autocorrect": 1,
    }
    try:
        resp = requests.get(LASTFM_API, params=params, timeout=8)
        if resp.status_code != 200:
            return []
        data = resp.json()
        if "error" in data:
            return []
        tags = data.get("toptags", {}).get("tag", [])
        # Filter: only tags with count > 10 to avoid noise
        return [t["name"].lower() for t in tags if int(t.get("count", 0)) > 10][:10]
    except Exception:
        return []


def get_artist_tags(api_key: str, artist: str) -> list[str]:
    """
    Fallback: fetch top tags for the artist if track lookup returns nothing.
    """
    params = {
        "method": "artist.getTopTags",
        "api_key": api_key,
        "artist": artist,
        "format": "json",
        "autocorrect": 1,
    }
    try:
        resp = requests.get(LASTFM_API, params=params, timeout=8)
        if resp.status_code != 200:
            return []
        data = resp.json()
        if "error" in data:
            return []
        tags = data.get("toptags", {}).get("tag", [])
        return [t["name"].lower() for t in tags if int(t.get("count", 0)) > 20][:8]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Feature generation from tags
# ---------------------------------------------------------------------------

def features_from_tags(tags: list[str], track_id: str) -> dict:
    """
    Compute audio features by applying tag deltas to the base feature set.
    Small per-track noise ensures tracks aren't identical even with same tags.
    """
    rng = random.Random(hash(track_id))
    features = dict(BASE_FEATURES)

    for tag in tags:
        deltas = TAG_FEATURE_MAP.get(tag, {})
        for feat, delta in deltas.items():
            features[feat] = features.get(feat, 0) + delta

    # Add small noise for variety
    for feat in features:
        lo, hi = FEATURE_BOUNDS[feat]
        if feat == "tempo":
            features[feat] += rng.gauss(0, 4)
        else:
            features[feat] += rng.gauss(0, 0.04)
        features[feat] = max(lo, min(hi, features[feat]))

    return {k: round(v, 4) for k, v in features.items()}


# ---------------------------------------------------------------------------
# Main enrichment logic
# ---------------------------------------------------------------------------

def load_cache() -> dict[str, list[str]]:
    """Load previously fetched tags to avoid redundant API calls."""
    if not os.path.exists(CACHE_CSV):
        return {}
    cache_df = pd.read_csv(CACHE_CSV)
    cache = {}
    for _, row in cache_df.iterrows():
        tags = str(row["tags"]).split("|") if row["tags"] and str(row["tags"]) != "nan" else []
        cache[row["track_id"]] = tags
    _log(f"Loaded {len(cache)} cached tag lookups")
    return cache


def save_cache(cache: dict[str, list[str]]):
    rows = [{"track_id": tid, "tags": "|".join(tags)} for tid, tags in cache.items()]
    pd.DataFrame(rows).to_csv(CACHE_CSV, index=False)


def enrich():
    load_dotenv()
    api_key = os.getenv("LASTFM_API_KEY", "").strip()

    if not api_key:
        print("\n" + "="*55)
        print("  Last.fm API key not found!")
        print("="*55)
        print("\n1. Go to: https://www.last.fm/api/account/create")
        print("2. Fill in: Application name (e.g. 'spotify-taste')")
        print("            Contact email")
        print("            Application website (can be http://localhost)")
        print("3. Click Submit — you'll get an API key instantly")
        print("4. Add this line to your .env file:")
        print("   LASTFM_API_KEY=your_api_key_here")
        print("5. Re-run:  python enrich_data.py\n")
        return

    if not os.path.exists(RAW_CSV):
        _log(f"ERROR: {RAW_CSV} not found. Run python fetch_data.py first.")
        return

    df = pd.read_csv(RAW_CSV)
    _log(f"Loaded {len(df)} rows from {RAW_CSV}")

    # Load cache (avoid re-fetching)
    cache = load_cache()

    # Get unique tracks to look up
    unique_tracks = df.drop_duplicates("track_id")[["track_id", "track_name", "artist_name"]]
    to_fetch = unique_tracks[~unique_tracks["track_id"].isin(cache.keys())]
    _log(f"Need to fetch tags for {len(to_fetch)} new tracks ({len(cache)} already cached)")

    # Fetch tags from Last.fm
    fetched = 0
    failed = 0
    for _, row in to_fetch.iterrows():
        tid = row["track_id"]
        artist = row["artist_name"]
        title = row["track_name"]

        tags = get_lastfm_tags(api_key, artist, title)
        if not tags:
            # Fallback to artist-level tags
            tags = get_artist_tags(api_key, artist)
            if tags:
                _log(f"  ~ {title[:35]} — used artist tags: {tags[:4]}")
            else:
                _log(f"  ✗ {title[:35]} — no tags found")
                failed += 1
        else:
            _log(f"  ✓ {title[:35]} — {tags[:5]}")
            fetched += 1

        cache[tid] = tags
        time.sleep(REQUEST_DELAY)

        # Save cache periodically
        if (fetched + failed) % 20 == 0:
            save_cache(cache)

    save_cache(cache)
    _log(f"Tag fetch complete: {fetched} found, {failed} not found")

    # Generate improved features from tags
    _log("Generating tag-based audio features...")
    feature_cols = list(BASE_FEATURES.keys())

    new_features = []
    for _, row in df.iterrows():
        tid = row["track_id"]
        tags = cache.get(tid, [])
        feats = features_from_tags(tags, tid)
        new_features.append(feats)

    feat_df = pd.DataFrame(new_features, index=df.index)

    # Overwrite feature columns in df
    for col in feature_cols:
        df[col] = feat_df[col]

    # Add tags column for reference
    df["lastfm_tags"] = df["track_id"].map(
        lambda tid: "|".join(cache.get(tid, []))
    )

    df.to_csv(ENRICHED_CSV, index=False)
    _log(f"Saved enriched data to {ENRICHED_CSV}")

    # Update synthetic flag to show improved source
    flag_path = os.path.join(DATA_DIR, ".synthetic_features")
    if os.path.exists(flag_path):
        with open(flag_path, "w") as f:
            f.write("lastfm_enriched")
    _log("Updated synthetic flag → features are now Last.fm tag-enriched")

    # Summary
    tracks_with_tags = sum(1 for tags in cache.values() if tags)
    pct = tracks_with_tags / len(cache) * 100 if cache else 0
    _log(f"\nSummary: {tracks_with_tags}/{len(cache)} tracks ({pct:.0f}%) got real Last.fm tags")
    _log("Now run:  python preprocess.py && python analysis.py")
    _log("Then reload your Streamlit dashboard.")


if __name__ == "__main__":
    print("=" * 55)
    print("  Last.fm Tag Enrichment")
    print("=" * 55)
    enrich()
