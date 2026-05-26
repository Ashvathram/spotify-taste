"""
analysis.py — Cluster tracks, reduce dimensionality, assign personas.

Reads:  data/tracks.csv        (from preprocess.py)
Writes: data/tracks.csv        (adds cluster, persona, pca_x, pca_y, umap_x, umap_y)
        data/elbow_plot.json   (plotly figure for elbow method)

Clustering:
  - KMeans k=5, random_state=42
  - Features: danceability, energy, valence, tempo_norm, acousticness, instrumentalness

Persona assignment (centroid-based, priority ordered):
  - High instrumentalness → "Focus Zone"
  - Low valence + low energy → "Dark & Moody"
  - High energy + high danceability → "Party Mode"
  - Low energy + high acousticness → "Chill Acoustic"
  - High valence + high tempo_norm → "Feel Good"

Dimensionality reduction:
  - PCA → 2D (fast, always available)
  - UMAP → 2D (higher quality, falls back gracefully if unavailable)

Usage:
    python analysis.py
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

DATA_DIR = "data"
TRACKS_CSV = os.path.join(DATA_DIR, "tracks.csv")
ELBOW_JSON = os.path.join(DATA_DIR, "elbow_plot.json")

CLUSTER_FEATURES = [
    "danceability", "energy", "valence", "tempo_norm",
    "acousticness", "instrumentalness",
]

N_CLUSTERS = 5
RANDOM_STATE = 42

# Persona definitions: (label, emoji, description, conditions)
# Each condition is (feature, direction) where direction is "high" or "low"
# Priority is ORDER in this list — first match wins.
PERSONA_RULES = [
    {
        "label": "Focus Zone",
        "emoji": "🎯",
        "description": "You zone in deep. High instrumentalness tracks dominate — pure concentration fuel.",
        "primary": ("instrumentalness", "high"),
        "secondary": None,
    },
    {
        "label": "Dark & Moody",
        "emoji": "🌑",
        "description": "Brooding and introspective. Low energy and valence — you feel the weight of every note.",
        "primary": ("valence", "low"),
        "secondary": ("energy", "low"),
    },
    {
        "label": "Party Mode",
        "emoji": "🔥",
        "description": "Let's go. High energy meets high danceability — you're always ready to move.",
        "primary": ("energy", "high"),
        "secondary": ("danceability", "high"),
    },
    {
        "label": "Chill Acoustic",
        "emoji": "🌿",
        "description": "Laid-back and organic. Low energy, high acousticness — the soundtrack for quiet moments.",
        "primary": ("energy", "low"),
        "secondary": ("acousticness", "high"),
    },
    {
        "label": "Feel Good",
        "emoji": "☀️",
        "description": "Pure serotonin. High valence and tempo — music that makes life feel lighter.",
        "primary": ("valence", "high"),
        "secondary": ("tempo_norm", "high"),
    },
]


def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def load_tracks(path: str = TRACKS_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Preprocessed data not found at {path}. "
            "Run `python fetch_data.py` then `python preprocess.py` first."
        )
    df = pd.read_csv(path)
    _log(f"Loaded {len(df)} tracks from {path}")
    return df


def elbow_method(X: np.ndarray, k_max: int = 10) -> go.Figure:
    """Compute inertia for k=2..k_max and return a plotly elbow figure."""
    _log("Running elbow method to justify k=5...")
    ks = list(range(2, k_max + 1))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        km.fit(X)
        inertias.append(km.inertia_)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks, y=inertias,
        mode="lines+markers",
        line=dict(color="#1DB954", width=2),
        marker=dict(size=8, color="#1DB954"),
        name="Inertia",
    ))
    fig.add_vline(x=N_CLUSTERS, line_dash="dash", line_color="#ff4d4d",
                  annotation_text=f"k={N_CLUSTERS}", annotation_position="top right")
    fig.update_layout(
        title="KMeans Elbow Method",
        xaxis_title="Number of clusters (k)",
        yaxis_title="Inertia (within-cluster sum of squares)",
        template="plotly_dark",
        paper_bgcolor="#121212",
        plot_bgcolor="#121212",
    )
    _log("  → Elbow figure ready")
    return fig


def fit_kmeans(X: np.ndarray) -> tuple[KMeans, np.ndarray]:
    """Fit KMeans and return (model, labels)."""
    _log(f"Running KMeans with k={N_CLUSTERS}...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init="auto")
    labels = km.fit_predict(X)
    _log(f"  → Cluster sizes: { {i: int((labels==i).sum()) for i in range(N_CLUSTERS)} }")
    return km, labels


def _score_centroid(centroid: np.ndarray, rule: dict, feature_names: list[str]) -> float:
    """
    Score how well a centroid matches a persona rule.
    Returns a float in [0, 2] (primary + optional secondary match).
    """
    def feature_val(name):
        idx = feature_names.index(name)
        return centroid[idx]

    primary_feat, primary_dir = rule["primary"]
    score = feature_val(primary_feat) if primary_dir == "high" else (1 - feature_val(primary_feat))

    if rule["secondary"]:
        sec_feat, sec_dir = rule["secondary"]
        sec_score = feature_val(sec_feat) if sec_dir == "high" else (1 - feature_val(sec_feat))
        score += sec_score

    return score


def assign_personas(km: KMeans, feature_names: list[str]) -> dict[int, dict]:
    """
    Assign one persona per cluster based on centroid values.
    Each persona can only be assigned once (greedy best-match).
    Returns dict: cluster_id → persona info dict.
    """
    _log("Assigning personas to clusters...")
    centroids = km.cluster_centers_  # shape (k, n_features)
    n_clusters = centroids.shape[0]

    # Build score matrix: clusters × personas
    score_matrix = np.zeros((n_clusters, len(PERSONA_RULES)))
    for ci in range(n_clusters):
        for pi, rule in enumerate(PERSONA_RULES):
            score_matrix[ci, pi] = _score_centroid(centroids[ci], rule, feature_names)

    # Greedy assignment: pick highest score, remove both cluster and persona
    cluster_to_persona = {}
    assigned_personas = set()
    assigned_clusters = set()

    # Sort all (cluster, persona) pairs by score descending
    pairs = []
    for ci in range(n_clusters):
        for pi in range(len(PERSONA_RULES)):
            pairs.append((score_matrix[ci, pi], ci, pi))
    pairs.sort(reverse=True)

    for score, ci, pi in pairs:
        if ci in assigned_clusters or pi in assigned_personas:
            continue
        cluster_to_persona[ci] = PERSONA_RULES[pi]
        assigned_clusters.add(ci)
        assigned_personas.add(pi)
        _log(f"  → Cluster {ci} → {PERSONA_RULES[pi]['emoji']} {PERSONA_RULES[pi]['label']} (score={score:.3f})")

    # Fallback: any unassigned cluster gets the remaining persona
    remaining_rules = [r for i, r in enumerate(PERSONA_RULES) if i not in assigned_personas]
    for ci in range(n_clusters):
        if ci not in cluster_to_persona and remaining_rules:
            cluster_to_persona[ci] = remaining_rules.pop(0)

    return cluster_to_persona


def run_pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce to 2D with PCA. Returns (coords, explained_variance_ratio)."""
    _log("Running PCA (2D)...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    _log(f"  → PCA explained variance: {pca.explained_variance_ratio_.round(3)}")
    return coords, pca.explained_variance_ratio_


def run_umap(X: np.ndarray) -> np.ndarray | None:
    """Reduce to 2D with UMAP. Returns None if umap-learn is unavailable."""
    try:
        import umap  # noqa: F401 — local import so failure is graceful
        from umap import UMAP
        _log("Running UMAP (2D) — this may take a moment...")
        reducer = UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1)
        coords = reducer.fit_transform(X)
        _log("  → UMAP done")
        return coords
    except ImportError:
        _log("  ⚠ umap-learn not installed — UMAP step skipped, will use PCA in dashboard")
        return None
    except Exception as e:
        _log(f"  ⚠ UMAP failed ({e}) — falling back to PCA in dashboard")
        return None


def analyze() -> pd.DataFrame:
    """Main entry point."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df = load_tracks()

    # Validate required columns
    missing = [c for c in CLUSTER_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Re-run preprocess.py.")

    X = df[CLUSTER_FEATURES].values.astype(np.float32)

    # Elbow plot (saved for optional display in app)
    elbow_fig = elbow_method(X)
    with open(ELBOW_JSON, "w") as f:
        f.write(elbow_fig.to_json())
    _log(f"Saved elbow plot to {ELBOW_JSON}")

    # KMeans
    km, labels = fit_kmeans(X)
    df["cluster"] = labels

    # Persona assignment
    persona_map = assign_personas(km, CLUSTER_FEATURES)
    df["persona"] = df["cluster"].map(lambda c: persona_map[c]["label"])
    df["persona_emoji"] = df["cluster"].map(lambda c: persona_map[c]["emoji"])
    df["persona_description"] = df["cluster"].map(lambda c: persona_map[c]["description"])

    # PCA
    pca_coords, _ = run_pca(X)
    df["pca_x"] = pca_coords[:, 0]
    df["pca_y"] = pca_coords[:, 1]

    # UMAP (optional)
    umap_coords = run_umap(X)
    if umap_coords is not None:
        df["umap_x"] = umap_coords[:, 0]
        df["umap_y"] = umap_coords[:, 1]
    else:
        # Mirror PCA so downstream code always has umap_x/umap_y
        df["umap_x"] = df["pca_x"]
        df["umap_y"] = df["pca_y"]

    # Save back
    df.to_csv(TRACKS_CSV, index=False)
    _log(f"Saved enriched data (with clusters + embeddings) to {TRACKS_CSV}")

    return df


if __name__ == "__main__":
    print("=" * 50)
    print("  Spotify Music Analyzer")
    print("=" * 50)
    df = analyze()

    print("\nPersona distribution:")
    print(df["persona"].value_counts().to_string())

    print("\nSample:")
    print(df[["track_name", "artist_name", "persona", "energy", "valence"]].head(10).to_string())

    print(f"\n✅ Done! Enriched data saved to {TRACKS_CSV}")
