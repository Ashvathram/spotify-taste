"""
app.py — 🎵 Spotify Music DNA  (Streamlit Cloud deployable, multi-user)

All pipeline logic (fetch → preprocess → cluster → visualise) runs in-memory
per session. No files are written to disk. OAuth tokens are isolated in
st.session_state, so concurrent users never collide.

Deploy steps:
  1. Push this single file + requirements.txt to a GitHub repo.
  2. Connect repo to Streamlit Community Cloud.
  3. Add secrets in the Streamlit dashboard:
       SPOTIFY_CLIENT_ID     = "..."
       SPOTIFY_CLIENT_SECRET = "..."
  4. Set the Redirect URI in your Spotify Developer Dashboard to:
       https://<your-app>.streamlit.app/
  5. Done — share the URL.
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import os
import random
import time
import urllib.parse
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# ── Constants ──────────────────────────────────────────────────────────────────
SCOPE = "user-top-read user-read-recently-played"
TIME_RANGES = ["short_term", "medium_term", "long_term"]

AUDIO_FEATURE_KEYS = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "speechiness",
    "loudness", "liveness",
]
CLUSTER_FEATURES = [
    "danceability", "energy", "valence", "tempo_norm",
    "acousticness", "instrumentalness",
]
AUDIO_FEATURES_RADAR = [
    "danceability", "energy", "valence",
    "acousticness", "instrumentalness", "speechiness", "liveness",
]

PERSONA_COLORS = {
    "Party Mode":    "#FF4D4D",
    "Chill Acoustic":"#4DC9FF",
    "Feel Good":     "#FFD700",
    "Dark & Moody":  "#9B59B6",
    "Focus Zone":    "#1DB954",
}
PERSONA_RULES = [
    {"label": "Focus Zone",    "emoji": "🎯", "description": "You zone in deep. High instrumentalness tracks dominate — pure concentration fuel.",          "primary": ("instrumentalness","high"), "secondary": None},
    {"label": "Dark & Moody",  "emoji": "🌑", "description": "Brooding and introspective. Low energy and valence — you feel the weight of every note.",     "primary": ("valence","low"),          "secondary": ("energy","low")},
    {"label": "Party Mode",    "emoji": "🔥", "description": "Let's go. High energy meets high danceability — you're always ready to move.",                 "primary": ("energy","high"),          "secondary": ("danceability","high")},
    {"label": "Chill Acoustic","emoji": "🌿", "description": "Laid-back and organic. Low energy, high acousticness — the soundtrack for quiet moments.",    "primary": ("energy","low"),           "secondary": ("acousticness","high")},
    {"label": "Feel Good",     "emoji": "☀️", "description": "Pure serotonin. High valence and tempo — music that makes life feel lighter.",                 "primary": ("valence","high"),         "secondary": ("tempo_norm","high")},
]

GENRE_SEEDS = {
    "pop":         dict(danceability=0.72,energy=0.68,valence=0.65,tempo=118,acousticness=0.18,instrumentalness=0.02,speechiness=0.06,loudness=0.70,liveness=0.14),
    "hip hop":     dict(danceability=0.80,energy=0.70,valence=0.55,tempo=96, acousticness=0.12,instrumentalness=0.04,speechiness=0.22,loudness=0.72,liveness=0.12),
    "rap":         dict(danceability=0.78,energy=0.72,valence=0.50,tempo=98, acousticness=0.10,instrumentalness=0.03,speechiness=0.26,loudness=0.73,liveness=0.13),
    "r&b":         dict(danceability=0.74,energy=0.58,valence=0.58,tempo=100,acousticness=0.25,instrumentalness=0.03,speechiness=0.08,loudness=0.65,liveness=0.11),
    "soul":        dict(danceability=0.66,energy=0.55,valence=0.60,tempo=95, acousticness=0.40,instrumentalness=0.04,speechiness=0.06,loudness=0.60,liveness=0.16),
    "rock":        dict(danceability=0.54,energy=0.82,valence=0.50,tempo=128,acousticness=0.10,instrumentalness=0.08,speechiness=0.05,loudness=0.82,liveness=0.18),
    "metal":       dict(danceability=0.40,energy=0.92,valence=0.30,tempo=148,acousticness=0.05,instrumentalness=0.20,speechiness=0.05,loudness=0.90,liveness=0.16),
    "indie":       dict(danceability=0.58,energy=0.60,valence=0.52,tempo=118,acousticness=0.28,instrumentalness=0.10,speechiness=0.05,loudness=0.62,liveness=0.14),
    "folk":        dict(danceability=0.48,energy=0.40,valence=0.55,tempo=100,acousticness=0.72,instrumentalness=0.06,speechiness=0.04,loudness=0.48,liveness=0.12),
    "acoustic":    dict(danceability=0.50,energy=0.38,valence=0.54,tempo=96, acousticness=0.80,instrumentalness=0.05,speechiness=0.04,loudness=0.45,liveness=0.11),
    "jazz":        dict(danceability=0.58,energy=0.42,valence=0.60,tempo=112,acousticness=0.60,instrumentalness=0.30,speechiness=0.04,loudness=0.52,liveness=0.20),
    "classical":   dict(danceability=0.28,energy=0.28,valence=0.42,tempo=108,acousticness=0.90,instrumentalness=0.82,speechiness=0.03,loudness=0.38,liveness=0.10),
    "electronic":  dict(danceability=0.76,energy=0.80,valence=0.55,tempo=128,acousticness=0.05,instrumentalness=0.55,speechiness=0.05,loudness=0.78,liveness=0.10),
    "edm":         dict(danceability=0.80,energy=0.88,valence=0.60,tempo=130,acousticness=0.04,instrumentalness=0.60,speechiness=0.04,loudness=0.82,liveness=0.09),
    "dance":       dict(danceability=0.82,energy=0.82,valence=0.65,tempo=124,acousticness=0.06,instrumentalness=0.25,speechiness=0.05,loudness=0.78,liveness=0.11),
    "latin":       dict(danceability=0.84,energy=0.76,valence=0.74,tempo=110,acousticness=0.22,instrumentalness=0.04,speechiness=0.07,loudness=0.72,liveness=0.15),
    "reggae":      dict(danceability=0.78,energy=0.60,valence=0.72,tempo=90, acousticness=0.30,instrumentalness=0.06,speechiness=0.08,loudness=0.62,liveness=0.14),
    "country":     dict(danceability=0.60,energy=0.62,valence=0.65,tempo=120,acousticness=0.45,instrumentalness=0.02,speechiness=0.04,loudness=0.65,liveness=0.14),
    "blues":       dict(danceability=0.54,energy=0.52,valence=0.48,tempo=104,acousticness=0.48,instrumentalness=0.10,speechiness=0.04,loudness=0.58,liveness=0.18),
    "punk":        dict(danceability=0.52,energy=0.88,valence=0.52,tempo=158,acousticness=0.05,instrumentalness=0.04,speechiness=0.08,loudness=0.86,liveness=0.22),
    "ambient":     dict(danceability=0.30,energy=0.22,valence=0.35,tempo=88, acousticness=0.65,instrumentalness=0.75,speechiness=0.03,loudness=0.30,liveness=0.08),
    "k-pop":       dict(danceability=0.78,energy=0.76,valence=0.68,tempo=122,acousticness=0.12,instrumentalness=0.04,speechiness=0.08,loudness=0.74,liveness=0.12),
    "trap":        dict(danceability=0.76,energy=0.68,valence=0.42,tempo=72, acousticness=0.08,instrumentalness=0.15,speechiness=0.18,loudness=0.75,liveness=0.10),
    "default":     dict(danceability=0.60,energy=0.60,valence=0.55,tempo=112,acousticness=0.25,instrumentalness=0.08,speechiness=0.06,loudness=0.65,liveness=0.13),
}

BG_COLOR  = "#0D0D0D"
CARD_BG   = "#1A1A1A"
PLOTLY_TEMPLATE = "plotly_dark"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Music DNA",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0D0D0D;
    color: #EFEFEF;
}
h1,h2,h3 { font-family: 'Space Mono', monospace; }

.stat-card {
    background:#1A1A1A; border:1px solid #2A2A2A; border-radius:12px;
    padding:20px 24px; text-align:center; transition:border-color .2s;
}
.stat-card:hover { border-color:#1DB954; }
.stat-value {
    font-family:'Space Mono',monospace; font-size:2.2rem;
    font-weight:700; color:#1DB954; margin:4px 0;
}
.stat-label {
    font-size:.78rem; text-transform:uppercase;
    letter-spacing:.12em; color:#888;
}
.persona-card {
    background:linear-gradient(135deg,#1A1A1A 0%,#111 100%);
    border:2px solid #1DB954; border-radius:16px;
    padding:28px 32px; margin-top:16px;
}
.section-header {
    font-family:'Space Mono',monospace; font-size:.72rem;
    letter-spacing:.18em; text-transform:uppercase;
    color:#1DB954; margin-bottom:4px;
}
div[data-testid="stSidebar"] { background-color:#111; border-right:1px solid #222; }

/* Login page */
.login-box {
    max-width:480px; margin:80px auto; text-align:center;
    background:#1A1A1A; border:1px solid #2A2A2A;
    border-radius:20px; padding:48px 40px;
}
.spotify-btn {
    display:inline-block; background:#1DB954; color:#000 !important;
    font-weight:700; font-family:'Space Mono',monospace;
    padding:14px 36px; border-radius:50px; text-decoration:none;
    font-size:1rem; transition:background .2s; margin-top:24px;
}
.spotify-btn:hover { background:#1ed760; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# AUTH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_oauth() -> SpotifyOAuth:
    """Create a SpotifyOAuth manager from st.secrets (Streamlit Cloud)
    or environment variables (local dev)."""
    client_id     = st.secrets.get("SPOTIFY_CLIENT_ID",     os.getenv("SPOTIFY_CLIENT_ID", ""))
    client_secret = st.secrets.get("SPOTIFY_CLIENT_SECRET", os.getenv("SPOTIFY_CLIENT_SECRET", ""))
    redirect_uri  = st.secrets.get("SPOTIFY_REDIRECT_URI",  os.getenv("SPOTIFY_REDIRECT_URI",
                                    "http://localhost:8501/"))
    return SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=SCOPE,
        cache_handler=spotipy.cache_handler.MemoryCacheHandler(),
        open_browser=False,
        show_dialog=False,
    )


def handle_oauth_callback() -> bool:
    """
    Check if Spotify redirected back with ?code=... in the URL.
    If so, exchange for tokens and store in session_state.
    Returns True if tokens were successfully obtained.
    """
    params = st.query_params
    code  = params.get("code")
    error = params.get("error")

    if error:
        st.error(f"Spotify auth denied: {error}")
        return False

    if code and "token_info" not in st.session_state:
        oauth = _get_oauth()
        try:
            token_info = oauth.get_access_token(code, as_dict=True)
            st.session_state["token_info"] = token_info
            # Clear the code from URL so refresh doesn't re-trigger
            st.query_params.clear()
            return True
        except Exception as e:
            st.error(f"Token exchange failed: {e}")
            return False

    return "token_info" in st.session_state


def get_spotify_client() -> spotipy.Spotify | None:
    """Return an authenticated Spotify client from session tokens, refreshing if needed."""
    if "token_info" not in st.session_state:
        return None
    oauth = _get_oauth()
    token_info = st.session_state["token_info"]
    # Auto-refresh if expired
    if oauth.is_token_expired(token_info):
        try:
            token_info = oauth.refresh_access_token(token_info["refresh_token"])
            st.session_state["token_info"] = token_info
        except Exception:
            del st.session_state["token_info"]
            return None
    return spotipy.Spotify(auth=token_info["access_token"])


def render_login():
    """Show the login page with a Spotify connect button."""
    oauth = _get_oauth()
    auth_url = oauth.get_authorize_url()

    st.markdown(f"""
    <div class="login-box">
        <div style="font-size:3rem">🎵</div>
        <h1 style="color:#1DB954;font-size:1.8rem;margin:12px 0 4px">Music DNA</h1>
        <p style="color:#888;font-size:.95rem;margin-bottom:0">
            Decode your Spotify listening history.<br>
            Mood clusters · Taste fingerprint · Genre trends.
        </p>
        <a class="spotify-btn" href="{auth_url}">Connect with Spotify</a>
        <p style="color:#444;font-size:.72rem;margin-top:24px">
            Read-only access · Your data stays in your session
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE — all in-memory, no disk I/O
# ══════════════════════════════════════════════════════════════════════════════

def _genre_seed(genres: list[str]) -> dict:
    matched = []
    for g in genres:
        g_lower = g.lower()
        for key in GENRE_SEEDS:
            if key in g_lower:
                matched.append(GENRE_SEEDS[key])
                break
    if not matched:
        return GENRE_SEEDS["default"]
    if len(matched) == 1:
        return matched[0]
    a, b = matched[0], matched[1]
    return {k: a[k] * 0.6 + b[k] * 0.4 for k in a}


def generate_synthetic_features(all_tracks: list[dict], genres_map: dict) -> dict:
    features_map = {}
    for t in all_tracks:
        tid = t["track_id"]
        if tid in features_map:
            continue
        genres = genres_map.get(t["artist_id"], [])
        seed = _genre_seed(genres)
        rng = random.Random(hash(tid))
        noise = lambda: rng.gauss(0, 0.06)
        features_map[tid] = {
            "danceability":    max(0., min(1., seed["danceability"]    + noise())),
            "energy":          max(0., min(1., seed["energy"]          + noise())),
            "valence":         max(0., min(1., seed["valence"]         + noise())),
            "tempo":           max(60, min(200, seed["tempo"]          + rng.gauss(0, 8))),
            "acousticness":    max(0., min(1., seed["acousticness"]    + noise())),
            "instrumentalness":max(0., min(1., seed["instrumentalness"]+ noise())),
            "speechiness":     max(0., min(1., seed["speechiness"]     + noise())),
            "loudness":        max(0., min(1., seed["loudness"]        + noise())),
            "liveness":        max(0., min(1., seed["liveness"]        + noise())),
        }
    return features_map


def run_pipeline(sp: spotipy.Spotify) -> tuple[pd.DataFrame, bool]:
    """
    Full pipeline: fetch → preprocess → cluster → reduce.
    Returns (dataframe, is_synthetic).
    """

    # ── 1. Fetch top tracks ──────────────────────────────────────────────────
    all_tracks = []
    for tr in TIME_RANGES:
        results = sp.current_user_top_tracks(limit=50, time_range=tr)
        for item in results["items"]:
            all_tracks.append({
                "track_id":   item["id"],
                "track_name": item["name"],
                "artist_id":  item["artists"][0]["id"],
                "artist_name":item["artists"][0]["name"],
                "album_name": item["album"]["name"],
                "popularity": item.get("popularity", 0),
                "duration_ms":item.get("duration_ms", 0),
                "explicit":   item.get("explicit", False),
                "time_range": tr,
                "added_at":   None,
            })

    # ── 2. Fetch recently played ─────────────────────────────────────────────
    try:
        results = sp.current_user_recently_played(limit=50)
        for item in results["items"]:
            t = item["track"]
            all_tracks.append({
                "track_id":   t["id"],
                "track_name": t["name"],
                "artist_id":  t["artists"][0]["id"],
                "artist_name":t["artists"][0]["name"],
                "album_name": t["album"]["name"],
                "popularity": t.get("popularity", 0),
                "duration_ms":t.get("duration_ms", 0),
                "explicit":   t.get("explicit", False),
                "time_range": "recently_played",
                "added_at":   item["played_at"],
            })
    except Exception:
        pass

    unique_track_ids  = list({t["track_id"]  for t in all_tracks})
    unique_artist_ids = list({t["artist_id"] for t in all_tracks})

    # ── 3. Fetch artist genres ───────────────────────────────────────────────
    genres_map = {}
    try:
        for i in range(0, len(unique_artist_ids), 50):
            batch = unique_artist_ids[i:i+50]
            results = sp.artists(batch)
            for artist in results["artists"]:
                if artist:
                    genres_map[artist["id"]] = artist.get("genres", [])
            time.sleep(0.05)
    except Exception:
        pass

    # ── 4. Audio features (real → synthetic fallback) ────────────────────────
    features_map: dict = {}
    is_synthetic = False
    try:
        for i in range(0, len(unique_track_ids), 100):
            batch = unique_track_ids[i:i+100]
            results = sp.audio_features(batch)
            for feat in (results or []):
                if feat:
                    features_map[feat["id"]] = {k: feat.get(k) for k in AUDIO_FEATURE_KEYS}
            time.sleep(0.05)
    except spotipy.exceptions.SpotifyException as e:
        if e.http_status == 403:
            is_synthetic = True
        else:
            raise

    if not features_map or is_synthetic:
        is_synthetic = True
        features_map = generate_synthetic_features(all_tracks, genres_map)

    # ── 5. Build flat DataFrame ──────────────────────────────────────────────
    rows = []
    for t in all_tracks:
        tid  = t["track_id"]
        feat = features_map.get(tid)
        if feat is None:
            continue
        genres     = genres_map.get(t["artist_id"], [])
        genres_str = ", ".join(genres) if genres else "Unknown"
        row = {
            "track_id":   tid,
            "track_name": t["track_name"],
            "artist_name":t["artist_name"],
            "artist_id":  t["artist_id"],
            "album_name": t["album_name"],
            "popularity": t.get("popularity", 0),
            "duration_ms":t.get("duration_ms", 0),
            "explicit":   t.get("explicit", False),
            "genres":     genres_str,
            "time_range": t["time_range"],
            "added_at":   t["added_at"],
        }
        row.update(feat)
        rows.append(row)

    df = pd.DataFrame(rows)

    # ── 6. Preprocess ────────────────────────────────────────────────────────
    df = df.dropna(subset=AUDIO_FEATURE_KEYS)
    df["genres"]  = df["genres"].fillna("Unknown")
    df["added_at"]= pd.to_datetime(df["added_at"], errors="coerce", utc=True)
    df["popularity"]= pd.to_numeric(df["popularity"], errors="coerce").fillna(0).astype(int)

    scaler = MinMaxScaler()
    norm_vals = scaler.fit_transform(df[AUDIO_FEATURE_KEYS])
    norm_df   = pd.DataFrame(norm_vals, columns=AUDIO_FEATURE_KEYS, index=df.index)
    df["tempo_norm"]    = norm_df["tempo"]
    df["loudness_norm"] = norm_df["loudness"]
    for col in AUDIO_FEATURE_KEYS:
        df[col] = norm_df[col]

    # ── 7. KMeans clustering ─────────────────────────────────────────────────
    X = df[CLUSTER_FEATURES].values.astype(np.float32)
    km     = KMeans(n_clusters=5, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    df["cluster"] = labels

    # ── 8. Persona assignment ─────────────────────────────────────────────────
    centroids = km.cluster_centers_
    n_clusters = centroids.shape[0]
    feat_names = CLUSTER_FEATURES

    def score_centroid(centroid, rule):
        def fval(name):
            return centroid[feat_names.index(name)]
        pf, pd_ = rule["primary"]
        s = fval(pf) if pd_ == "high" else (1 - fval(pf))
        if rule["secondary"]:
            sf, sd = rule["secondary"]
            s += fval(sf) if sd == "high" else (1 - fval(sf))
        return s

    score_matrix = np.zeros((n_clusters, len(PERSONA_RULES)))
    for ci in range(n_clusters):
        for pi, rule in enumerate(PERSONA_RULES):
            score_matrix[ci, pi] = score_centroid(centroids[ci], rule)

    pairs = sorted(
        [(score_matrix[ci, pi], ci, pi)
         for ci in range(n_clusters) for pi in range(len(PERSONA_RULES))],
        reverse=True
    )
    cluster_to_persona = {}
    assigned_p = set(); assigned_c = set()
    for score, ci, pi in pairs:
        if ci in assigned_c or pi in assigned_p:
            continue
        cluster_to_persona[ci] = PERSONA_RULES[pi]
        assigned_c.add(ci); assigned_p.add(pi)

    df["persona"]             = df["cluster"].map(lambda c: cluster_to_persona[c]["label"])
    df["persona_emoji"]       = df["cluster"].map(lambda c: cluster_to_persona[c]["emoji"])
    df["persona_description"] = df["cluster"].map(lambda c: cluster_to_persona[c]["description"])

    # ── 9. Dimensionality reduction ──────────────────────────────────────────
    pca   = PCA(n_components=2, random_state=42)
    pca_c = pca.fit_transform(X)
    df["pca_x"] = pca_c[:, 0]
    df["pca_y"] = pca_c[:, 1]

    try:
        from umap import UMAP
        umap_c = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1).fit_transform(X)
        df["umap_x"] = umap_c[:, 0]
        df["umap_y"] = umap_c[:, 1]
    except Exception:
        df["umap_x"] = df["pca_x"]
        df["umap_y"] = df["pca_y"]

    return df.reset_index(drop=True), is_synthetic


# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS  (identical to original, but reading from in-memory df)
# ══════════════════════════════════════════════════════════════════════════════

def make_scatter(df: pd.DataFrame) -> go.Figure:
    has_umap = "umap_x" in df.columns and df["umap_x"].notna().any()
    x_col = "umap_x" if has_umap else "pca_x"
    y_col = "umap_y" if has_umap else "pca_y"
    method = "UMAP" if has_umap else "PCA"
    fig = go.Figure()
    for persona, group in df.groupby("persona"):
        color = PERSONA_COLORS.get(persona, "#888")
        emoji = group["persona_emoji"].iloc[0] if "persona_emoji" in group.columns else ""
        fig.add_trace(go.Scatter(
            x=group[x_col], y=group[y_col], mode="markers",
            name=f"{emoji} {persona}",
            marker=dict(size=11, color=color, opacity=0.88,
                        line=dict(width=1, color="rgba(0,0,0,0.4)")),
            customdata=group[["track_name","artist_name","valence","energy","persona"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                "<span style='color:#aaa'>────────────</span><br>"
                "Persona: %{customdata[4]}<br>"
                "Valence: %{customdata[2]:.2f} · Energy: %{customdata[3]:.2f}"
                "<extra></extra>"
            ),
        ))
    fig.update_layout(
        title=f"Taste Space ({method} projection)",
        paper_bgcolor=BG_COLOR, plot_bgcolor="#111",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        legend=dict(bgcolor="#1A1A1A", bordercolor="#2A2A2A", borderwidth=1,
                    title="Persona", font=dict(size=12)),
        height=560,
        hoverlabel=dict(bgcolor="#1A1A1A", bordercolor="#2A2A2A", font_size=13),
    )
    return fig


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"


def make_radar(df: pd.DataFrame) -> go.Figure:
    features   = AUDIO_FEATURES_RADAR
    categories = [f.replace("_"," ").title() for f in features]
    fig = go.Figure()
    for tr, color, label in [("short_term","#FF4D4D","Recent (4 weeks)"),
                               ("long_term", "#1DB954","All-time (years)")]:
        subset = df[df["time_range"] == tr]
        if subset.empty: continue
        vals = [subset[f].mean() for f in features] + [subset[features[0]].mean()]
        cats = categories + [categories[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself",
            fillcolor=hex_to_rgba(color, 0.20),
            line=dict(color=color, width=2), name=label,
        ))
    fig.update_layout(
        polar=dict(bgcolor="#111",
                   radialaxis=dict(visible=True, range=[0,1], color="#555", gridcolor="#2A2A2A"),
                   angularaxis=dict(color="#999", gridcolor="#2A2A2A")),
        showlegend=True,
        legend=dict(bgcolor="#1A1A1A", bordercolor="#2A2A2A", borderwidth=1),
        paper_bgcolor=BG_COLOR, template=PLOTLY_TEMPLATE,
        title="Audio Fingerprint (normalized features)", height=480,
    )
    return fig


def make_genre_bar(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    all_genres = []
    for g in df["genres"].dropna():
        if g != "Unknown":
            all_genres.extend([x.strip() for x in g.split(",")])
    if all_genres:
        counts = pd.Series(all_genres).value_counts().head(top_n)
        title  = f"Top {top_n} Genres"; x_title = "Track count"
        cs     = [[0,"#1a3d2b"],[1,"#1DB954"]]
    else:
        counts = df["artist_name"].value_counts().head(top_n)
        title  = f"Top {top_n} Artists"; x_title = "Times in top tracks"
        cs     = [[0,"#1a1a3d"],[1,"#7B68EE"]]
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index, orientation="h",
        marker=dict(color=list(range(len(counts))), colorscale=cs,
                    showscale=False, line=dict(width=0)),
        text=counts.values, textposition="outside",
        textfont=dict(color="#aaa", size=11),
        hovertemplate="<b>%{y}</b><br>"+x_title+": %{x}<extra></extra>",
    ))
    fig.update_layout(
        title=title, xaxis_title=x_title,
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        template=PLOTLY_TEMPLATE, paper_bgcolor=BG_COLOR, plot_bgcolor="#111",
        height=500, margin=dict(l=140, r=50),
        xaxis=dict(showgrid=True, gridcolor="#1A1A1A"),
    )
    return fig


def make_valence_timeline(df: pd.DataFrame) -> go.Figure:
    recent = df[df["time_range"] == "recently_played"].copy()
    recent_with_time = recent.dropna(subset=["added_at","valence"])
    if not recent_with_time.empty:
        recent_with_time = recent_with_time.sort_values("added_at")
        recent_with_time["valence_smooth"] = recent_with_time["valence"].rolling(5,min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_with_time["added_at"], y=recent_with_time["valence"],
            mode="markers", marker=dict(size=7, color="#444", opacity=0.6),
            name="Track valence",
            customdata=recent_with_time[["track_name","artist_name"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Valence: %{y:.2f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=recent_with_time["added_at"], y=recent_with_time["valence_smooth"],
            mode="lines", line=dict(color="#FFD700", width=2.5), name="5-track rolling avg",
        ))
        fig.add_hline(y=0.5, line_dash="dot", line_color="#333",
                      annotation_text="Neutral", annotation_font_color="#555")
        fig.update_layout(title="Mood Over Time (recently played)",
                          xaxis_title="Played at", yaxis_title="Valence",
                          yaxis=dict(range=[0,1]), height=360)
    else:
        persona_order = (df.groupby("persona")["valence"].median()
                         .sort_values().index.tolist())
        fig = go.Figure()
        for persona in persona_order:
            color  = PERSONA_COLORS.get(persona,"#888")
            subset = df[df["persona"]==persona]
            emoji  = subset["persona_emoji"].iloc[0] if "persona_emoji" in subset.columns else ""
            r,g,b  = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
            fig.add_trace(go.Box(
                y=subset["valence"], name=f"{emoji} {persona}",
                marker_color=color, line_color=color,
                fillcolor=f"rgba({r},{g},{b},0.15)",
                boxpoints="all", jitter=0.4, pointpos=0,
                marker=dict(size=5, opacity=0.5, color=color),
                customdata=subset[["track_name","artist_name"]].values,
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Valence: %{y:.2f}<extra></extra>",
            ))
        fig.add_hline(y=0.5, line_dash="dot", line_color="#333",
                      annotation_text="Neutral", annotation_font_color="#555")
        fig.update_layout(title="Valence Distribution by Persona",
                          yaxis_title="Valence (0=dark, 1=happy)",
                          yaxis=dict(range=[-0.05,1.05]),
                          showlegend=False, height=360)
    fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=BG_COLOR, plot_bgcolor="#111",
                      legend=dict(bgcolor="#1A1A1A"),
                      hoverlabel=dict(bgcolor="#1A1A1A", bordercolor="#2A2A2A", font_size=13))
    return fig


def make_energy_histogram(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=df["energy"], nbinsx=25,
        marker=dict(
            color=df["energy"].values if len(df)>0 else [],
            colorscale=[[0,"#1A1A1A"],[0.5,"#FF6B6B"],[1,"#FF4D4D"]],
            showscale=False, line=dict(color="#0D0D0D", width=0.5),
        ), opacity=0.9,
    ))
    fig.update_layout(title="Energy Distribution",
                      xaxis_title="Energy (normalized)", yaxis_title="Track count",
                      template=PLOTLY_TEMPLATE, paper_bgcolor=BG_COLOR,
                      plot_bgcolor="#111", height=300)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD SECTIONS
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("## 🎛 Filters")
    st.sidebar.markdown("---")

    all_ranges = ["short_term","medium_term","long_term","recently_played"]
    available  = [r for r in all_ranges if r in df["time_range"].unique()]
    range_labels = {
        "short_term":"Short (4 weeks)", "medium_term":"Medium (6 months)",
        "long_term":"Long-term (years)", "recently_played":"Recently played",
    }
    selected_ranges = st.sidebar.multiselect(
        "Time range", options=available,
        default=[r for r in available if r != "recently_played"],
        format_func=lambda x: range_labels.get(x, x),
    )
    personas = sorted(df["persona"].dropna().unique().tolist()) if "persona" in df.columns else []
    emoji_map = {}
    if "persona_emoji" in df.columns:
        emoji_map = df.drop_duplicates("persona").set_index("persona")["persona_emoji"].to_dict()
    selected_personas = st.sidebar.multiselect(
        "Persona filter", options=personas, default=personas,
        format_func=lambda p: f"{emoji_map.get(p,'')} {p}",
    )
    all_genres = sorted({x.strip() for g in df["genres"].dropna()
                         if g != "Unknown" for x in g.split(",")})
    selected_genres = st.sidebar.multiselect(
        "Genre filter", options=all_genres, default=[], placeholder="All genres",
    ) if all_genres else []
    if not all_genres:
        st.sidebar.markdown(
            "<span style='color:#444;font-size:.72rem'>Genre filter unavailable<br>"
            "(Spotify API restricted)</span>", unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<span style='color:#555;font-size:.75rem'>"
        f"{len(df):,} tracks · {df['artist_name'].nunique()} artists<br>"
        f"Clustered with KMeans k=5</span>", unsafe_allow_html=True)

    if st.sidebar.button("🔓 Log out"):
        for key in ["token_info","music_data","is_synthetic"]:
            st.session_state.pop(key, None)
        st.rerun()

    filtered = df.copy()
    if selected_ranges:
        filtered = filtered[filtered["time_range"].isin(selected_ranges)]
    if selected_personas and "persona" in filtered.columns:
        filtered = filtered[filtered["persona"].isin(selected_personas)]
    if selected_genres:
        mask = filtered["genres"].apply(lambda g: any(genre in str(g) for genre in selected_genres))
        filtered = filtered[mask]
    return filtered


def render_personality(df: pd.DataFrame):
    st.markdown('<p class="section-header">01 — Your Music Personality</p>', unsafe_allow_html=True)
    avg_v = df["valence"].mean(); avg_e = df["energy"].mean(); avg_d = df["danceability"].mean()
    all_genres = [x.strip() for g in df["genres"].dropna() if g != "Unknown" for x in g.split(",")]
    if all_genres:
        top_label_key = "Top Genre"
        top_label_val = pd.Series(all_genres).value_counts().idxmax()
    else:
        top_label_key = "Top Artist"
        top_label_val = df["artist_name"].value_counts().idxmax() if not df.empty else "—"

    dominant = df["persona"].value_counts().idxmax() if "persona" in df.columns else "—"
    persona_row = df[df["persona"] == dominant].iloc[0] if "persona" in df.columns and not df.empty else None

    c1, c2, c3, c4 = st.columns(4)
    def stat_card(col, label, value):
        col.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
        </div>""", unsafe_allow_html=True)

    stat_card(c1, "Mood Score (Valence)", f"{avg_v:.2f}")
    stat_card(c2, "Avg Energy",           f"{avg_e:.2f}")
    stat_card(c3, "Avg Danceability",     f"{avg_d:.2f}")
    top_str = str(top_label_val)
    stat_card(c4, top_label_key, top_str[:18] + ("…" if len(top_str)>18 else ""))

    if persona_row is not None:
        emoji = persona_row.get("persona_emoji","🎵")
        desc  = persona_row.get("persona_description","")
        color = PERSONA_COLORS.get(dominant,"#1DB954")
        count = df["persona"].value_counts()[dominant]
        st.markdown(f"""
        <div class="persona-card" style="border-color:{color}">
            <span style="font-size:3rem">{emoji}</span>
            <div style="font-family:'Space Mono',monospace;font-size:1.6rem;font-weight:700;color:{color}">{dominant}</div>
            <div style="color:#bbb;font-size:.95rem;margin-top:8px">{desc}</div>
            <div style="margin-top:12px;color:#555;font-size:.72rem;font-family:monospace">
                Dominant cluster in your filtered library ({count} tracks)
            </div>
        </div>""", unsafe_allow_html=True)


def render_taste_space(df: pd.DataFrame):
    st.markdown("---")
    st.markdown('<p class="section-header">02 — Taste Space</p>', unsafe_allow_html=True)
    st.plotly_chart(make_scatter(df), use_container_width=True)


def render_fingerprint(df: pd.DataFrame):
    st.markdown("---")
    st.markdown('<p class="section-header">03 — Audio Fingerprint</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([3,2])
    with col1:
        st.plotly_chart(make_radar(df), use_container_width=True)
    with col2:
        st.markdown("##### Feature Breakdown")
        for feat in AUDIO_FEATURES_RADAR:
            val   = df[feat].mean()
            label = feat.replace("_"," ").title()
            bar_c = "#1DB954" if val > 0.5 else "#FF4D4D"
            st.markdown(f"""
            <div style="margin-bottom:10px">
                <div style="display:flex;justify-content:space-between;font-size:.8rem;color:#aaa;margin-bottom:3px">
                    <span>{label}</span><span style="color:#eee;font-family:monospace">{val:.2f}</span>
                </div>
                <div style="background:#1A1A1A;border-radius:4px;height:6px;overflow:hidden">
                    <div style="width:{val*100:.1f}%;background:{bar_c};height:100%;border-radius:4px"></div>
                </div>
            </div>""", unsafe_allow_html=True)
        short_df = df[df["time_range"]=="short_term"]
        long_df  = df[df["time_range"]=="long_term"]
        if not short_df.empty and not long_df.empty:
            st.markdown("##### 🔄 Taste Shift (Short → Long term)")
            for feat in ["energy","valence","danceability","acousticness"]:
                delta = short_df[feat].mean() - long_df[feat].mean()
                arrow = "↑" if delta > 0.02 else ("↓" if delta < -0.02 else "→")
                color = "#1DB954" if delta > 0.02 else ("#FF4D4D" if delta < -0.02 else "#888")
                st.markdown(
                    f'<span style="color:#aaa;font-size:.82rem">{feat.title()}: '
                    f'<span style="color:{color};font-family:monospace">{arrow} {delta:+.2f}</span></span>',
                    unsafe_allow_html=True)


def render_trends(df: pd.DataFrame):
    st.markdown("---")
    st.markdown('<p class="section-header">04 — Genre & Mood Trends</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([2,3])
    with c1:
        st.plotly_chart(make_genre_bar(df), use_container_width=True)
    with c2:
        st.plotly_chart(make_valence_timeline(df), use_container_width=True)
        st.plotly_chart(make_energy_histogram(df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Step 1 — handle OAuth callback
    authenticated = handle_oauth_callback()

    if not authenticated:
        render_login()
        return

    sp = get_spotify_client()
    if sp is None:
        render_login()
        return

    # Step 2 — run pipeline (cached in session_state)
    if "music_data" not in st.session_state:
        with st.spinner("🎵 Fetching your Spotify data and crunching clusters…"):
            try:
                df, is_synthetic = run_pipeline(sp)
                st.session_state["music_data"]   = df
                st.session_state["is_synthetic"] = is_synthetic
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                return

    df           = st.session_state["music_data"]
    is_synthetic = st.session_state.get("is_synthetic", False)

    # Step 3 — header
    try:
        user = sp.current_user()
        display_name = user.get("display_name", "")
    except Exception:
        display_name = ""

    st.markdown(
        f'<h1 style="font-family:Space Mono,monospace;color:#1DB954;letter-spacing:.04em;margin-bottom:0">'
        f'🎵 {display_name + chr(39) + "s " if display_name else ""}Music DNA</h1>',
        unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#555;font-size:.85rem;margin-top:4px">'
        'Your Spotify listening history · decoded · clustered · visualized</p>',
        unsafe_allow_html=True)

    if is_synthetic:
        st.warning(
            "⚠️ **Synthetic audio features** — Spotify's `/audio-features` endpoint returned 403 "
            "(restricted for most developer apps since late 2024). Energy, valence, danceability etc. "
            "are **estimated from genre seeds**, not measured. Clustering still works.",
        )

    # Step 4 — sidebar filters
    filtered = render_sidebar(df)
    if filtered.empty:
        st.warning("No tracks match the current filters. Try broadening your selection.")
        return

    st.markdown(
        f'<p style="color:#444;font-size:.78rem;margin-bottom:24px">'
        f'Showing {len(filtered):,} of {len(df):,} tracks</p>',
        unsafe_allow_html=True)

    # Step 5 — dashboard sections
    render_personality(filtered)
    render_taste_space(filtered)
    render_fingerprint(filtered)
    render_trends(filtered)

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;color:#2A2A2A;font-size:.7rem;font-family:monospace">'
        'built with spotipy · scikit-learn · plotly · streamlit</p>',
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
