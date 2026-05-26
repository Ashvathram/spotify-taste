"""
app.py — Streamlit dashboard: 🎵 My Music DNA

Sections:
  1. Your Music Personality  — stat cards + dominant persona
  2. Taste Space             — UMAP/PCA scatter, colored by persona
  3. Audio Fingerprint       — radar chart comparing time ranges
  4. Genre & Mood Trends     — genre bar, valence over time, energy histogram

Run with:
    streamlit run app.py
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = "data"
TRACKS_CSV = os.path.join(DATA_DIR, "tracks.csv")
ELBOW_JSON = os.path.join(DATA_DIR, "elbow_plot.json")
SYNTHETIC_FLAG = os.path.join(DATA_DIR, ".synthetic_features")

AUDIO_FEATURES_RADAR = [
    "danceability", "energy", "valence",
    "acousticness", "instrumentalness", "speechiness", "liveness",
]

PERSONA_COLORS = {
    "Party Mode":     "#FF4D4D",
    "Chill Acoustic": "#4DC9FF",
    "Feel Good":      "#FFD700",
    "Dark & Moody":   "#9B59B6",
    "Focus Zone":     "#1DB954",
}

PLOTLY_TEMPLATE = "plotly_dark"
BG_COLOR = "#0D0D0D"
CARD_BG = "#1A1A1A"

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="My Music DNA",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0D0D0D;
    color: #EFEFEF;
  }

  h1, h2, h3 {
    font-family: 'Space Mono', monospace;
  }

  .stat-card {
    background: #1A1A1A;
    border: 1px solid #2A2A2A;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
  }
  .stat-card:hover { border-color: #1DB954; }
  .stat-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #1DB954;
    margin: 4px 0;
  }
  .stat-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #888;
  }

  .persona-card {
    background: linear-gradient(135deg, #1A1A1A 0%, #111 100%);
    border: 2px solid #1DB954;
    border-radius: 16px;
    padding: 28px 32px;
    margin-top: 16px;
  }
  .persona-emoji { font-size: 3rem; }
  .persona-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #1DB954;
  }
  .persona-desc { color: #bbb; font-size: 0.95rem; margin-top: 8px; }

  .section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #1DB954;
    margin-bottom: 4px;
  }

  div[data-testid="stSidebar"] {
    background-color: #111;
    border-right: 1px solid #222;
  }

  .stSelectbox label, .stMultiSelect label { color: #888 !important; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame | None:
    if not os.path.exists(TRACKS_CSV):
        return None
    df = pd.read_csv(TRACKS_CSV)
    if "added_at" in df.columns:
        df["added_at"] = pd.to_datetime(df["added_at"], errors="coerce", utc=True)
    return df


@st.cache_data
def load_elbow() -> dict | None:
    if not os.path.exists(ELBOW_JSON):
        return None
    with open(ELBOW_JSON) as f:
        return json.load(f)


# ── Chart helpers ──────────────────────────────────────────────────────────────

def make_scatter(df: pd.DataFrame, use_umap: bool = True) -> go.Figure:
    """UMAP or PCA scatter colored by persona."""
    x_col = "umap_x" if (use_umap and "umap_x" in df.columns) else "pca_x"
    y_col = "umap_y" if (use_umap and "umap_y" in df.columns) else "pca_y"
    method = "UMAP" if x_col == "umap_x" else "PCA"

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="persona",
        color_discrete_map=PERSONA_COLORS,
        hover_data={
            "track_name": True,
            "artist_name": True,
            "persona": True,
            "valence": ":.2f",
            "energy": ":.2f",
            x_col: False,
            y_col: False,
        },
        labels={"persona": "Persona"},
        title=f"Taste Space ({method} projection)",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.5, color="#000")))
    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor="#111",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        legend=dict(bgcolor="#1A1A1A", bordercolor="#2A2A2A", borderwidth=1),
        height=520,
    )
    return fig


def make_radar(df: pd.DataFrame) -> go.Figure:
    """Radar chart comparing short_term vs long_term audio fingerprints."""
    features = AUDIO_FEATURES_RADAR
    categories = [f.replace("_", " ").title() for f in features]

    fig = go.Figure()

    # hex -> rgba helper (Plotly doesn't accept 8-digit hex alpha)
    def hex_to_rgba(hex_color: str, alpha: float) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    configs = [
        ("short_term", "#FF4D4D", "Recent (4 weeks)"),
        ("long_term",  "#1DB954", "All-time (years)"),
    ]

    for tr, color, label in configs:
        subset = df[df["time_range"] == tr]
        if subset.empty:
            continue
        vals = [subset[f].mean() for f in features]
        vals += [vals[0]]  # close the polygon
        cats = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=cats,
            fill="toself",
            fillcolor=hex_to_rgba(color, 0.20),
            line=dict(color=color, width=2),
            name=label,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="#111",
            radialaxis=dict(visible=True, range=[0, 1], color="#555", gridcolor="#2A2A2A"),
            angularaxis=dict(color="#999", gridcolor="#2A2A2A"),
        ),
        showlegend=True,
        legend=dict(bgcolor="#1A1A1A", bordercolor="#2A2A2A", borderwidth=1),
        paper_bgcolor=BG_COLOR,
        template=PLOTLY_TEMPLATE,
        title="Audio Fingerprint (normalized features)",
        height=480,
    )
    return fig


def make_genre_bar(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of top N genres."""
    all_genres = []
    for g in df["genres"].dropna():
        if g == "Unknown":
            continue
        all_genres.extend([x.strip() for x in g.split(",")])

    if not all_genres:
        return go.Figure().update_layout(title="No genre data", template=PLOTLY_TEMPLATE)

    genre_counts = pd.Series(all_genres).value_counts().head(top_n)

    fig = go.Figure(go.Bar(
        x=genre_counts.values,
        y=genre_counts.index,
        orientation="h",
        marker=dict(
            color=genre_counts.values,
            colorscale=[[0, "#2A2A2A"], [1, "#1DB954"]],
            showscale=False,
        ),
        text=genre_counts.values,
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Top {top_n} Genres",
        xaxis_title="Track count",
        yaxis=dict(autorange="reversed"),
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor="#111",
        height=480,
        margin=dict(l=160),
    )
    return fig


def make_valence_timeline(df: pd.DataFrame) -> go.Figure:
    """Valence over time from recently_played tracks."""
    recent = df[df["time_range"] == "recently_played"].copy()
    recent = recent.dropna(subset=["added_at", "valence"])
    recent = recent.sort_values("added_at")

    if recent.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Valence Over Time (no recently_played data)",
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=BG_COLOR,
        )
        return fig

    # 5-track rolling average for a smoother line
    recent["valence_smooth"] = recent["valence"].rolling(5, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent["added_at"],
        y=recent["valence"],
        mode="markers",
        marker=dict(size=6, color="#555", opacity=0.6),
        name="Raw valence",
        hovertemplate="%{customdata[0]}<br>Valence: %{y:.2f}<extra></extra>",
        customdata=recent[["track_name"]].values,
    ))
    fig.add_trace(go.Scatter(
        x=recent["added_at"],
        y=recent["valence_smooth"],
        mode="lines",
        line=dict(color="#FFD700", width=2.5),
        name="Rolling avg (5 tracks)",
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#444",
                  annotation_text="Neutral mood", annotation_font_color="#666")
    fig.update_layout(
        title="Mood Over Time (valence from recently played)",
        xaxis_title="Played at",
        yaxis_title="Valence (0 = dark, 1 = happy)",
        yaxis=dict(range=[0, 1]),
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor="#111",
        legend=dict(bgcolor="#1A1A1A"),
        height=360,
    )
    return fig


def make_energy_histogram(df: pd.DataFrame) -> go.Figure:
    """Energy distribution histogram."""
    fig = go.Figure(go.Histogram(
        x=df["energy"],
        nbinsx=25,
        marker=dict(
            color=df["energy"].values if len(df) > 0 else [],
            colorscale=[[0, "#1A1A1A"], [0.5, "#FF6B6B"], [1, "#FF4D4D"]],
            showscale=False,
            line=dict(color="#0D0D0D", width=0.5),
        ),
        opacity=0.9,
    ))
    fig.update_layout(
        title="Energy Distribution",
        xaxis_title="Energy (normalized)",
        yaxis_title="Track count",
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor="#111",
        height=300,
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("## 🎛 Filters")
    st.sidebar.markdown("---")

    # Time range filter
    all_ranges = ["short_term", "medium_term", "long_term", "recently_played"]
    available_ranges = [r for r in all_ranges if r in df["time_range"].unique()]
    range_labels = {
        "short_term": "Short (4 weeks)",
        "medium_term": "Medium (6 months)",
        "long_term": "Long-term (years)",
        "recently_played": "Recently played",
    }

    selected_ranges = st.sidebar.multiselect(
        "Time range",
        options=available_ranges,
        default=[r for r in available_ranges if r != "recently_played"],
        format_func=lambda x: range_labels.get(x, x),
    )

    # Genre filter
    all_genres = set()
    for g in df["genres"].dropna():
        if g != "Unknown":
            all_genres.update([x.strip() for x in g.split(",")])
    all_genres = sorted(all_genres)

    selected_genres = st.sidebar.multiselect(
        "Genres (filter tracks)",
        options=all_genres,
        default=[],
        placeholder="All genres",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<span style='color:#555; font-size:0.75rem'>Data from Spotify API • "
        "Clustered with KMeans k=5</span>",
        unsafe_allow_html=True,
    )

    # Apply filters
    filtered = df.copy()
    if selected_ranges:
        filtered = filtered[filtered["time_range"].isin(selected_ranges)]
    if selected_genres:
        mask = filtered["genres"].apply(
            lambda g: any(genre in str(g) for genre in selected_genres)
        )
        filtered = filtered[mask]

    return filtered


# ── Section 1: Personality ─────────────────────────────────────────────────────

def render_personality(df: pd.DataFrame):
    st.markdown('<p class="section-header">01 — Your Music Personality</p>', unsafe_allow_html=True)

    # Compute stats
    avg_valence = df["valence"].mean()
    avg_energy = df["energy"].mean()
    avg_dance = df["danceability"].mean()

    # Top genre
    all_genres = []
    for g in df["genres"].dropna():
        if g != "Unknown":
            all_genres.extend([x.strip() for x in g.split(",")])
    top_genre = pd.Series(all_genres).value_counts().idxmax() if all_genres else "Unknown"

    # Dominant persona
    dominant_persona = df["persona"].value_counts().idxmax() if "persona" in df.columns else "—"
    persona_row = df[df["persona"] == dominant_persona].iloc[0] if "persona" in df.columns and not df.empty else None

    # Stat cards
    c1, c2, c3, c4 = st.columns(4)
    def stat_card(col, label, value, suffix=""):
        col.markdown(f"""
        <div class="stat-card">
          <div class="stat-label">{label}</div>
          <div class="stat-value">{value}{suffix}</div>
        </div>
        """, unsafe_allow_html=True)

    stat_card(c1, "Mood Score (Valence)", f"{avg_valence:.2f}")
    stat_card(c2, "Avg Energy", f"{avg_energy:.2f}")
    stat_card(c3, "Avg Danceability", f"{avg_dance:.2f}")
    stat_card(c4, "Top Genre", top_genre[:18] + ("…" if len(top_genre) > 18 else ""))

    # Persona card
    if persona_row is not None:
        emoji = persona_row.get("persona_emoji", "🎵")
        desc = persona_row.get("persona_description", "")
        persona_color = PERSONA_COLORS.get(dominant_persona, "#1DB954")
        st.markdown(f"""
        <div class="persona-card" style="border-color: {persona_color}">
          <span class="persona-emoji">{emoji}</span>
          <div class="persona-label" style="color:{persona_color}">{dominant_persona}</div>
          <div class="persona-desc">{desc}</div>
          <div style="margin-top:12px; color:#555; font-size:0.72rem; font-family: monospace">
            Dominant cluster in your filtered library ({df['persona'].value_counts()[dominant_persona]} tracks)
          </div>
        </div>
        """, unsafe_allow_html=True)


# ── Section 2: Taste Space ─────────────────────────────────────────────────────

def render_taste_space(df: pd.DataFrame):
    st.markdown("---")
    st.markdown('<p class="section-header">02 — Taste Space</p>', unsafe_allow_html=True)

    has_umap = "umap_x" in df.columns and df["umap_x"].notna().any()
    use_umap = has_umap

    if not has_umap:
        st.info("UMAP coordinates not found — showing PCA projection instead.")

    if "pca_x" not in df.columns:
        st.warning("No dimensionality reduction data found. Run `python analysis.py` first.")
        return

    fig = make_scatter(df, use_umap=use_umap)
    st.plotly_chart(fig, use_container_width=True)


# ── Section 3: Audio Fingerprint ───────────────────────────────────────────────

def render_fingerprint(df: pd.DataFrame):
    st.markdown("---")
    st.markdown('<p class="section-header">03 — Audio Fingerprint</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        fig = make_radar(df)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### Feature Breakdown")
        for feat in AUDIO_FEATURES_RADAR:
            val = df[feat].mean()
            label = feat.replace("_", " ").title()
            bar_color = "#1DB954" if val > 0.5 else "#FF4D4D"
            st.markdown(f"""
            <div style="margin-bottom:10px">
              <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#aaa; margin-bottom:3px">
                <span>{label}</span><span style="color:#eee; font-family:monospace">{val:.2f}</span>
              </div>
              <div style="background:#1A1A1A; border-radius:4px; height:6px; overflow:hidden">
                <div style="width:{val*100:.1f}%; background:{bar_color}; height:100%; border-radius:4px; transition:width 0.3s"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        short_df = df[df["time_range"] == "short_term"]
        long_df = df[df["time_range"] == "long_term"]
        if not short_df.empty and not long_df.empty:
            st.markdown("##### 🔄 Taste Shift (Short → Long term)")
            for feat in ["energy", "valence", "danceability", "acousticness"]:
                s_val = short_df[feat].mean()
                l_val = long_df[feat].mean()
                delta = s_val - l_val
                arrow = "↑" if delta > 0.02 else ("↓" if delta < -0.02 else "→")
                color = "#1DB954" if delta > 0.02 else ("#FF4D4D" if delta < -0.02 else "#888")
                st.markdown(
                    f'<span style="color:#aaa; font-size:0.82rem">{feat.title()}: '
                    f'<span style="color:{color}; font-family:monospace">{arrow} {delta:+.2f}</span></span>',
                    unsafe_allow_html=True,
                )


# ── Section 4: Genre & Mood Trends ─────────────────────────────────────────────

def render_trends(df: pd.DataFrame):
    st.markdown("---")
    st.markdown('<p class="section-header">04 — Genre & Mood Trends</p>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 3])

    with c1:
        fig = make_genre_bar(df)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = make_valence_timeline(df)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = make_energy_histogram(df)
        st.plotly_chart(fig2, use_container_width=True)


# ── Optional: Elbow plot ────────────────────────────────────────────────────────

def render_elbow():
    elbow = load_elbow()
    if elbow:
        with st.expander("📐 KMeans Elbow Method (k justification)"):
            fig = go.Figure(elbow)
            fig.update_layout(height=300, paper_bgcolor=BG_COLOR, plot_bgcolor="#111")
            st.plotly_chart(fig, use_container_width=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown(
        '<h1 style="font-family: Space Mono, monospace; color:#1DB954; '
        'letter-spacing:0.04em; margin-bottom:0">🎵 My Music DNA</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#555; font-size:0.85rem; margin-top:4px">'
        'Your Spotify listening history · decoded · clustered · visualized</p>',
        unsafe_allow_html=True,
    )

    # Load data
    df = load_data()

    if df is None:
        st.error(
            "**No data found.** Run the pipeline first:\n\n"
            "```\npython fetch_data.py\npython preprocess.py\npython analysis.py\n```\n\n"
            "Then reload this page."
        )
        return

    if df.empty:
        st.warning("Data file exists but is empty. Try re-running the pipeline.")
        return

    # Synthetic features notice
    if os.path.exists(SYNTHETIC_FLAG):
        st.warning(
            "⚠️ **Synthetic audio features active** — Spotify's  endpoint "
            "returned 403 (restricted for most developer apps since late 2024). "
            "Energy, valence, danceability etc. are **estimated from genre seeds**, not measured. "
            "Clustering and visualisations still work, but treat the exact values as approximate.",
            icon=None,
        )

    # Sidebar filters
    filtered_df = render_sidebar(df)

    if filtered_df.empty:
        st.warning("No tracks match the current filters. Try broadening your selection.")
        return

    st.markdown(
        f'<p style="color:#444; font-size:0.78rem; margin-bottom:24px">'
        f'Showing {len(filtered_df):,} of {len(df):,} tracks</p>',
        unsafe_allow_html=True,
    )

    # All 4 sections
    render_personality(filtered_df)
    render_taste_space(filtered_df)
    render_fingerprint(filtered_df)
    render_trends(filtered_df)

    # Bonus: elbow plot
    render_elbow()

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; color:#2A2A2A; font-size:0.7rem; font-family:monospace">'
        'built with spotipy · scikit-learn · plotly · streamlit</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()