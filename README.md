# 🎵 Spotify Music Taste Analyzer

A data science project that analyzes your Spotify listening history to reveal your music personality — mood clusters, taste fingerprint, genre breakdown, and listening trends — displayed in an interactive Streamlit dashboard.

---

## What It Does

- Pulls your top tracks across short, medium, and long term from Spotify
- Fetches artist genres and audio features (with a smart fallback if Spotify restricts access)
- Clusters your music into 5 mood personas using KMeans
- Reduces your taste to a 2D map using UMAP / PCA
- Displays everything in a local Streamlit dashboard

---

## Mood Personas

| Persona | Description |
|---|---|
| 🔥 Party Mode | High energy + high danceability |
| 🌿 Chill Acoustic | Low energy + high acousticness |
| ☀️ Feel Good | High valence + fast tempo |
| 🌑 Dark & Moody | Low valence + low energy |
| 🎯 Focus Zone | High instrumentalness |

---

## Setup

### 1. Spotify Developer Credentials

1. Go to [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard)
2. Create an app — set redirect URI to `http://127.0.0.1:8888/callback`
3. Copy your `Client ID` and `Client Secret`

### 2. Environment File

Create a `.env` file in the project root:

```
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Run each step in order:

```bash
# Step 1 — fetch your data (opens browser for Spotify login on first run)
python fetch_data.py

# Step 2 — clean and normalize
python preprocess.py

# Step 3 — cluster and reduce dimensions
python analysis.py

# Step 4 — launch the dashboard
streamlit run app.py
```

---

## Project Structure

```
spotify-taste/
├── .env                  # your Spotify credentials (never commit this)
├── requirements.txt
├── auth.py               # OAuth flow
├── fetch_data.py         # pulls tracks, genres, audio features from Spotify
├── preprocess.py         # cleans and normalizes data
├── analysis.py           # KMeans clustering, PCA, UMAP, persona assignment
├── app.py                # Streamlit dashboard
└── data/                 # auto-created, stores tracks.csv and plots
```

---

## Dashboard Sections

**Your Music Personality** — stat cards for avg mood, energy, danceability, and your dominant persona

**Taste Space** — 2D UMAP scatter of all your tracks colored by mood cluster, hover for track details

**Audio Fingerprint** — radar chart of your average audio profile, comparing short-term vs long-term taste

**Genre & Mood Trends** — top 15 genres, valence over time, energy distribution

---

## Notes

- Spotify restricted the `/audio-features` endpoint for most developer apps in late 2024. If your app hits a 403, `fetch_data.py` automatically generates synthetic audio features seeded by genre — clustering and the dashboard still work normally, with a notice shown.
- The `.spotify_cache` file stores your OAuth token locally so you don't need to log in every run.
- All data is stored locally in `data/` — nothing is uploaded anywhere.

---

## Tech Stack

`spotipy` · `pandas` · `numpy` · `scikit-learn` · `umap-learn` · `plotly` · `streamlit`
