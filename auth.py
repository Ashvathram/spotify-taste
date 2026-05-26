"""
auth.py — Spotify OAuth flow.
Returns an authenticated spotipy.Spotify client.

Usage:
    from auth import get_spotify_client
    sp = get_spotify_client()

Or run directly to test auth:
    python auth.py
"""

import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Scopes needed across the whole project
SCOPE = "user-top-read user-read-recently-played"

# Where spotipy caches the OAuth token between runs
CACHE_PATH = ".spotify_cache"


def get_spotify_client() -> spotipy.Spotify:
    """
    Authenticate with Spotify via OAuth and return a ready-to-use client.

    On first run this opens a browser window for login + authorization.
    On subsequent runs it reuses the cached token (auto-refreshed if expired).
    """
    load_dotenv()

    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")

    if not client_id or client_id == "your_client_id_here":
        raise EnvironmentError(
            "SPOTIFY_CLIENT_ID is not set. "
            "Edit .env with your credentials from https://developer.spotify.com/dashboard"
        )
    if not client_secret or client_secret == "your_client_secret_here":
        raise EnvironmentError(
            "SPOTIFY_CLIENT_SECRET is not set. "
            "Edit .env with your credentials from https://developer.spotify.com/dashboard"
        )

    auth_manager = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=SCOPE,
        cache_path=CACHE_PATH,
        open_browser=True,
    )

    sp = spotipy.Spotify(auth_manager=auth_manager)

    # Quick check: fetch current user to confirm auth works
    user = sp.current_user()
    print(f"✅ Authenticated as: {user['display_name']} ({user['id']})")
    return sp


if __name__ == "__main__":
    print("Testing Spotify authentication...")
    sp = get_spotify_client()
    print("Auth successful! You're good to go.")
