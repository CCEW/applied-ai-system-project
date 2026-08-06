import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Scoring configuration constants
# Tunable weights — changing these adjusts recommendation behavior.
GENRE_WEIGHT = 2.5
MOOD_WEIGHT = 1.0
ENERGY_MAX = 2.5
ENERGY_EXPONENT = 1.2  # exponent on energy difference to control sharpness
ACOUSTIC_WEIGHT = 1.0

# Adaptive multipliers for emphasized preferences (e.g. "party" -> high energy)
ADAPTIVE_ENERGY_BOOST = 1.5
HIGH_ENERGY_THRESHOLD = 0.75

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        user_prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        scored_songs = sorted(
            (
                (song, score, build_explanation(reasons))
                for song in self.songs
                for score, reasons in [score_song(user_prefs, song_to_dict(song))]
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return [song for song, _, _ in scored_songs[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        user_prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        _, reasons = score_song(user_prefs, song_to_dict(song))
        return build_explanation(reasons)


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    """
    print(f"Loading songs from {csv_path}...")
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        songs = []
        for row in reader:
            songs.append(
                {
                    "id": int(row["id"]),
                    "title": row["title"],
                    "artist": row["artist"],
                    "genre": row["genre"],
                    "mood": row["mood"],
                    "energy": float(row["energy"]),
                    "tempo_bpm": float(row["tempo_bpm"]),
                    "valence": float(row["valence"]),
                    "danceability": float(row["danceability"]),
                    "acousticness": float(row["acousticness"]),
                }
            )
    return songs


def song_to_dict(song: Song) -> Dict:
    return {
        "id": song.id,
        "title": song.title,
        "artist": song.artist,
        "genre": song.genre,
        "mood": song.mood,
        "energy": song.energy,
        "tempo_bpm": song.tempo_bpm,
        "valence": song.valence,
        "danceability": song.danceability,
        "acousticness": song.acousticness,
    }


def build_explanation(reasons: List[str]) -> str:
    if not reasons:
        return "This song did not match your profile closely."
    return "Matched because " + "; ".join(reasons) + "."


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    """
    score = 0.0
    reasons: List[str] = []

    user_genre = user_prefs.get("genre")
    song_genre = song.get("genre")
    if user_genre and song_genre and str(user_genre).lower() == str(song_genre).lower():
        score += GENRE_WEIGHT
        reasons.append(f"genre match (+{GENRE_WEIGHT:.1f})")

    user_mood = user_prefs.get("mood")
    song_mood = song.get("mood")
    if user_mood and song_mood and str(user_mood).lower() == str(song_mood).lower():
        score += MOOD_WEIGHT
        reasons.append(f"mood match (+{MOOD_WEIGHT:.1f})")

    target_energy = user_prefs.get("energy")
    song_energy = song.get("energy")
    if target_energy is not None and song_energy is not None:
        target_energy_value = float(target_energy)
        song_energy_value = float(song_energy)
        # Sharper reward for closer energy matches using an exponent
        energy_closeness = max(0.0, 1.0 - abs(song_energy_value - target_energy_value) ** ENERGY_EXPONENT)
        energy_points = ENERGY_MAX * energy_closeness
        # Adaptive boost when the user explicitly requests high energy (e.g., "party")
        adaptive_multiplier = 1.0
        try:
            if float(target_energy_value) >= HIGH_ENERGY_THRESHOLD:
                adaptive_multiplier = ADAPTIVE_ENERGY_BOOST
        except Exception:
            adaptive_multiplier = 1.0
        energy_points *= adaptive_multiplier
        score += energy_points
        reasons.append(f"energy similarity (+{energy_points:.1f})")

    likes_acoustic = user_prefs.get("likes_acoustic", False)
    song_acousticness = song.get("acousticness")
    if song_acousticness is not None:
        acousticness_value = float(song_acousticness)
        acoustic_score = acousticness_value if likes_acoustic else max(0.0, 1.0 - acousticness_value)
        acoustic_points = ACOUSTIC_WEIGHT * acoustic_score
        score += acoustic_points
        reasons.append(f"acoustic preference (+{acoustic_points:.2f})")

    return round(score, 3), reasons


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    """
    scored_songs = sorted(
        (
            (song, score, build_explanation(reasons))
            for song in songs
            for score, reasons in [score_song(user_prefs, song)]
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return scored_songs[:k]
