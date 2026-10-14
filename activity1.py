import random
import time
import pandas as pd
from textblob import TextBlob
from colorama import init, Fore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

init(autoreset=True)

# Load the movie dataset
try:
    df = pd.read_csv("imdb_top_1000.csv")
except FileNotFoundError:
    print(Fore.RED + "Error: imdb_top_1000.csv not found!")
    raise SystemExit

# Handle missing values
df["Genre"] = df["Genre"].fillna("")
df["Overview"] = df["Overview"].fillna("")
df["Series_Title"] = df["Series_Title"].fillna("Unknown")
df["IMDB_Rating"] = pd.to_numeric(df["IMDB_Rating"], errors="coerce")
df = df.dropna(subset=["IMDB_Rating"]).reset_index(drop=True)

# Combine genre and overview for content-based recommendations
df["Combined_Features"] = df["Genre"] + " " + df["Overview"]

# Convert movie text into TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["Combined_Features"])


def dots():
    for _ in range(3):
        print(Fore.YELLOW + ".", end="", flush=True)
        time.sleep(0.4)
    print()


def get_genres():
    genres = sorted({
        genre.strip()
        for movie_genres in df["Genre"]
        for genre in movie_genres.split(",")
        if genre.strip()
    })
    return genres


def get_genre(genres):
    print(Fore.YELLOW + "\nAvailable Genres:")
    print("0. Any Genre")

    for i, genre in enumerate(genres, 1):
        print(f"{i}. {genre}")

    while True:
        choice = input(
            Fore.CYAN + "\nEnter genre number or name: "
        ).strip()

        if choice == "0" or choice.lower() in ["any", "all", "skip"]:
            return None

        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(genres):
                return genres[index - 1]

        for genre in genres:
            if genre.lower() == choice.lower():
                return genre

        print(Fore.RED + "Invalid genre. Try again.")


def get_rating():
    while True:
        value = input(
            Fore.CYAN
            + "\nEnter minimum IMDb rating (0-10) or press Enter to skip: "
        ).strip()

        if value == "":
            return None

        try:
            rating = float(value)
            if 0 <= rating <= 10:
                return rating
        except ValueError:
            pass

        print(Fore.RED + "Enter a valid rating between 0 and 10.")


def sentiment_analysis(text):
    polarity = TextBlob(str(text)).sentiment.polarity

    if polarity > 0.05:
        label = "Positive 😊"
    elif polarity < -0.05:
        label = "Negative 😞"
    else:
        label = "Neutral 😐"

    return polarity, label


def show_movie(movie):
    polarity, sentiment = sentiment_analysis(movie["Overview"])

    print(Fore.CYAN + "\n" + "=" * 55)
    print(Fore.YELLOW + f"🎬 Title: {movie['Series_Title']}")
    print(Fore.GREEN + f"🎭 Genre: {movie['Genre']}")
    print(Fore.GREEN + f"⭐ IMDb Rating: {movie['IMDB_Rating']:.1f}/10")
    print(Fore.BLUE + f"📝 Overview: {movie['Overview']}")
    print(
        Fore.MAGENTA
        + f"💭 Sentiment: {sentiment} (Polarity: {polarity:.2f})"
    )
    print(Fore.CYAN + "=" * 55)


def filter_movies(genre=None, rating=None):
    movies = df.copy()

    if genre:
        movies = movies[
            movies["Genre"].apply(
                lambda genres: genre.lower()
                in [g.strip().lower() for g in genres.split(",")]
            )
        ]

    if rating is not None:
        movies = movies[movies["IMDB_Rating"] >= rating]

    return movies


def recommend_ai(genre=None, mood=None, rating=None, n=5):
    candidates = filter_movies(genre, rating)

    if candidates.empty:
        return []

    print(Fore.BLUE + "\nFinding AI recommendations", end="", flush=True)
    dots()

    # Build a search query from the user's genre and mood
    query = f"{genre or ''} {mood or ''}".strip()

    if not query:
        return candidates.sample(
            n=min(n, len(candidates))
        ).to_dict("records")

    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity between the query and all movies
    similarities = cosine_similarity(
        query_vector, tfidf_matrix
    ).flatten()

    mood_polarity = sentiment_analysis(mood)[0] if mood else 0

    scored_movies = []

    for index in candidates.index:
        overview = df.at[index, "Overview"]
        movie_polarity = sentiment_analysis(overview)[0]

        similarity = similarities[index]

        # Give a small preference to movie sentiment matching the user's mood
        if mood_polarity > 0.05:
            mood_match = max(0, movie_polarity)
        elif mood_polarity < -0.05:
            mood_match = max(0, -movie_polarity)
        else:
            mood_match = 0

        score = similarity + (0.1 * mood_match)
        scored_movies.append((score, index))

    scored_movies.sort(reverse=True)

    return [
        df.loc[index].to_dict()
        for score, index in scored_movies[:n]
    ]


def recommend_random():
    movie = df.sample(n=1).iloc[0]
    return movie.to_dict()


def main():
    genres = get_genres()

    print(
        Fore.BLUE
        + "\n🎥 Welcome to the AI Movie Recommendation System! 🎥"
    )

    name = input(Fore.YELLOW + "\nWhat's your name? ").strip()

    if not name:
        name = "Movie Lover"

    print(Fore.GREEN + f"\nWelcome, {name}!")

    while True:
        print(Fore.YELLOW + "\nChoose your recommendation type:")
        print("1. AI-Based Recommendation")
        print("2. Random Recommendation")
        print("3. Exit")

        choice = input(Fore.CYAN + "\nEnter your choice: ").strip()

        if choice == "1":
            genre = get_genre(genres)

            mood = input(
                Fore.YELLOW
                + "\nHow do you feel today? Describe your mood: "
            ).strip()

            print(Fore.BLUE + "\nAnalyzing your mood", end="", flush=True)
            dots()

            mood_polarity, mood_label = sentiment_analysis(mood)

            print(
                Fore.GREEN
                + f"Your mood: {mood_label} "
                + f"(Polarity: {mood_polarity:.2f})"
            )

            rating = get_rating()

            recommendations = recommend_ai(
                genre=genre,
                mood=mood,
                rating=rating,
                n=5
            )

            if not recommendations:
                print(
                    Fore.RED
                    + "No movies found. Try another genre or rating."
                )
            else:
                print(
                    Fore.YELLOW
                    + f"\n🍿 AI Recommendations for {name}:"
                )

                for movie in recommendations:
                    show_movie(movie)

        elif choice == "2":
            print(Fore.BLUE + "\nSelecting a random movie", end="", flush=True)
            dots()

            movie = recommend_random()

            print(Fore.YELLOW + f"\n🎲 Random Movie for {name}:")
            show_movie(movie)

        elif choice == "3":
            print(
                Fore.GREEN
                + f"\nEnjoy your movies, {name}! 🎬"
            )
            break

        else:
            print(Fore.RED + "Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
