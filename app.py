
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# --------------------------------
# PROJECT PATHS
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "movie_recommendation_model.pkl"
MOVIES_PATH = BASE_DIR / "movies.csv"


# --------------------------------
# LOAD MODEL
# --------------------------------

@st.cache_resource
def load_model():

    model_data = joblib.load(MODEL_PATH)

    user_sim_df = model_data["user_similarity"]
    user_movie_matrix = model_data["user_movie_matrix"]

    return user_sim_df, user_movie_matrix


@st.cache_data
def load_movies():

    if MOVIES_PATH.exists():
        return pd.read_csv(MOVIES_PATH)

    return pd.DataFrame()


# --------------------------------
# MOVIE TITLE FUNCTION
# --------------------------------

def get_movie_title(movie_id, movies):

    if movies.empty:
        return f"Movie {movie_id}"

    column_lookup = {
        str(col).strip().lower(): col
        for col in movies.columns
    }

    id_col = column_lookup.get("movie_id")
    title_col = column_lookup.get("movie_title")

    if title_col is None:
        title_col = column_lookup.get("title")

    if id_col is None or title_col is None:
        return f"Movie {movie_id}"

    movie_data = movies[
        movies[id_col].astype(str) == str(movie_id)
    ]

    if not movie_data.empty:
        return str(movie_data.iloc[0][title_col])

    return f"Movie {movie_id}"


# --------------------------------
# RECOMMENDATION FUNCTION
# --------------------------------

def recommend_movies(
    target_user,
    user_sim_df,
    user_movie_matrix,
    movies,
    number_of_recommendations=10
):

    # Movies already rated by selected user
    user_ratings = user_movie_matrix.loc[target_user]

    watched_movies = user_ratings[
        user_ratings > 0
    ].index

    # Find similar users
    sim_users = user_sim_df[target_user].drop(
        target_user,
        errors="ignore"
    )

    sim_users = sim_users[
        sim_users > 0
    ].sort_values(ascending=False).head(20)

    recommendations = []

    for movie in user_movie_matrix.columns:

        # Skip already rated movies
        if movie in watched_movies:
            continue

        weighted_sum = 0
        similarity_sum = 0

        # Calculate recommendation score internally
        for sim_user, similarity in sim_users.items():

            rating = user_movie_matrix.loc[sim_user, movie]

            if rating > 0:
                weighted_sum += rating * similarity
                similarity_sum += similarity

        if similarity_sum > 0:

            score = weighted_sum / similarity_sum

        else:

            # Fallback when no similar users are available
            movie_ratings = user_movie_matrix[movie]
            movie_ratings = movie_ratings[movie_ratings > 0]

            if movie_ratings.empty:
                continue

            score = movie_ratings.mean()

        recommendations.append({
            "Movie_ID": movie,
            "Movie_Title": get_movie_title(movie, movies),
            "_score": score
        })

    # Rank internally without displaying scores
    recommendations.sort(
        key=lambda x: x["_score"],
        reverse=True
    )

    # Remove internal score from displayed results
    final_recommendations = [
        {
            "Movie_ID": item["Movie_ID"],
            "Movie_Title": item["Movie_Title"]
        }
        for item in recommendations[:number_of_recommendations]
    ]

    return final_recommendations, sim_users


# --------------------------------
# MAIN APPLICATION
# --------------------------------

st.title("🎬 Personalized Movie Recommendation System")

st.write(
    "Discover movies based on user preferences "
    "using Collaborative Filtering."
)

st.divider()


# --------------------------------
# LOAD DATA
# --------------------------------

try:

    user_sim_df, user_movie_matrix = load_model()
    movies_df = load_movies()

except Exception as error:

    st.error(f"Unable to load model: {error}")
    st.stop()


# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.header("⚙️ Settings")

st.sidebar.success("Model loaded successfully!")

st.sidebar.metric(
    "Total Users",
    user_movie_matrix.shape[0]
)

st.sidebar.metric(
    "Total Movies",
    user_movie_matrix.shape[1]
)

number_of_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=20,
    value=10
)


# --------------------------------
# USER SELECTION
# --------------------------------

st.subheader("Select User")

users = list(user_movie_matrix.index)

selected_user = st.selectbox(
    "Choose User ID",
    options=users
)

watched_count = int(
    (user_movie_matrix.loc[selected_user] > 0).sum()
)

col1, col2 = st.columns(2)

col1.metric(
    "Selected User",
    str(selected_user)
)

col2.metric(
    "Movies Already Rated",
    watched_count
)


# --------------------------------
# GENERATE RECOMMENDATIONS
# --------------------------------

if st.button("🎯 Get Recommendations", type="primary"):

    recommendations, similar_users = recommend_movies(
        selected_user,
        user_sim_df,
        user_movie_matrix,
        movies_df,
        number_of_recommendations
    )

    st.session_state["recommendations"] = recommendations
    st.session_state["similar_users"] = similar_users
    st.session_state["selected_user"] = selected_user


# --------------------------------
# DISPLAY RECOMMENDATIONS
# --------------------------------

if (
    "recommendations" in st.session_state
    and st.session_state["selected_user"] == selected_user
):

    recommendations = st.session_state["recommendations"]
    similar_users = st.session_state["similar_users"]

    st.divider()

    st.subheader("🍿 Recommended Movies")

    if recommendations:

        result_df = pd.DataFrame(recommendations)

        result_df.index = range(1, len(result_df) + 1)
        result_df.index.name = "Rank"

        st.dataframe(
            result_df,
            use_container_width=True
        )

        st.success(
            f"{len(recommendations)} movies recommended successfully!"
        )

    else:

        st.warning(
            "No recommendations available for this user."
        )

    # --------------------------------
    # SIMILAR USERS
    # --------------------------------

    st.subheader("👥 Similar Users")

    if not similar_users.empty:

        similar_df = pd.DataFrame({
            "User_ID": similar_users.index
        })

        st.dataframe(
            similar_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No similar users found. "
            "Recommendations are generated using available movie ratings."
        )


# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.caption(
    "Developed using Python, Pandas, Scikit-learn and Streamlit."
)