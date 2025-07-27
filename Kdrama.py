import requests
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import base64


st.set_page_config(page_title="Kdrama Recommender", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

/* Set base font globally */
html, body, [class*="css"], .stApp {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 10px !important;
    background-color: #fdf6e3 !important;
}

/* Ensure Streamlit text widgets are pixelated */
h1, h2, h3, h4, h5, h6, p, span, div, label {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 10px !important;
}

/* Title styling */
h1 {
    text-align: center;
    font-size: 22px !important;
    color: #1a1a1a !important;
    text-shadow: 1px 1px #ffffff;
    margin-top: 2rem;
    margin-bottom: 2rem;
}

/* Pixel font for selectbox dropdown and label */
div[data-baseweb="select"] * {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 10px !important;
}

/* Pixel font for buttons */
.stButton > button {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 10px !important;
    font-weight: bold;
    background-color: #fddbb0;
    color: #333;
    box-shadow: 1px 1px 2px #888;
    border: none;
    padding: 0.4rem 1rem;
    border-radius: 6px;
}

/* Pixel font for sliders */
.css-1xarl3l, .css-1cpxqw2, .stSlider label, .stSlider span, .stSlider div {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 10px !important;
}

/* Recommendation layout */
.recommendation-box {
    display: flex;
    align-items: flex-start;
    gap: 20px;
    margin-bottom: 30px;

}
.recommendation-box img {
    border-radius: 8px;
}
.recommendation-box .text {
    flex: 1;
    font-size: 10px !important;
    color: black !important; 
}
</style>
""", unsafe_allow_html=True)

# API CALL
url = "https://api.themoviedb.org/3/discover/tv"

headers = {

    "Authorization" : "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI2ODNlOGUzZGZjMjNmMWI1ZWVmNGRhZWM3NDQxYzUxMyIsIm5iZiI6MTc1MjE5MTEyMy41NDIsInN1YiI6IjY4NzA1MDkzNjVlNjlmNDUxNWJhMmIwNiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.QhPI8kgfYTPUohySWlPTuDFMAkKktVKxNINQiTgR-Bw",
    "accept": "application/json"

}

dramas = []
for page in range(1,6):
    params = {

        "with_original_language": "ko",
        "sort_by": "popularity.desc",
        "page": page,
        "language": "en-US"

    }

    response = requests.get(url, headers=headers,params = params)
    results = response.json().get("results", [])

    for show in results:
        dramas.append({
            "Title": show.get("name"),
            "Overview": show.get("overview") or "",
            "Rating": show.get("vote_average"),
            "Poster": f"https://image.tmdb.org/t/p/w500{show.get('poster_path')}" if show.get("poster_path") else None
        })

df = pd.DataFrame(dramas)
df.drop_duplicates(subset="Title", inplace=True)
df.reset_index(drop=True, inplace=True)


#  Cosine Similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Overview'])

similarity = cosine_similarity(tfidf_matrix,tfidf_matrix)
index = pd.Series(df.index, index = df['Title']).drop_duplicates()

# Recommendation
with open("download.png","rb") as image:
    img = image.read()

convert_img = base64.b64encode(img).decode()

st.markdown(f"""
    <div style='display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 2rem;'>
        <img src="data:image/png;base64,{convert_img}" width="60"/>
        <h1 style='font-family: "Press Start 2P", monospace; font-size: 22px; color: #1a1a1a; text-shadow: 1px 1px #ffffff; margin: 0;'>KDRAMA<br>RECOMMENDER</h1>
    </div>
""", unsafe_allow_html=True)


chosen_title = st.selectbox("Pick a Kdrama:", df["Title"].unique())
num_recs = st.selectbox("Number of Recommendations", options=[1, 3, 5, 7, 10], index=3)



if st.button("Get Recommendations"):
    if chosen_title in index:
        idx = index[chosen_title]
        sim_scores = list(enumerate(similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_indices = [i[0] for i in sim_scores[1:num_recs + 1]]

        recommended = df.iloc[sim_indices].reset_index(drop=True)

        for _, row in recommended.iterrows():
            col1, col2 = st.columns([1, 3])  # Adjust column width ratio
            with col1:
                if row["Poster"]:
                    st.image(row["Poster"], width=150)
            with col2:
                # st.markdown(f"### {row['Title']}")
                # st.write(row["Overview"])
                st.markdown(f"<h3 style='color:#000000; font-family: \"Press Start 2P\", monospace; font-size: 12px;'>{row['Title']}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='color:#000000; font-family: \"Press Start 2P\", monospace; font-size: 10px;'>{row['Overview']}</p>", unsafe_allow_html=True)



    else:
        st.warning("Selected drama not found in dataset.")

