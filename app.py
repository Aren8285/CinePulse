import streamlit as st
import pandas as pd
import pickle
import requests
import nltk
import random
import re
import wikipedia
from googleapiclient.discovery import build
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect, LangDetectException

try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
except KeyError:
    st.error("API keys not set in secrets. Please configure them in Streamlit Cloud settings.")
    st.stop()

COLOR_BG = "#000000"
COLOR_ACCENT_PURPLE = "#9929EA"
COLOR_ACCENT_PINK = "#FF5FCF"
COLOR_ACCENT_YELLOW = "#FAEB92"
COLOR_CARD_BG = "#121212"

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

KEYWORD_MAP = {
    "visuals": ["cgi", "vfx", "visuals", "cinematography", "look", "beautiful"],
    "acting": ["acting", "performance", "cast", "actor", "actress", "role", "villain"],
    "story": ["story", "plot", "writing", "script", "narrative", "ending"],
    "action": ["action", "fight", "stunt", "battle"],
    "pacing": ["pacing", "slow", "boring", "drag", "long"]
}

VERDICT_TEMPLATES = {
    "positive": [
        "Audiences are raving about the {pro}. A solid hit.",
        "The consensus is highly positive, with specific praise for the {pro}.",
        "Fans are loving this, especially the {pro}.",
        "A crowd pleaser! Viewers were particularly impressed by the {pro}."
    ],
    "mixed": [
        "Reactions are divided. While the {pro} is praised, some disliked the {con}.",
        "A polarizing watch. Viewers enjoyed the {pro} but criticized the {con}.",
        "Good concepts, but execution received mixed reviews regarding the {con}."
    ],
    "negative": [
        "Reception is chilly. Major complaints focus on the {con}.",
        "Viewers were disappointed, citing issues with the {con}.",
        "Failed to meet expectations, largely due to the {con}."
    ]
}

def is_valid_comment(text):
    if re.search(r'\d{1,2}:\d{2}', text): return False 
    if len(text) < 20: return False 
    if len(text) > 400: return False 
    
    relevant_words = ['movie', 'film', 'scene', 'actor', 'plot', 'cgi', 'story', 'ending', 
                      'character', 'marvel', 'dc', 'action', 'trailer', 'best', 'worst', 'part']
    if not any(w in text.lower() for w in relevant_words): return False
    
    try:
        if detect(text) != 'en': return False
    except LangDetectException:
        return False
        
    return True

def analyze_consensus(comments, avg_score):
    found_pros = []
    found_cons = []
    text_blob = " ".join(comments).lower()
    
    for category, keywords in KEYWORD_MAP.items():
        if any(k in text_blob for k in keywords):
            if avg_score > 0.1: 
                found_pros.append(category)
            elif avg_score < -0.1: 
                found_cons.append(category)
            else:
                if random.random() > 0.5: found_pros.append(category)
                else: found_cons.append(category)

    pro = found_pros[0] if found_pros else "entertainment value"
    con = found_cons[0] if found_cons else "pacing"
    
    if avg_score >= 0.2:
        template = random.choice(VERDICT_TEMPLATES["positive"])
    elif avg_score <= -0.15:
        template = random.choice(VERDICT_TEMPLATES["negative"])
    else:
        template = random.choice(VERDICT_TEMPLATES["mixed"])
        
    return template.format(pro=pro, con=con)

def get_vibe_text(score):
    if score >= 0.5: return "Universal Acclaim", COLOR_ACCENT_YELLOW
    elif score >= 0.35: return "Hype is Real", COLOR_ACCENT_PINK
    elif score >= 0.15: return "Generally Positive", COLOR_ACCENT_PURPLE
    elif score >= -0.1: return "Mixed / Polarizing", "#FFFFFF"
    elif score >= -0.4: return "Underwhelming", "#FF8800"
    else: return "Critical Flop", "#FF0000"

def fetch_movie_details_and_rating(movie_id, title):
    placeholder_img = f"https://placehold.co/500x750/121212/FF5FCF?text={title.replace(' ', '+')}&font=Montserrat"
    
    poster = placeholder_img
    overview = "Plot details currently unavailable."
    live_rating = 6.0 

    if TMDB_API_KEY != "YOUR_TMDB_API_KEY_HERE":
        try:
            url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
            resp = requests.get(url, timeout=1.0)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('poster_path'):
                    poster = "https://image.tmdb.org/t/p/w500/" + data['poster_path']
                if data.get('overview') and len(data.get('overview')) > 10:
                    overview = data.get('overview')
                if data.get('vote_average'):
                    live_rating = float(data.get('vote_average'))
        except:
            pass

    if overview == "Plot details currently unavailable.":
        try:
            wiki_summary = wikipedia.summary(title + " film", sentences=3)
            if wiki_summary: overview = wiki_summary
        except: pass
            
    if poster == placeholder_img:
        wiki_img = fetch_poster_wiki(title)
        if wiki_img: poster = wiki_img

    return poster, overview, live_rating

def fetch_poster_wiki(title):
    try:
        search_res = wikipedia.search(title + " film poster")
        if not search_res: return None
        page = wikipedia.page(search_res[0], auto_suggest=False)
        for img in page.images:
            lower_img = img.lower()
            if ('poster' in lower_img or 'cover' in lower_img) and not 'svg' in lower_img:
                return img
    except:
        return None
    return None

def get_youtube_data(movie_title, rating_val):
    try: rating_val = float(rating_val)
    except: rating_val = 6.0
    
    tmdb_sentiment = (rating_val - 6.0) / 2.5
    tmdb_sentiment = max(min(tmdb_sentiment, 1.0), -1.0)

    if YOUTUBE_API_KEY == "YOUR_YOUTUBE_API_KEY_HERE":
        return generate_synthetic_data(movie_title, rating_val)

    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        request = youtube.search().list(q=f"{movie_title} official trailer", part='id,snippet', type='video', maxResults=1)
        response = request.execute()
        
        if not response['items']: return generate_synthetic_data(movie_title, rating_val)
            
        video_id = response['items'][0]['id']['videoId']
        comments_req = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100, textFormat="plainText")
        comments_resp = comments_req.execute()
        
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        valid_comments = []
        
        for item in comments_resp['items']:
            text = item['snippet']['topLevelComment']['snippet']['textDisplay']
            clean_text = text.replace('\n', ' ').strip()
            
            if is_valid_comment(clean_text):
                score = analyzer.polarity_scores(clean_text)['compound']
                scores.append(score)
                valid_comments.append(clean_text)
            
        if len(scores) < 3: return generate_synthetic_data(movie_title, rating_val)
            
        avg_yt_score = sum(scores) / len(scores)

        if rating_val >= 8.0:
            avg_yt_score = max(avg_yt_score, -0.05)
        
        if rating_val >= 7.5:
            final_vibe_score = (0.8 * tmdb_sentiment) + (0.2 * avg_yt_score)
        elif rating_val <= 6.0:
            final_vibe_score = (0.8 * tmdb_sentiment) + (0.2 * avg_yt_score)
        else:
            final_vibe_score = (0.5 * tmdb_sentiment) + (0.5 * avg_yt_score)
        
        vibe_text, vibe_color = get_vibe_text(final_vibe_score)
        consensus_summary = analyze_consensus(valid_comments, final_vibe_score)
        
        return vibe_text, vibe_color, final_vibe_score, consensus_summary
            
    except:
        return generate_synthetic_data(movie_title, rating_val)

def generate_synthetic_data(movie_title, rating_val):
    tmdb_sentiment = (rating_val - 6.4) / 1.5
    tmdb_sentiment = max(min(tmdb_sentiment, 1.0), -1.0)
    
    jitter = random.uniform(-0.05, 0.05)
    final_score = tmdb_sentiment + jitter

    if final_score > 0.2:
        pro = random.choice(["storytelling", "visuals", "lead performance"])
        summary = f"A solid hit! Viewers are particularly impressed by the {pro}."
    elif final_score < -0.15:
        con = random.choice(["pacing", "script", "CGI"])
        summary = f"Reception is chilly. Major complaints focus on the {con}."
    else:
        pro = "action"
        con = "plot depth"
        summary = f"Reactions are divided. While the {pro} is praised, some disliked the {con}."

    vibe_text, vibe_color = get_vibe_text(final_score)
    return vibe_text, vibe_color, final_score, summary

def recommend(selected_movie_title):
    try: movie_index = movies[movies['title'] == selected_movie_title].index[0]
    except: return []

    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:16]

    candidates = []
    selected_tokens = set(selected_movie_title.lower().split())
    first_word = selected_movie_title.split()[0]

    for i in movies_list:
        idx = i[0]
        title = movies.iloc[idx].title
        sim_score = i[1]
        movie_id = movies.iloc[idx].movie_id
        
        try: rating = movies.iloc[idx].vote_average
        except: rating = 6.0
        
        bonus = 0.0
        if title.startswith(first_word): bonus += 0.3
        rating_boost = (rating / 10.0) * 0.15
        
        relevance_score = sim_score + bonus + rating_boost
        
        candidates.append({
            "title": title,
            "movie_id": movie_id,
            "similarity": sim_score,
            "rating": rating,
            "relevance": relevance_score
        })
    
    candidates = sorted(candidates, key=lambda x: x['relevance'], reverse=True)[:5]
    return candidates

st.set_page_config(page_title="CinePulse", layout="wide", page_icon="🎬")

st.markdown(f"""
<style>
    .stApp {{ background-color: {COLOR_BG}; }}
    
    h1 {{
        color: #FFFFFF !important;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        margin-bottom: 0px;
    }}
    .sub-header {{
        text-align: center;
        color: {COLOR_ACCENT_YELLOW};
        font-family: 'Courier New', monospace;
        font-size: 16px;
        margin-bottom: 30px;
        letter-spacing: 1px;
    }}
    
    .movie-card {{
        background-color: {COLOR_CARD_BG};
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #333;
    }}
    
    .metric-label {{
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #888;
        margin-top: 12px;
        margin-bottom: 2px;
    }}
    .metric-value {{
        font-size: 22px;
        font-weight: 800;
        color: #FFF;
    }}
    .metric-highlight {{ color: {COLOR_ACCENT_PINK}; }}
    
    .stButton>button {{
        width: 100%;
        background-color: {COLOR_ACCENT_PURPLE};
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: {COLOR_ACCENT_PINK};
        box-shadow: 0px 0px 10px {COLOR_ACCENT_PINK};
        transform: translateY(-2px);
    }}
    .stButton>button:active {{
        transform: translateY(1px);
    }}
    
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {COLOR_ACCENT_PURPLE} 0%, {COLOR_ACCENT_PINK} 100%);
    }}
    
    .streamlit-expanderHeader {{ color: {COLOR_ACCENT_YELLOW} !important; font-size: 14px; }}
    
    .summary-text {{
        font-size: 14px;
        color: #e0e0e0;
        margin-bottom: 10px;
        line-height: 1.4;
    }}
    .summary-label {{
        font-weight: bold;
        color: {COLOR_ACCENT_PURPLE};
        margin-bottom: 4px;
    }}
</style>
""", unsafe_allow_html=True)

try:
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open('similarity.pkl', 'rb'))
except:
    st.error("Data missing. Run generate_model.py")
    st.stop()

st.title("CINEPULSE")
st.markdown(f"<div class='sub-header'>An Intelligent Approach to Personalized Movie Recommendations</div>", unsafe_allow_html=True)

selected_movie = st.selectbox("", movies['title'].values)

if st.button('Get Recommendations'):
    with st.spinner('Compiling Viewer Verdicts...'):
        recommendations = recommend(selected_movie)
        
        if not recommendations:
            st.error("No data found.")
        else:
            cols = st.columns(5)
            for idx, col in enumerate(cols):
                rec = recommendations[idx]
                
                poster_url, plot_overview, live_rating = fetch_movie_details_and_rating(rec['movie_id'], rec['title'])
                
                vibe_text, vibe_color, sentiment_score, consensus_summary = get_youtube_data(rec['title'], live_rating)
                
                if len(plot_overview) > 280:
                    short_plot = plot_overview[:280].rsplit(' ', 1)[0] + "..."
                else:
                    short_plot = plot_overview
                
                norm_sentiment = (sentiment_score + 1) / 2
                norm_rating = float(live_rating) / 10.0
                sequel_boost = 0.15 if selected_movie.split()[0] in rec['title'] else 0.0
                hybrid_score = (0.4 * rec['similarity']) + (0.3 * norm_rating) + (0.2 * norm_sentiment) + sequel_boost
                likelihood = int(hybrid_score * 100)
                if likelihood > 98: likelihood = 98
                
                with col:
                    st.image(poster_url, use_container_width=True)
                    
                    st.markdown(f"""
                    <div class="movie-card">
                        <div style="height: 50px; overflow: hidden; font-weight: bold; font-size: 16px; color: #fff; margin-bottom: 5px;">
                            {rec['title']}
                        </div>
                        <div class="metric-label">LIKELIHOOD MATCH</div>
                        <div class="metric-value metric-highlight">{likelihood}%</div>
                        <div class="metric-label">LIVE VIBE</div>
                        <div class="metric-value" style="color:{vibe_color}; font-size: 18px;">{vibe_text}</div>
                        <div class="metric-label">CONTENT SIMILARITY</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(int(rec['similarity']*100))
                    
                    with st.expander("VIEWER VERDICT"):
                        st.markdown(f"<div class='summary-label'>THE GIST</div><div class='summary-text'>{short_plot}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='summary-label'>WHAT VIEWERS THINK</div><div class='summary-text'>{consensus_summary}</div>", unsafe_allow_html=True)