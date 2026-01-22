# 🎬 CinePulse

**CinePulse** is an intelligent movie recommendation and sentiment analysis system that combines **content-based filtering**, **public rating signals**, and **real-time audience sentiment** to generate personalized movie recommendations along with an interpretable “live vibe” indicator.

The system integrates structured movie metadata with unstructured user-generated text data to approximate audience reception in a robust and explainable manner.

---

## 📌 Key Objectives

- Recommend movies based on **content similarity**
- Analyze **viewer sentiment** from real-world discussion data
- Balance **critical consensus** and **audience opinion**
- Present recommendations in a **clear, interpretable UI**

---

## 🧠 System Overview

CinePulse operates through three primary analytical layers:

### 1. Content-Based Recommendation
- Uses cosine similarity on vectorized movie features
- Identifies movies with similar themes, genres, and metadata
- Produces a ranked recommendation list

### 2. Sentiment Intelligence Layer
- Extracts user comments from YouTube trailers
- Applies **VADER sentiment analysis** to natural language text
- Filters low-quality and non-relevant comments
- Aggregates sentiment scores to estimate audience response

### 3. Consensus Calibration
- Normalizes external ratings (TMDB)
- Dynamically weights:
  - Historical critical reception
  - Real-time audience sentiment
- Produces a continuous **vibe score** mapped to qualitative labels

---

## 📊 Live Vibe Classification

The system maps sentiment scores to interpretable categories:

| Score Range | Live Vibe |
|------------|----------|
| ≥ 0.65 | Universal Acclaim |
| ≥ 0.35 | Hype is Real |
| ≥ 0.15 | Generally Positive |
| ≥ -0.15 | Mixed / Polarizing |
| ≥ -0.50 | Underwhelming |
| < -0.50 | Critical Flop |

---

## 🛠️ Technologies Used

### Programming & Libraries
- Python
- Pandas, NumPy
- NLTK (VADER Sentiment Analysis)
- Streamlit

### APIs & Data Sources
- TMDB API (movie metadata & ratings)
- YouTube Data API (viewer comments)
- Wikipedia API (fallback plot summaries)

### Machine Learning Concepts
- Vector similarity (cosine similarity)
- Sentiment polarity analysis
- Weighted score normalization
- Heuristic-based calibration

---

## 📁 Project Structure

```

├── app.py
├── movie_dict.pkl
├── similarity.pkl
├── README.md

````

- `movie_dict.pkl`: Preprocessed movie metadata
- `similarity.pkl`: Precomputed similarity matrix
- `app.py`: Streamlit application logic

---

## 🧪 Methodology Highlights

- **Noise Reduction:** Aggressive filtering of short, spam-like, or irrelevant comments
- **Bias Mitigation:** Prevents contrarian sentiment from overpowering established consensus
- **Explainability:** Provides textual verdicts highlighting perceived strengths and weaknesses
- **Fallback Strategy:** Graceful degradation when external APIs are unavailable

---

## 🎯 Output Metrics

Each recommended movie includes:
- Likelihood match percentage
- Content similarity score
- Live vibe indicator
- Consensus-based viewer verdict

---

## 🚀 Running the Application

```bash
pip install -r requirements.txt
streamlit run app.py
````

Ensure valid API keys are configured for:

* TMDB
* YouTube Data API

---

## 📈 Future Scope

* Hybrid recommendation (collaborative + content-based)
* Emotion-aware sentiment modeling
* Multilingual comment analysis
* Temporal sentiment tracking
* Model-based sentiment classification (BERT)

---

## 📚 Academic Relevance

CinePulse demonstrates practical application of:

* Data preprocessing
* Natural language processing
* Recommendation systems
* Real-world data integration
* Interpretability in ML systems

---

## 📜 License

This project is intended for academic and educational use.
