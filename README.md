# 🎵 Music AI Recommender

An intelligent music recommendation system powered by BERT embeddings and Spotify API.

## 🌟 Features

- **🔍 Smart Search**: Find music by song name or natural language description
- **🎯 Hybrid Recommendations**: Combines BERT semantic similarity with audio features
- **📊 Data Analytics**: Visualizations of music trends and patterns
- **🎵 Spotify Integration**: Direct links to songs on Spotify
- **🤖 AI-Powered**: Uses sentence-transformers for semantic understanding

## 🚀 Live Demo

**[Try it here →](https://your-app.streamlit.app)** *(link will be available after deployment)*

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/music-ai-recommender.git
cd music-ai-recommender

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app_music.py
```

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML Model**: BERT (sentence-transformers)
- **Data**: Spotify API + Kaggle dataset
- **Features**: 
  - Semantic search with BERT embeddings
  - Audio feature analysis (energy, valence, danceability, tempo)
  - Mood classification
  - Popularity scoring

## 📊 Dataset

- **81,000+** tracks from Spotify
- **Audio features**: energy, valence, danceability, tempo, etc.
- **Metadata**: artist, album, genres, year, mood

## 🎯 Model Performance

- **Precision@10**: ~50%+ (hybrid scoring: genre 40% + audio 40% + mood 20%)
- **Recall@10**: ~30%+
- **Embedding Dim**: 384 (BERT MiniLM)

## 📝 License

MIT License

## 👤 Author

Your Name - [GitHub](https://github.com/yourusername)

---

Made with ❤️ and 🎵
