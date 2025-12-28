# 🎵 Music AI Recommender System

Hệ thống gợi ý nhạc thông minh sử dụng **BERT Embeddings**, **Spotify API** và **Hybrid Filtering**.

## 📋 Tính năng

- 🔍 **Semantic Search**: Tìm nhạc theo mô tả (e.g., "upbeat summer vibes", "sad piano")
- 🎯 **Content-based Recommendations**: Gợi ý dựa trên audio features & genres
- 📊 **Data Visualization**: Phân tích popularity, genres, mood, audio features
- 🎛️ **Advanced Filters**: Year, mood, popularity, genre
- 🎵 **Audio Preview**: Nghe thử 30s trực tiếp từ Spotify
- 📈 **Model Evaluation**: Precision, Recall, RMSE, MAE, F1-Score

---

## 🚀 Cài đặt

### 1. Cài đặt Python packages

```bash
pip install -r requirements.txt
```

### 2. Lấy Spotify API Credentials

1. Truy cập: https://developer.spotify.com/dashboard
2. Click **Create an App**
3. Điền thông tin app (tùy ý)
4. Copy **Client ID** và **Client Secret**
5. Paste vào file `crawl_music_data.py` (dòng 11-12):

```python
CLIENT_ID = 'your_client_id_here'
CLIENT_SECRET = 'your_client_secret_here'
```

---

## 📂 Workflow

### Bước 1: Crawl dữ liệu từ Spotify

```bash
python crawl_music_data.py
```

**Output:**
- `data/music_data_raw.json` - Dữ liệu gốc
- `data/music_data.csv` - Dữ liệu đã parse

**Thu thập:**
- ~2,000-5,000 tracks (tùy số playlists)
- 20+ playlists (Top Charts, Genres, Moods)
- Audio features: danceability, energy, valence, tempo, etc.
- Metadata: artist, album, genres, popularity, release date

---

### Bước 2: Làm sạch & chuẩn bị dữ liệu

```bash
python data_cleaning_music.py
```

**Các bước xử lý:**
1. Loại bỏ missing values
2. Filtering: popularity >= 20, duration 30s-10min
3. Feature engineering: mood, tempo_category, year
4. BERT embeddings (384 dimensions)
5. Lưu vào `data/processed/music_data_final.csv`

---

### Bước 3: Tạo visualizations

```bash
python visualization_music.py
```

**Tạo 6 biểu đồ:**
1. Popularity Distribution
2. Top 15 Artists
3. Genre Distribution
4. Audio Features Correlation
5. Mood Distribution (Pie chart)
6. Energy vs Valence Map (Scatter)

Lưu trong folder `visualizations/`

---

### Bước 4: Chạy Streamlit App

```bash
streamlit run app_music.py
```

Truy cập: http://localhost:8501

---

## 🎛️ Audio Features (Spotify)

| Feature | Mô tả | Range |
|---------|-------|-------|
| **Danceability** | Khả năng nhảy theo nhịp | 0.0 - 1.0 |
| **Energy** | Cường độ năng lượng | 0.0 - 1.0 |
| **Valence** | Độ tích cực/vui vẻ | 0.0 - 1.0 |
| **Tempo** | BPM (beats per minute) | 50 - 200 |
| **Acousticness** | Độ acoustic | 0.0 - 1.0 |
| **Instrumentalness** | Độ instrumental (không lời) | 0.0 - 1.0 |
| **Speechiness** | Độ có lời nói | 0.0 - 1.0 |
| **Liveness** | Độ live performance | 0.0 - 1.0 |
| **Loudness** | Độ lớn (dB) | -60 - 0 |

---

## 🧠 Thuật toán

### 1. BERT Semantic Search
```
Input: User description
↓
Encode với BERT (all-MiniLM-L6-v2)
↓
Cosine Similarity với tất cả tracks
↓
Hybrid Score = 0.7 × Similarity + 0.3 × Quality
```

### 2. Content-based Filtering
```
Input: Track name
↓
Find track in database
↓
Cosine Similarity với embeddings
↓
Filter by popularity, year, mood
↓
Top-N recommendations
```

### 3. Mood Classification
```
Valence (Happiness) × Energy = Mood
───────────────────────────────────
High Valence + High Energy = Happy & Energetic
High Valence + Low Energy = Happy & Calm
Low Valence + High Energy = Sad & Energetic
Low Valence + Low Energy = Sad & Calm
```

---

## 📊 Evaluation Metrics

- **Precision@K**: % recommendations có cùng genre
- **Recall@K**: % relevant tracks được gợi ý
- **RMSE**: Root Mean Square Error (popularity)
- **MAE**: Mean Absolute Error (popularity)
- **F1-Score**: Harmonic mean của Precision & Recall

---

## 🔧 Customization

### Thay đổi playlists để crawl

Edit `crawl_music_data.py`, dòng 143:

```python
playlist_ids = [
    ('playlist_id', 'Playlist Name'),
    # Thêm playlists khác...
]
```

Lấy playlist_id từ URL:
```
https://open.spotify.com/playlist/{PLAYLIST_ID}
```

### Thay đổi filters

Edit `data_cleaning_music.py`:

```python
df_clean = df[
    (df['popularity'] >= 30) &  # Tăng min popularity
    (df['duration_ms'] >= 60000)  # Min 1 phút
]
```

---

## 📁 Cấu trúc Project

```
DS/
├── crawl_music_data.py          # Crawl từ Spotify API
├── data_cleaning_music.py       # Làm sạch & BERT encoding
├── visualization_music.py       # Tạo biểu đồ
├── app_music.py                 # Streamlit app chính
├── requirements.txt             # Python dependencies
├── README_MUSIC.md             # File này
│
├── data/
│   ├── music_data_raw.json     # Dữ liệu gốc
│   ├── music_data.csv          # Dữ liệu đã parse
│   └── processed/
│       ├── music_data_final.csv        # Dữ liệu đã clean
│       └── bert_embeddings_music.pkl   # BERT vectors
│
├── models/
│   ├── bert_model_music.pkl            # BERT model
│   └── cosine_similarity_matrix_music.pkl  # Similarity matrix
│
└── visualizations/
    ├── 01_popularity_distribution.png
    ├── 02_top_artists.png
    ├── 03_genre_distribution.png
    ├── 04_audio_features_correlation.png
    ├── 05_mood_distribution.png
    └── 06_energy_valence_map.png
```

---

## ⚠️ Lưu ý

1. **Rate Limiting**: Spotify API có giới hạn requests. Script có delay để tránh bị block.
2. **Preview URLs**: Không phải track nào cũng có preview 30s.
3. **Genres**: Lấy từ artist info, không phải từ track.
4. **Dataset Size**: Càng nhiều playlists = càng nhiều tracks = dataset lớn hơn.

---

## 🔄 So sánh với Anime Version

| Feature | Anime Version | Music Version |
|---------|--------------|---------------|
| **Data Source** | Jikan API (MyAnimeList) | Spotify API |
| **Items** | ~5,000 anime | ~2,000-5,000 tracks |
| **Main Feature** | Synopsis + Genres | Audio Features + Genres |
| **Quality Metric** | Score (0-10) | Popularity (0-100) |
| **Embeddings** | Synopsis + Genres | Name + Artist + Genres + Mood |
| **Preview** | Trailer (video) | Audio preview (30s) |
| **Unique Features** | Studios, Demographics | Danceability, Energy, Valence |

---

## 💡 Cải tiến có thể

- ✅ Thêm Collaborative Filtering (user behavior)
- ✅ Lyrics analysis với NLP
- ✅ Playlist generation
- ✅ Multi-track similarity (create mixtape)
- ✅ Real-time Spotify integration (personal library)
- ✅ Fine-tune BERT on music domain
- ✅ Emotion detection từ audio signal

---

## 📝 Dependencies

```
streamlit>=1.31.0
pandas>=2.2.0
numpy>=1.26.3
sentence-transformers>=2.3.1
scikit-learn>=1.4.0
matplotlib>=3.8.2
seaborn>=0.13.1
spotipy>=2.23.0
requests>=2.31.0
```

---

## 🎯 Kết quả mong đợi

- **Precision@10**: 60-80%
- **Recall@10**: 40-60%
- **F1-Score**: 50-70%
- **RMSE**: 10-20 (popularity)
- **MAE**: 8-15 (popularity)

---

## 📧 Liên hệ

Nếu có vấn đề, check:
1. Spotify API credentials đã đúng chưa
2. Internet connection
3. Python packages đã cài đủ chưa
4. Folder `data/`, `models/`, `visualizations/` đã tồn tại chưa

---

**🎵 Happy Music Discovering! 🎵**
