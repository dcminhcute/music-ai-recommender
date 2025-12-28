# 🚀 QUICK START GUIDE - MUSIC RECOMMENDER

## ✅ ĐÃ CÀI ĐẶT

- [x] Python packages (streamlit, pandas, spotipy, sentence-transformers, etc.)
- [x] Virtual environment (venv)

---

## 📋 BƯỚC TIẾP THEO

### 1. Lấy Spotify API Credentials (BẮT BUỘC)

**Truy cập:** https://developer.spotify.com/dashboard

**Các bước:**
1. Login với Spotify account (hoặc tạo mới)
2. Click **"Create an App"**
3. Điền:
   - App Name: `Music Recommender` (hoặc tên tùy ý)
   - App Description: `AI Music Recommendation System`
   - Redirect URI: `http://localhost` (không quan trọng)
4. Tick checkbox **"I understand..."**
5. Click **"Create"**
6. Vào app vừa tạo → Click **"Settings"**
7. Copy:
   - ✅ **Client ID**
   - ✅ **Client Secret** (click "View client secret")

---

### 2. Cập nhật Credentials vào Code

**Mở file:** `crawl_music_data.py`

**Tìm dòng 11-12:**
```python
CLIENT_ID = 'YOUR_CLIENT_ID_HERE'
CLIENT_SECRET = 'YOUR_CLIENT_SECRET_HERE'
```

**Thay thế bằng credentials của bạn:**
```python
CLIENT_ID = 'abc123def456...'  # Paste Client ID
CLIENT_SECRET = 'xyz789uvw...'  # Paste Client Secret
```

**💾 Save file!**

---

### 3. Chạy Pipeline (3 bước)

#### 📥 **BƯỚC 1: Crawl data từ Spotify**
```bash
python crawl_music_data.py
```
⏱️ Thời gian: ~5-10 phút (tùy số playlists)  
📊 Kết quả: ~2,000-5,000 tracks

---

#### 🧹 **BƯỚC 2: Clean & prepare data**
```bash
python data_cleaning_music.py
```
⏱️ Thời gian: ~2-5 phút (BERT encoding)  
🤖 Tạo embeddings 384-dimensional vectors

---

#### 📊 **BƯỚC 3: Create visualizations**
```bash
python visualization_music.py
```
⏱️ Thời gian: ~30 giây  
🖼️ Tạo 6 biểu đồ

---

### 4. Chạy Streamlit App

```bash
streamlit run app_music.py
```

🌐 **Truy cập:** http://localhost:8501

---

## 🎯 Tính năng App

### Tab 1: 🔍 Find Music
- Tìm kiếm theo tên bài hát
- Semantic search (mô tả: "upbeat summer song")
- Lọc theo năm, mood, popularity
- Nghe preview 30s
- Link Spotify trực tiếp

### Tab 2: 📊 Data Analysis
- Popularity distribution
- Top artists & genres
- Audio features correlation
- Mood distribution
- Energy vs Valence map

### Tab 3: 🎯 Model Evaluation
- Precision@10, Recall@10
- RMSE, MAE, F1-Score
- Per-track metrics
- Visual comparisons

---

## ⚠️ TROUBLESHOOTING

### Lỗi: "Error initializing Spotify API"
**Nguyên nhân:** Credentials sai hoặc chưa cập nhật  
**Giải pháp:** Check lại Client ID & Secret trong `crawl_music_data.py`

---

### Lỗi: "FileNotFoundError: music_data_final.csv"
**Nguyên nhân:** Chưa chạy crawl + cleaning  
**Giải pháp:** Chạy lần lượt:
```bash
python crawl_music_data.py
python data_cleaning_music.py
```

---

### Lỗi: "ReadTimeoutError" khi crawl
**Nguyên nhân:** Internet chậm hoặc Spotify API timeout  
**Giải pháp:** Chạy lại script, nó sẽ skip errors và tiếp tục

---

### App không hiển thị visualizations
**Nguyên nhân:** Chưa chạy visualization script  
**Giải pháp:**
```bash
python visualization_music.py
```

---

## 📁 Cấu trúc Files

```
DS/
├── 🆕 crawl_music_data.py       ← Crawl từ Spotify
├── 🆕 data_cleaning_music.py    ← Clean & BERT
├── 🆕 visualization_music.py    ← Tạo charts
├── 🆕 app_music.py              ← Streamlit app
├── 🆕 README_MUSIC.md          ← Full documentation
├── 🆕 SETUP_MUSIC.md           ← File này
│
├── 📁 data/
│   ├── music_data_raw.json
│   ├── music_data.csv
│   └── processed/
│       ├── music_data_final.csv
│       └── bert_embeddings_music.pkl
│
├── 📁 models/
│   ├── bert_model_music.pkl
│   └── cosine_similarity_matrix_music.pkl
│
└── 📁 visualizations/
    └── (6 PNG files)
```

---

## 🔄 So sánh Anime vs Music Version

| Aspect | Anime (Cũ) | Music (Mới) |
|--------|-----------|------------|
| Data Source | Jikan API | **Spotify API** |
| Items | Anime shows | **Songs/Tracks** |
| Main Files | `app.py` | **`app_music.py`** |
| Crawl | `crawl_anime_data.py` | **`crawl_music_data.py`** |
| Data | `anime_data_final.csv` | **`music_data_final.csv`** |
| Preview | Video trailer | **Audio 30s** |
| Features | Synopsis, genres | **Audio features (energy, valence, etc.)** |

---

## 🎵 Audio Features Explained

- **Danceability** (0-1): Khả năng nhảy theo
- **Energy** (0-1): Cường độ, sôi động
- **Valence** (0-1): Tích cực, vui vẻ (0 = buồn, 1 = vui)
- **Tempo**: BPM (beats/minute)
- **Acousticness** (0-1): Độ acoustic
- **Instrumentalness** (0-1): Có lời hay không
- **Speechiness** (0-1): Độ có lời nói
- **Liveness** (0-1): Độ live performance

---

## 💡 Tips

### Tăng số lượng tracks
Edit `crawl_music_data.py` → Thêm playlists vào list (dòng 143)

### Custom filters
Edit `data_cleaning_music.py` → Thay đổi điều kiện lọc (dòng 42-44)

### Thay đổi model
Edit `data_cleaning_music.py` & `app_music.py` → Đổi `'all-MiniLM-L6-v2'` thành model khác

---

## ✅ Checklist

- [ ] Có Spotify account
- [ ] Đã tạo Spotify Developer App
- [ ] Đã copy Client ID & Secret
- [ ] Đã paste vào `crawl_music_data.py`
- [ ] Chạy `python crawl_music_data.py` ✅
- [ ] Chạy `python data_cleaning_music.py` ✅
- [ ] Chạy `python visualization_music.py` ✅
- [ ] Chạy `streamlit run app_music.py` ✅
- [ ] App mở tại http://localhost:8501 🎉

---

## 🎯 Expected Results

### Dataset
- ✅ 2,000-5,000 unique tracks
- ✅ 100-500 unique artists
- ✅ 15-30 genres
- ✅ Year range: 1960s-2024

### Performance
- ✅ Precision@10: 60-80%
- ✅ Recall@10: 40-60%
- ✅ F1-Score: 50-70%
- ✅ BERT embeddings: 384 dims

---

**🎵 Enjoy your Music AI Recommender! 🎵**

Need help? Check [README_MUSIC.md](README_MUSIC.md) for full documentation.
