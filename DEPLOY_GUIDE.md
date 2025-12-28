# 🚀 Hướng dẫn Deploy lên Streamlit Cloud

## Bước 1: Push code lên GitHub

### 1.1. Khởi tạo Git repository
```bash
cd C:\Users\Laptop\Downloads\DS
git init
git add .
git commit -m "Initial commit: Music AI Recommender"
```

### 1.2. Tạo GitHub repository
1. Truy cập https://github.com/new
2. Đặt tên repo: `music-ai-recommender`
3. Chọn **Public** (bắt buộc cho Streamlit Cloud miễn phí)
4. **KHÔNG** chọn "Initialize this repository with README"
5. Click **Create repository**

### 1.3. Push code lên GitHub
```bash
git remote add origin https://github.com/YOUR_USERNAME/music-ai-recommender.git
git branch -M main
git push -u origin main
```

**⚠️ LƯU Ý**: Files lớn như `bert_embeddings_music.pkl` và `bert_model_music.pkl` đã được thêm vào `.gitignore`. App sẽ tự động download/generate khi deploy.

---

## Bước 2: Deploy lên Streamlit Cloud

### 2.1. Đăng ký Streamlit Cloud
1. Truy cập: https://streamlit.io/cloud
2. Click **Sign up** và đăng nhập bằng GitHub account
3. Cho phép Streamlit truy cập GitHub repos

### 2.2. Deploy app
1. Click **New app**
2. Chọn repository: `YOUR_USERNAME/music-ai-recommender`
3. Branch: `main`
4. Main file path: `app_music.py`
5. Click **Deploy!**

### 2.3. Chờ deployment hoàn tất
- Quá trình này mất khoảng **5-10 phút**
- Streamlit sẽ:
  - Install dependencies từ `requirements.txt`
  - Download BERT model (~100MB)
  - Generate embeddings (có thể lâu)

---

## Bước 3: Tối ưu cho Cloud Deployment

### 3.1. Giảm kích thước dataset (nếu cần)
```python
# Trong app_music.py, thêm cache để tránh reload nhiều lần
@st.cache_data(ttl=3600)
def load_data():
    return pd.read_csv("data/processed/music_data_final.csv")
```

### 3.2. Bật secrets management (nếu có API keys)
1. Trong Streamlit Cloud dashboard, click app của bạn
2. Click **Settings** → **Secrets**
3. Thêm secrets (format TOML):
```toml
[spotify]
client_id = "your_client_id"
client_secret = "your_secret"
```

---

## 🌐 Link ứng dụng

Sau khi deploy thành công, bạn sẽ có link dạng:
```
https://YOUR_USERNAME-music-ai-recommender-app-music-abc123.streamlit.app
```

---

## 🔧 Troubleshooting

### Lỗi "File too large"
- GitHub giới hạn file 100MB
- Streamlit Cloud giới hạn 1GB tổng
- **Giải pháp**: Sử dụng Git LFS hoặc download files lớn khi runtime

### App chạy chậm
- Streamlit Cloud có resource giới hạn
- **Giải pháp**: 
  - Giảm số lượng tracks trong dataset
  - Cache aggressive hơn
  - Optimize embeddings

### Out of memory
- **Giải pháp**: 
  - Load embeddings on-demand thay vì toàn bộ
  - Sử dụng float16 thay vì float32
  - Sample dataset nhỏ hơn

---

## 📱 Chia sẻ app

Sau khi deploy thành công, bạn có thể:
- ✅ Chia sẻ link công khai với bất kỳ ai
- ✅ Embed vào website
- ✅ Share trên social media
- ✅ Không cần chạy local nữa!

---

## 🎉 Chúc mừng!

App của bạn đã có thể truy cập từ bất kỳ đâu trên thế giới! 🌍
