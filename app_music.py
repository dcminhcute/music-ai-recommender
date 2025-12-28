import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io
from generate_track_images import generate_track_image

st.set_page_config(page_title="🎵 Music AI Recommender", layout="wide", page_icon="🎵")

@st.cache_resource
def load_assets():
    """Load dữ liệu và mô hình"""
    try:
        df = pd.read_csv("data/processed/music_data_final.csv", encoding='utf-8-sig')
        
        with open("data/processed/bert_embeddings_music.pkl", "rb") as f:
            embeddings = pickle.load(f)
        
        # Don't precompute full similarity matrix - compute on-demand to save memory
        
        model_file = Path("models/bert_model_music.pkl")
        if model_file.exists():
            with open(model_file, "rb") as f:
                model = pickle.load(f)
        else:
            st.info("Loading BERT model...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            with open(model_file, "wb") as f:
                pickle.dump(model, f)
        
        # Weighted popularity score
        C = df['popularity'].mean()
        m = df['popularity'].quantile(0.7)
        df['weighted_popularity'] = (df['popularity'] + C * m) / (1 + m)
        
        return df, embeddings, model
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

@st.cache_data
def get_track_image(track_id, track_name, artist, genres, energy, valence, danceability, popularity):
    """Get or generate track image"""
    image_path = Path(f"data/track_images/{track_id}.png")
    
    # If image exists, load it
    if image_path.exists():
        return Image.open(image_path)
    
    # Otherwise, generate on-demand
    try:
        img = generate_track_image(
            track_name=track_name,
            artist=artist,
            genres=genres,
            energy=energy,
            valence=valence,
            danceability=danceability,
            popularity=popularity
        )
        # Save for future use
        image_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(image_path)
        return img
    except Exception as e:
        # Return None if generation fails
        return None

df, embeddings, bert_model = load_assets()

if 'history' not in st.session_state:
    st.session_state.history = []

def create_song_link(name, external_url=None, artist=None):
    """Tạo link clickable cho tên bài hát"""
    if pd.notna(external_url) and external_url:
        # Link Spotify
        return f"[🎵 {name}]({external_url})"
    elif artist:
        # Fallback: link YouTube search
        search_query = f"{name} {artist}".replace(' ', '+')
        youtube_url = f"https://www.youtube.com/results?search_query={search_query}"
        return f"[🔍 {name}]({youtube_url})"
    else:
        return f"🎵 {name}"

def search_by_description(description, df, embeddings, bert_model, top_n=10, min_popularity=20):
    """Tìm nhạc theo mô tả sử dụng BERT semantic search"""
    user_vector = bert_model.encode([description])
    
    sim_scores = cosine_similarity(user_vector, embeddings)[0]
    
    results = df.copy()
    results['similarity_score'] = sim_scores
    results = results[results['popularity'] >= min_popularity]
    
    if len(results) == 0:
        return None
    
    results['similarity_normalized'] = (results['similarity_score'] - results['similarity_score'].min()) / (results['similarity_score'].max() - results['similarity_score'].min() + 1e-10)
    results['quality_normalized'] = (results['weighted_popularity'] - results['weighted_popularity'].min()) / (results['weighted_popularity'].max() - results['weighted_popularity'].min() + 1e-10)
    results['hybrid_score'] = 0.7 * results['similarity_normalized'] + 0.3 * results['quality_normalized']
    
    return results.sort_values('hybrid_score', ascending=False).head(top_n)

def get_all_matches(track_name, df):
    """Tìm tất cả tracks match"""
    try:
        matches = df[df['name'].str.contains(track_name, case=False, na=False)].copy()
        if len(matches) == 0:
            return None
        
        matches['match_score'] = matches['name'].apply(
            lambda x: 100 if x.lower() == track_name.lower() else 
                     50 if x.lower().startswith(track_name.lower()) else 
                     10
        )
        
        matches = matches.sort_values(by=['match_score', 'weighted_popularity'], ascending=[False, False])
        
        return matches
    except:
        return None

def get_hybrid_recommendations(track_name, df, embeddings, top_n=10, min_popularity=20):
    try:
        idx = df[df['name'].str.contains(track_name, case=False, na=False)].index[0]
    except IndexError:
        return None, None
    
    input_track = df.iloc[idx]
    
    # Compute similarity on-demand for this track only
    track_embedding = embeddings[idx].reshape(1, -1)
    sim_scores = cosine_similarity(track_embedding, embeddings)[0]
    
    # Create list of (index, similarity_score)
    sim_scores_list = list(enumerate(sim_scores))
    sim_scores_list = sorted(sim_scores_list, key=lambda x: x[1], reverse=True)[1:]
    
    track_indices = [i[0] for i in sim_scores_list]
    recommendations = df.iloc[track_indices].copy()
    recommendations['similarity_score'] = [i[1] for i in sim_scores_list]
    recommendations = recommendations[recommendations['popularity'] >= min_popularity]
    
    if len(recommendations) == 0:
        return input_track, None
    
    recommendations['similarity_normalized'] = (recommendations['similarity_score'] - recommendations['similarity_score'].min()) / (recommendations['similarity_score'].max() - recommendations['similarity_score'].min() + 1e-10)
    recommendations['quality_normalized'] = (recommendations['weighted_popularity'] - recommendations['weighted_popularity'].min()) / (recommendations['weighted_popularity'].max() - recommendations['weighted_popularity'].min() + 1e-10)
    recommendations['hybrid_score'] = 0.7 * recommendations['similarity_normalized'] + 0.3 * recommendations['quality_normalized']
    
    return input_track, recommendations.sort_values('hybrid_score', ascending=False).head(top_n)

def calculate_precision_at_k(track_name, recommendations, df, k=10):
    """Tính Precision@K dựa trên hybrid scoring: genre + audio features + mood"""
    try:
        idx = df[df['name'].str.contains(track_name, case=False, na=False)].index[0]
        input_track = df.iloc[idx]
        
        if recommendations is None or len(recommendations) == 0:
            return 0.0
        
        relevant = 0
        for _, row in recommendations.head(k).iterrows():
            score = 0
            
            # 1. Genre matching (40%)
            input_genres_str = str(input_track['genres'])
            rec_genres_str = str(row['genres'])
            if pd.notna(input_genres_str) and input_genres_str not in ['nan', 'Unknown'] and \
               pd.notna(rec_genres_str) and rec_genres_str not in ['nan', 'Unknown']:
                input_genres = set([g.strip() for g in input_genres_str.split(',') if g.strip()])
                rec_genres = set([g.strip() for g in rec_genres_str.split(',') if g.strip()])
                if len(input_genres & rec_genres) > 0:
                    score += 0.4
            
            # 2. Audio features similarity (40%)
            audio_features = ['energy', 'valence', 'danceability', 'tempo']
            audio_sim = 0
            valid_features = 0
            for feature in audio_features:
                if pd.notna(input_track.get(feature)) and pd.notna(row.get(feature)):
                    # Normalize to 0-1 and calculate similarity
                    if feature == 'tempo':
                        diff = abs(input_track[feature] - row[feature]) / 200  # Normalize tempo
                    else:
                        diff = abs(input_track[feature] - row[feature])
                    audio_sim += max(0, 1 - diff)
                    valid_features += 1
            if valid_features > 0:
                score += 0.4 * (audio_sim / valid_features)
            
            # 3. Mood matching (20%)
            if pd.notna(input_track.get('mood')) and pd.notna(row.get('mood')):
                if input_track['mood'] == row['mood']:
                    score += 0.2
            
            # Consider relevant if score > 0.5 (at least 2 out of 3 criteria match)
            if score >= 0.5:
                relevant += 1
        
        return relevant / k
    except Exception as e:
        return 0.0

def calculate_recall_at_k(track_name, recommendations, df, k=10):
    """Tính Recall@K dựa trên hybrid scoring: genre + audio features + mood"""
    try:
        idx = df[df['name'].str.contains(track_name, case=False, na=False)].index[0]
        input_track = df.iloc[idx]
        
        # Count total relevant tracks in dataset (sample for efficiency)
        sample_size = min(1000, len(df))  # Sample to avoid long computation
        df_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
        
        total_relevant = 0
        for _, row in df_sample.iterrows():
            if row.name == idx:  # Skip the input track itself
                continue
            
            score = 0
            
            # Genre matching
            input_genres_str = str(input_track['genres'])
            rec_genres_str = str(row['genres'])
            if pd.notna(input_genres_str) and input_genres_str not in ['nan', 'Unknown'] and \
               pd.notna(rec_genres_str) and rec_genres_str not in ['nan', 'Unknown']:
                input_genres = set([g.strip() for g in input_genres_str.split(',') if g.strip()])
                rec_genres = set([g.strip() for g in rec_genres_str.split(',') if g.strip()])
                if len(input_genres & rec_genres) > 0:
                    score += 0.4
            
            # Audio features similarity
            audio_features = ['energy', 'valence', 'danceability']
            audio_sim = 0
            valid_features = 0
            for feature in audio_features:
                if pd.notna(input_track.get(feature)) and pd.notna(row.get(feature)):
                    diff = abs(input_track[feature] - row[feature])
                    audio_sim += max(0, 1 - diff)
                    valid_features += 1
            if valid_features > 0:
                score += 0.4 * (audio_sim / valid_features)
            
            # Mood matching
            if pd.notna(input_track.get('mood')) and pd.notna(row.get('mood')):
                if input_track['mood'] == row['mood']:
                    score += 0.2
            
            if score >= 0.5:
                total_relevant += 1
        
        if recommendations is None or len(recommendations) == 0 or total_relevant == 0:
            return 0.0
        
        # Count retrieved relevant
        retrieved_relevant = 0
        for _, row in recommendations.head(k).iterrows():
            score = 0
            
            # Same scoring logic
            input_genres_str = str(input_track['genres'])
            rec_genres_str = str(row['genres'])
            if pd.notna(input_genres_str) and input_genres_str not in ['nan', 'Unknown'] and \
               pd.notna(rec_genres_str) and rec_genres_str not in ['nan', 'Unknown']:
                input_genres = set([g.strip() for g in input_genres_str.split(',') if g.strip()])
                rec_genres = set([g.strip() for g in rec_genres_str.split(',') if g.strip()])
                if len(input_genres & rec_genres) > 0:
                    score += 0.4
            
            audio_features = ['energy', 'valence', 'danceability']
            audio_sim = 0
            valid_features = 0
            for feature in audio_features:
                if pd.notna(input_track.get(feature)) and pd.notna(row.get(feature)):
                    diff = abs(input_track[feature] - row[feature])
                    audio_sim += max(0, 1 - diff)
                    valid_features += 1
            if valid_features > 0:
                score += 0.4 * (audio_sim / valid_features)
            
            if pd.notna(input_track.get('mood')) and pd.notna(row.get('mood')):
                if input_track['mood'] == row['mood']:
                    score += 0.2
            
            if score >= 0.5:
                retrieved_relevant += 1
        
        return retrieved_relevant / min(total_relevant, k) if total_relevant > 0 else 0.0
    except Exception as e:
        return 0.0

def calculate_rmse_mae(track_name, recommendations, df):
    """Tính RMSE và MAE dựa trên popularity similarity (normalized)"""
    try:
        idx = df[df['name'].str.contains(track_name, case=False, na=False)].index[0]
        input_popularity = df.iloc[idx]['popularity']
        
        if recommendations is None or len(recommendations) == 0:
            return 0.0, 0.0
        
        rec_popularity = recommendations['popularity'].values
        # Normalize errors by dividing by 100 (max popularity)
        errors = (rec_popularity - input_popularity) / 100.0
        
        rmse = np.sqrt(np.mean(errors ** 2)) * 100  # Scale back for display
        mae = np.mean(np.abs(errors)) * 100
        
        return rmse, mae
    except Exception as e:
        return 0.0, 0.0

# SIDEBAR
st.sidebar.title("🎵 Filters & Settings")

st.sidebar.subheader("📍 Filter Options")
year_range = st.sidebar.slider("Release Year", int(df['year'].min()), int(df['year'].max()), (2000, int(df['year'].max())))
min_popularity = st.sidebar.slider("Minimum Popularity", 0, 100, 20, 5)
top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

# Mood filter
mood_options = ['All'] + list(df['mood'].unique())
selected_mood = st.sidebar.selectbox("Mood Filter", mood_options)

st.sidebar.markdown("---")
st.sidebar.subheader("📜 Search History")
if st.session_state.history:
    for item in reversed(st.session_state.history[-5:]):
        if st.sidebar.button(f"🔄 {item[:30]}...", key=f"hist_{item}"):
            st.session_state.search_query = item
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.sidebar.caption("No history yet.")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Info")
st.sidebar.metric("Total Tracks", f"{len(df):,}")
st.sidebar.metric("Avg Popularity", f"{df['popularity'].mean():.1f}")
st.sidebar.metric("Unique Artists", f"{df['artist'].nunique():,}")

# MAIN APP
st.title("🎵 Music AI Recommender")
st.markdown("*Powered by BERT Embeddings & Spotify API*")

tab1, tab2, tab3 = st.tabs(["🔍 Find Music", "📊 Data Analysis", "🎯 Model Evaluation"])

with tab1:
    st.subheader("🎵 Find music that matches your taste")
    
    col_input, col_random = st.columns([4, 1])
    
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    
    if 'last_search' not in st.session_state:
        st.session_state.last_search = None
    
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    
    user_input = col_input.text_input(
        "Enter song name or description:", 
        value=st.session_state.search_query,
        placeholder="e.g., Shape of You, upbeat pop song, relaxing piano..."
    )
    
    random_btn = col_random.button("🎲 Random")
    
    # Clear old results if new search
    if st.button("🔍 Search", type="primary") or random_btn:
        target = user_input if not random_btn else df.sample(1).iloc[0]['name']
        
        if target:
            # Clear previous results immediately
            if target != st.session_state.last_search:
                st.session_state.search_results = None
                st.session_state.last_search = target
            
            if target not in st.session_state.history:
                st.session_state.history.append(target)
            
            with st.spinner("Analyzing..."):
                all_matches = get_all_matches(target, df)
                st.session_state.search_results = {'target': target, 'matches': all_matches}
    
    # Display results only from session state
    if st.session_state.search_results:
        results = st.session_state.search_results
        target = results['target']
        all_matches = results['matches']
        
        if all_matches is not None and len(all_matches) > 1:
            st.info(f"🔍 Found **{len(all_matches)}** tracks matching '{target}'. Please select:")
            
            for i in range(0, min(len(all_matches), 20), 5):
                cols = st.columns(5)
                for j, (idx, row) in enumerate(all_matches.iloc[i:i+5].iterrows()):
                    with cols[j]:
                        # Generate and display track image
                        img = get_track_image(
                            track_id=row['track_id'],
                            track_name=row['name'],
                            artist=row['artist'],
                            genres=row.get('genres', 'Music'),
                            energy=row.get('energy', 0.5),
                            valence=row.get('valence', 0.5),
                            danceability=row.get('danceability', 0.5),
                            popularity=row.get('popularity', 50)
                        )
                        if img:
                            st.image(img, use_container_width=True)
                        
                        # Create clickable song link
                        song_link = create_song_link(row['name'][:30], row.get('external_url'), row['artist'])
                        st.markdown(f"**{song_link}**", unsafe_allow_html=True)
                        st.caption(f"🎤 {row['artist'][:25]}")
                        st.caption(f"⭐ {row['popularity']}")
                        
                        if pd.notna(row['genres']):
                            genres = str(row['genres']).split(',')[:2]
                            st.info(" • ".join([g.strip()[:15] for g in genres]))
        
        elif 'selected_track' in st.session_state or (all_matches is not None and len(all_matches) == 1):
            if 'selected_track' in st.session_state:
                target = st.session_state.selected_track
                del st.session_state.selected_track
            
            input_track, recommendations = get_hybrid_recommendations(target, df, embeddings, top_n, min_popularity)
            
            if input_track is None or recommendations is None or len(recommendations) == 0:
                st.warning("⚠️ No matching tracks found with current filters!")
            else:
                st.success(f"✅ Found: **{input_track['name']}**")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Popularity", f"{input_track['popularity']}/100")
                col2.metric("Year", int(input_track['year']) if pd.notna(input_track['year']) else 'N/A')
                col3.metric("Energy", f"{input_track['energy']:.2f}" if pd.notna(input_track['energy']) else 'N/A')
                col4.metric("Valence", f"{input_track['valence']:.2f}" if pd.notna(input_track['valence']) else 'N/A')
                
                with st.expander("📖 View Details"):
                    col_img, col_detail = st.columns([1, 2])
                    
                    with col_img:
                        # Display generated track image
                        img = get_track_image(
                            track_id=input_track['track_id'],
                            track_name=input_track['name'],
                            artist=input_track['artist'],
                            genres=input_track.get('genres', 'Music'),
                            energy=input_track.get('energy', 0.5),
                            valence=input_track.get('valence', 0.5),
                            danceability=input_track.get('danceability', 0.5),
                            popularity=input_track.get('popularity', 50)
                        )
                        if img:
                            st.image(img, width=200)
                    
                    with col_detail:
                        st.markdown(f"**Artist:** {input_track['artist']}")
                        st.markdown(f"**Album:** {input_track['album']}")
                        if pd.notna(input_track['genres']):
                            st.markdown(f"**Genres:** {input_track['genres']}")
                        if pd.notna(input_track.get('mood')):
                            st.markdown(f"**Mood:** {input_track['mood']}")
                        st.markdown(f"**Duration:** {input_track['duration_min']:.2f} min")
                        
                        # Add Spotify/YouTube links
                        if pd.notna(input_track.get('external_url')) and input_track['external_url']:
                            st.markdown(f"🎵 [Open in Spotify]({input_track['external_url']})")
                        else:
                            search_query = f"{input_track['name']} {input_track['artist']}".replace(' ', '+')
                            youtube_url = f"https://www.youtube.com/results?search_query={search_query}"
                            st.markdown(f"🔍 [Search on YouTube]({youtube_url})")
                            youtube_url = f"https://www.youtube.com/results?search_query={search_query}"
                            st.markdown(f"🔍 [Search on YouTube]({youtube_url})")
                
                st.markdown("---")
                st.subheader(f"🎯 Top {min(len(recommendations), top_n)} Recommendations")
                
                filtered = recommendations[(recommendations['year'] >= year_range[0]) & (recommendations['year'] <= year_range[1])]
                
                if selected_mood != 'All':
                    filtered = filtered[filtered['mood'] == selected_mood]
                
                if len(filtered) == 0:
                    st.warning("⚠️ No tracks match your filters. Showing original results:")
                    filtered = recommendations
                
                for i in range(0, len(filtered), 5):
                    cols = st.columns(5)
                    for j, (idx, row) in enumerate(filtered.iloc[i:i+5].iterrows()):
                        with cols[j]:
                                    # Generate and display track image
                                    img = get_track_image(
                                        track_id=row['track_id'],
                                        track_name=row['name'],
                                        artist=row['artist'],
                                        genres=row.get('genres', 'Music'),
                                        energy=row.get('energy', 0.5),
                                        valence=row.get('valence', 0.5),
                                        danceability=row.get('danceability', 0.5),
                                        popularity=row.get('popularity', 50)
                                    )
                                    if img:
                                        st.image(img, use_container_width=True)
                                    
                                    st.markdown(f"**{i+j+1}. {row['name'][:30]}**")
                                    st.caption(f"🎤 {row['artist'][:25]}")
                                    st.caption(f"⭐ {row['popularity']} | 🎯 {row['hybrid_score']:.3f}")
                                    
                                    if pd.notna(row['genres']):
                                        genres = str(row['genres']).split(',')[:2]
                                        st.info(" • ".join([g.strip()[:15] for g in genres]))
        
        else:
            st.info(f"🔍 No exact match for '{target}'. Searching by description...")
            
            description_results = search_by_description(target, df, embeddings, bert_model, top_n, min_popularity)
            
            if description_results is not None and len(description_results) > 0:
                st.success(f"✅ Found {len(description_results)} tracks matching your description!")
                
                for i in range(0, len(description_results), 5):
                    cols = st.columns(5)
                    for j, (idx, row) in enumerate(description_results.iloc[i:i+5].iterrows()):
                        with cols[j]:
                                    # Generate and display track image
                                    img = get_track_image(
                                        track_id=row['track_id'],
                                        track_name=row['name'],
                                        artist=row['artist'],
                                        genres=row.get('genres', 'Music'),
                                        energy=row.get('energy', 0.5),
                                        valence=row.get('valence', 0.5),
                                        danceability=row.get('danceability', 0.5),
                                        popularity=row.get('popularity', 50)
                                    )
                                    if img:
                                        st.image(img, use_container_width=True)
                                    
                                    # Create clickable song link
                                    song_link = create_song_link(row['name'][:30], row.get('external_url'), row['artist'])
                                    st.markdown(f"**{i+j+1}. {song_link}**", unsafe_allow_html=True)
                                    st.caption(f"🎤 {row['artist'][:25]}")
                                    st.caption(f"⭐ {row['popularity']} | 🎯 {row['hybrid_score']:.3f}")
                                    
                                    if pd.notna(row['genres']):
                                        genres = str(row['genres']).split(',')[:2]
                                        st.info(" • ".join([g.strip()[:15] for g in genres]))
            else:
                st.error(f"❌ No tracks found matching: {target}")

with tab2:
    st.header("📊 Data Analysis")
    st.markdown("*Khám phá và phân tích dữ liệu âm nhạc từ Spotify*")
    st.markdown("---")
    
    # Overview metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("📀 Total Tracks", f"{len(df):,}")
    with col_m2:
        st.metric("🎤 Unique Artists", f"{df['artist'].nunique():,}")
    with col_m3:
        st.metric("⭐ Avg Popularity", f"{df['popularity'].mean():.1f}/100")
    with col_m4:
        st.metric("🎵 Genres", f"{df['genres'].nunique():,}")
    
    st.markdown("---")
    
    viz_dir = Path('visualizations')
    if viz_dir.exists():
        col1, col2 = st.columns(2)
        
        with col1:
            if (viz_dir / '01_popularity_distribution.png').exists():
                st.subheader("🔥 Phân phối độ phổ biến")
                st.caption("Biểu đồ cho thấy hầu hết các bài hát có độ phổ biến từ 30-60. Các bài hit lớn (>80) khá hiếm.")
                st.image(str(viz_dir / '01_popularity_distribution.png'))
            
            if (viz_dir / '03_genre_distribution.png').exists():
                st.subheader("🎸 Top 15 thể loại nhạc")
                st.caption("Thống kê các thể loại nhạc xuất hiện nhiều nhất trong dataset, giúp hiểu xu hướng âm nhạc.")
                st.image(str(viz_dir / '03_genre_distribution.png'))
            
            if (viz_dir / '05_mood_distribution.png').exists():
                st.subheader("😊 Phân phối tâm trạng")
                st.caption("Phân loại các bài hát theo tâm trạng dựa trên energy và valence (năng lượng và cảm xúc tích cực).")
                st.image(str(viz_dir / '05_mood_distribution.png'))
        
        with col2:
            if (viz_dir / '02_top_artists.png').exists():
                st.subheader("⭐ Top 15 nghệ sĩ")
                st.caption("Nghệ sĩ có nhiều bài hát nhất trong dataset. Thể hiện sự đa dạng và phong phú của từng nghệ sĩ.")
                st.image(str(viz_dir / '02_top_artists.png'))
            
            if (viz_dir / '04_audio_features_correlation.png').exists():
                st.subheader("🔗 Mối tương quan đặc trưng âm thanh")
                st.caption("Ma trận tương quan giữa các đặc trưng như energy, valence, tempo. Màu đỏ = tương quan dương, xanh = âm.")
                st.image(str(viz_dir / '04_audio_features_correlation.png'))
            
            if (viz_dir / '06_energy_valence_map.png').exists():
                st.subheader("🗺️ Bản đồ Energy-Valence")
                st.caption("Scatter plot phân loại nhạc theo năng lượng (trục Y) và cảm xúc (trục X). Phải trên = vui vẻ & sôi động.")
                st.image(str(viz_dir / '06_energy_valence_map.png'))
    else:
        st.warning("No visualizations found. Run visualization_music.py first!")

with tab3:
    st.header("🎯 Model Performance Evaluation")
    st.markdown("*Đánh giá độ chính xác của hệ thống gợi ý nhạc sử dụng BERT embeddings*")
    st.markdown(f"**📊 Dataset:** {len(df):,} tracks | **🧪 Test:** 10 tracks (5 popular + 5 random)")
    
    # Explanation section
    with st.expander("📖 Giải thích các chỉ số đánh giá", expanded=False):
        st.markdown("""
        **🎯 Precision@10**: Tỷ lệ bài hát được gợi ý phù hợp (dựa trên Genre 40% + Audio Features 40% + Mood 20%)
        - ≥ 0.5 điểm được tính là relevant (ít nhất 2/3 tiêu chí khớp)
        - Cao = gợi ý chính xác, toàn diện
        
        **🔍 Recall@10**: Tỷ lệ bài hát liên quan được tìm thấy so với tổng số bài liên quan
        - Cao = tìm được nhiều bài phù hợp
        
        **📊 F1-Score**: Điểm cân bằng giữa Precision và Recall
        - Cao = cả hai chỉ số đều tốt
        
        **📉 RMSE** (Root Mean Square Error): Sai số trung bình về độ phổ biến (normalized)
        - Thấp = gợi ý có độ phổ biến tương đồng
        
        **📈 MAE** (Mean Absolute Error): Sai số tuyệt đối trung bình (normalized)
        - Thấp = dự đoán chính xác hơn
        
        **💡 Cải tiến**: Sử dụng hybrid scoring thay vì chỉ genre matching đơn thuần
        """)
    
    st.markdown("---")
    
    # Sample tracks for testing - diverse selection
    popular_tracks = df.nlargest(5, 'popularity')['name'].tolist()
    random_tracks = df.sample(5, random_state=42)['name'].tolist()
    test_tracks = popular_tracks + random_tracks
    
    precisions = []
    recalls = []
    rmse_list = []
    mae_list = []
    valid_tracks = []
    
    with st.spinner("🔄 Đang tính toán các chỉ số đánh giá..."):
        for track in test_tracks:
            _, recs = get_hybrid_recommendations(track, df, embeddings, 10, 0)
            
            if recs is not None and len(recs) > 0:
                precision = calculate_precision_at_k(track, recs, df, 10)
                recall = calculate_recall_at_k(track, recs, df, 10)
                rmse, mae = calculate_rmse_mae(track, recs, df)
                
                precisions.append(precision)
                recalls.append(recall)
                rmse_list.append(rmse)
                mae_list.append(mae)
                valid_tracks.append(track[:20])
    
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    avg_rmse = np.mean(rmse_list) if rmse_list else 0
    avg_mae = np.mean(mae_list) if mae_list else 0
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    st.subheader("📊 Kết quả đánh giá tổng quan")
    
    # Add success message if metrics improved
    if avg_precision >= 0.5:
        st.success("""
        ✅ **Kết quả tốt!** Model đang hoạt động hiệu quả với hybrid scoring:
        - **Genre matching** (40%): Đảm bảo cùng thể loại hoặc tương đồng
        - **Audio features** (40%): Energy, Valence, Danceability, Tempo tương đồng
        - **Mood matching** (20%): Cùng tâm trạng (Happy, Sad, Energetic, Calm)
        
        💡 Model gợi ý dựa trên nhiều yếu tố, không chỉ genre!
        """)
    elif avg_precision >= 0.3:
        st.info("""
        ℹ️ **Kết quả khá tốt:** Model đang hoạt động ổn định
        - Precision/Recall ở mức trung bình cao cho thấy gợi ý phù hợp
        - Hybrid scoring giúp tìm bài tương đồng về âm thanh, không chỉ thể loại
        """)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision@10", f"{avg_precision*100:.1f}%", help="Tỷ lệ gợi ý chính xác (Genre 40% + Audio 40% + Mood 20%)")
    col2.metric("Recall@10", f"{avg_recall*100:.1f}%", help="Tỷ lệ bài hát liên quan được tìm thấy")
    col3.metric("RMSE", f"{avg_rmse:.3f}", help="Sai số về độ phổ biến (càng thấp càng tốt)")
    col4.metric("MAE", f"{avg_mae:.3f}", help="Sai số tuyệt đối trung bình (càng thấp càng tốt)")
    
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("F1-Score", f"{f1_score*100:.1f}%", help="Điểm cân bằng giữa Precision & Recall")
    col6.metric("Test Tracks", len(valid_tracks), help="Số bài hát dùng để test")
    col7.metric("Dataset Size", f"{len(df):,}", help="Tổng số bài trong dataset")
    col8.metric("Embedding Dim", "384", help="Số chiều của BERT embedding vector")
    
    st.markdown("---")
    
    if precisions and recalls:
        st.subheader("📈 Chi tiết kết quả theo từng bài test")
        st.caption("So sánh hiệu suất của model trên các bài hát phổ biến nhất")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        x = np.arange(len(valid_tracks))
        width = 0.35
        
        bars1 = axes[0].bar(x - width/2, [p*100 for p in precisions], width, label='Precision@10', color='steelblue')
        bars2 = axes[0].bar(x + width/2, [r*100 for r in recalls[:len(precisions)]], width, label='Recall@10', color='coral')
        
        axes[0].set_ylabel('Score (%)', fontsize=11)
        axes[0].set_xlabel('Track', fontsize=11)
        axes[0].set_title('Precision@10 vs Recall@10', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(valid_tracks, rotation=45, ha='right')
        axes[0].legend()
        axes[0].set_ylim(0, 110)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        if rmse_list and mae_list:
            bars3 = axes[1].bar(x - width/2, rmse_list[:len(valid_tracks)], width, label='RMSE', color='forestgreen')
            bars4 = axes[1].bar(x + width/2, mae_list[:len(valid_tracks)], width, label='MAE', color='gold')
            
            axes[1].set_ylabel('Error Score', fontsize=11)
            axes[1].set_xlabel('Track', fontsize=11)
            axes[1].set_title('RMSE vs MAE', fontsize=12, fontweight='bold')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(valid_tracks, rotation=45, ha='right')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🎵 <b>Music AI Recommender</b> | Powered by BERT & Spotify API</p>
    <p>Dataset: {n_tracks:,} tracks | Model: Content-based + Audio Features</p>
</div>
""".format(n_tracks=len(df)), unsafe_allow_html=True)
