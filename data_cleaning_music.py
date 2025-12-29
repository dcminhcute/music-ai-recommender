import pandas as pd
import numpy as np
from pathlib import Path
import pickle

print("="*60)
print("MUSIC DATA CLEANING & PREPARATION")
print("="*60)

# 1. LOAD DỮ LIỆU
input_file = 'data/music_data.csv'
df = pd.read_csv(input_file, encoding='utf-8-sig')
print(f"\n✓ Loaded: {input_file}")
print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Columns: {df.columns.tolist()}")

# 2. XỬ LÝ MISSING VALUES
print("\n" + "="*60)
print("HANDLING MISSING VALUES")
print("="*60)

# Fill missing audio features with median
audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

for feature in audio_features:
    if feature in df.columns:
        missing = df[feature].isna().sum()
        if missing > 0:
            df[feature].fillna(df[feature].median(), inplace=True)
            print(f"  {feature}: filled {missing} missing values with median")

# Fill missing genres
if 'genres' in df.columns:
    df['genres'].fillna('Unknown', inplace=True)
    print(f"  genres: filled missing with 'Unknown'")

# 3. XỬ LÝ OUTLIERS & FILTERING
print("\n" + "="*60)
print("FILTERING DATA")
print("="*60)

before_count = len(df)
print(f"Tracks before filtering: {before_count}")

df_clean = df[
    (df['popularity'] >= 20) & 
    (df['duration_ms'] >= 30000) & 
    (df['duration_ms'] <= 600000)
].copy()

after_count = len(df_clean)
removed_count = before_count - after_count

print(f"Conditions: popularity >= 20 AND 30s <= duration <= 10min")
print(f"✓ Removed: {removed_count} tracks ({removed_count/before_count*100:.1f}%)")
print(f"✓ Remaining: {after_count} tracks")

# 4. FEATURE ENGINEERING
print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

# Convert duration to minutes
df_clean['duration_min'] = df_clean['duration_ms'] / 60000

# Extract year from release_date (if available)
if 'release_date' in df_clean.columns:
    df_clean['year'] = pd.to_datetime(df_clean['release_date'], errors='coerce').dt.year
else:
    df_clean['year'] = 2023

# Create mood categories based on valence & energy
def categorize_mood(row):
    valence = row['valence']
    energy = row['energy']
    
    if pd.isna(valence) or pd.isna(energy):
        return 'Unknown'
    
    if valence > 0.6:
        if energy > 0.6:
            return 'Happy & Energetic'
        else:
            return 'Happy & Calm'
    elif valence < 0.4:
        if energy > 0.6:
            return 'Sad & Energetic'
        else:
            return 'Sad & Calm'
    else:
        if energy > 0.6:
            return 'Neutral & Energetic'
        else:
            return 'Neutral & Calm'

df_clean['mood'] = df_clean.apply(categorize_mood, axis=1)
print(f"✓ Created 'mood' feature based on valence & energy")
print(f"  Mood distribution:\n{df_clean['mood'].value_counts()}")

# Tempo categories
def categorize_tempo(tempo):
    if pd.isna(tempo):
        return 'Unknown'
    if tempo < 80:
        return 'Slow'
    elif tempo < 120:
        return 'Medium'
    else:
        return 'Fast'

df_clean['tempo_category'] = df_clean['tempo'].apply(categorize_tempo)
print(f"\n✓ Created 'tempo_category' feature")
print(f"  Tempo distribution:\n{df_clean['tempo_category'].value_counts()}")

# 5. CHUẨN BỊ TEXT CHO BERT EMBEDDINGS
print("\n" + "="*60)
print("PREPARING TEXT FOR BERT")
print("="*60)

# Kết hợp: name + artist + genres + mood
df_clean['text_for_embedding'] = (
    df_clean['name'].fillna('') + ' ' +
    df_clean['artist'].fillna('') + ' ' +
    df_clean['genres'].fillna('') + ' ' +
    df_clean['mood'].fillna('')
).str.strip()

print(f"✓ Created 'text_for_embedding' from: name + artist + genres + mood")
print(f"\nSample texts (first 100 chars):")
for i in range(min(3, len(df_clean))):
    text = df_clean['text_for_embedding'].iloc[i]
    print(f"  {i+1}. {text[:100]}...")

# 6. VECTOR HOÁ BẰNG BERT
print("\n" + "="*60)
print("BERT ENCODING")
print("="*60)

bert_success = False

try:
    from sentence_transformers import SentenceTransformer
    
    print("Loading BERT model: all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Encoding {len(df_clean)} tracks (this may take a few minutes)...")
    embeddings = model.encode(
        df_clean['text_for_embedding'].tolist(),
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True
    )
    
    print(f"✓ Embeddings shape: {embeddings.shape}")
    print(f"  Each track is represented by a {embeddings.shape[1]}-dimensional vector")
    
    # Save embeddings
    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    embeddings_file = output_dir / 'bert_embeddings_music.pkl'
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"✓ Saved embeddings to: {embeddings_file}")
    
    # Save model
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    model_file = model_dir / 'bert_model_music.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Saved BERT model to: {model_file}")
    
    # Drop text_for_embedding (không cần lưu vào CSV)
    df_clean = df_clean.drop(columns=['text_for_embedding'])
    
    bert_success = True
    
except ImportError:
    print("❌ sentence-transformers not installed!")
    print("   Run: pip install sentence-transformers")
    
except Exception as e:
    print(f"❌ Error during encoding: {e}")

# 7. LƯU DỮ LIỆU ĐÃ LÀM SẠCH
print("\n" + "="*60)
print("SAVING CLEANED DATA")
print("="*60)

output_dir = Path('data/processed')
output_dir.mkdir(exist_ok=True, parents=True)

output_file = output_dir / 'music_data_final.csv'
df_clean.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"✓ Saved: {output_file}")
print(f"  Rows: {len(df_clean)}")
print(f"  Columns: {len(df_clean.columns)}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Original tracks: {before_count}")
print(f"After cleaning: {after_count}")
print(f"Removed: {removed_count} ({removed_count/before_count*100:.1f}%)")
print(f"Year range: {df_clean['year'].min():.0f} - {df_clean['year'].max():.0f}")
print(f"Avg popularity: {df_clean['popularity'].mean():.2f}")
print(f"BERT encoding: {'✓ Success' if bert_success else '✗ Failed'}")
print("="*60)
