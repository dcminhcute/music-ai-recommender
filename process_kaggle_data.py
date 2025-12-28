"""
Process Kaggle Spotify Dataset
Convert Kaggle dataset format to our required format for music recommendation system
"""

import pandas as pd
import json
from pathlib import Path

def process_kaggle_dataset():
    """Process Kaggle Spotify dataset and convert to our format"""
    
    print("🎵 Processing Kaggle Spotify Dataset...")
    
    # Load Kaggle dataset
    input_path = Path('data/spotify_raw.csv')
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        print("Please download dataset from Kaggle and place it at: data/spotify_raw.csv")
        return
    
    print(f"✓ Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"✓ Loaded {len(df):,} tracks")
    
    # Display columns to understand structure
    print(f"\nDataset columns: {list(df.columns)}")
    print(f"\nFirst row sample:")
    print(df.head(1).to_dict('records')[0])
    
    # Map Kaggle columns to our format
    # Common Kaggle dataset columns: track_id, artists, album_name, track_name, popularity, 
    # duration_ms, explicit, danceability, energy, key, loudness, mode, speechiness, 
    # acousticness, instrumentalness, liveness, valence, tempo, time_signature, track_genre
    
    print("\n🔄 Converting to our format...")
    
    # Create our format
    processed_data = []
    
    for idx, row in df.iterrows():
        try:
            # Map columns (adjust based on actual Kaggle dataset structure)
            track_data = {
                'track_id': str(row.get('track_id', row.get('id', f'track_{idx}'))),
                'name': str(row.get('track_name', row.get('name', 'Unknown'))),
                'artist': str(row.get('artists', row.get('artist', 'Unknown Artist'))),
                'album': str(row.get('album_name', row.get('album', 'Unknown Album'))),
                'popularity': int(row.get('popularity', 0)),
                'duration_ms': int(row.get('duration_ms', 0)),
                'explicit': bool(row.get('explicit', False)),
                'genres': [str(row.get('track_genre', 'pop'))] if pd.notna(row.get('track_genre')) else ['pop'],
                
                # Audio features
                'danceability': float(row.get('danceability', 0.5)) if pd.notna(row.get('danceability')) else 0.5,
                'energy': float(row.get('energy', 0.5)) if pd.notna(row.get('energy')) else 0.5,
                'key': int(row.get('key', 0)) if pd.notna(row.get('key')) else 0,
                'loudness': float(row.get('loudness', -10)) if pd.notna(row.get('loudness')) else -10,
                'mode': int(row.get('mode', 1)) if pd.notna(row.get('mode')) else 1,
                'speechiness': float(row.get('speechiness', 0.1)) if pd.notna(row.get('speechiness')) else 0.1,
                'acousticness': float(row.get('acousticness', 0.5)) if pd.notna(row.get('acousticness')) else 0.5,
                'instrumentalness': float(row.get('instrumentalness', 0)) if pd.notna(row.get('instrumentalness')) else 0,
                'liveness': float(row.get('liveness', 0.1)) if pd.notna(row.get('liveness')) else 0.1,
                'valence': float(row.get('valence', 0.5)) if pd.notna(row.get('valence')) else 0.5,
                'tempo': float(row.get('tempo', 120)) if pd.notna(row.get('tempo')) else 120,
                'time_signature': int(row.get('time_signature', 4)) if pd.notna(row.get('time_signature')) else 4,
            }
            
            processed_data.append(track_data)
            
            if (idx + 1) % 10000 == 0:
                print(f"  Processed {idx + 1:,} tracks...")
                
        except Exception as e:
            print(f"  ⚠️  Error processing row {idx}: {e}")
            continue
    
    print(f"\n✓ Successfully processed {len(processed_data):,} tracks")
    
    # Save to JSON (raw format)
    json_output = Path('data/music_data_raw.json')
    print(f"\n💾 Saving to {json_output}...")
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved raw JSON")
    
    # Save to CSV (for easier processing)
    csv_output = Path('data/music_data.csv')
    print(f"\n💾 Saving to {csv_output}...")
    df_output = pd.DataFrame(processed_data)
    
    # Convert genres list to string
    df_output['genres'] = df_output['genres'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    
    df_output.to_csv(csv_output, index=False, encoding='utf-8')
    print(f"✓ Saved CSV")
    
    # Display statistics
    print("\n" + "="*60)
    print("📊 DATASET STATISTICS")
    print("="*60)
    print(f"Total tracks: {len(df_output):,}")
    print(f"Unique artists: {df_output['artist'].nunique():,}")
    print(f"Unique genres: {df_output['genres'].nunique():,}")
    print(f"Popularity range: {df_output['popularity'].min()} - {df_output['popularity'].max()}")
    print(f"Duration range: {df_output['duration_ms'].min()/1000:.0f}s - {df_output['duration_ms'].max()/1000:.0f}s")
    print(f"\nTop 10 artists:")
    print(df_output['artist'].value_counts().head(10))
    print(f"\nTop 10 genres:")
    print(df_output['genres'].value_counts().head(10))
    print("="*60)
    
    print("\n✅ Dataset processing complete!")
    print(f"📁 Files created:")
    print(f"   - {json_output}")
    print(f"   - {csv_output}")
    print("\n🎯 Next steps:")
    print("   1. Run: python data_cleaning_music.py")
    print("   2. Run: python visualization_music.py")
    print("   3. Run: streamlit run app_music.py")

if __name__ == '__main__':
    process_kaggle_dataset()
