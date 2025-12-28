"""
Enrich dataset with Spotify images and preview URLs
Adds album artwork and 30s audio preview to existing music data
"""

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from pathlib import Path
import time

# Spotify API Credentials
CLIENT_ID = '70fc0d3a209b4a2ba8140fca0c48e409'
CLIENT_SECRET = 'cdf1b5e778ad407eb5df3070b534388b'

def initialize_spotify():
    """Initialize Spotify API client"""
    try:
        auth_manager = SpotifyClientCredentials(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET
        )
        sp = spotipy.Spotify(auth_manager=auth_manager)
        print("✓ Spotify API initialized successfully!")
        return sp
    except Exception as e:
        print(f"❌ Failed to initialize Spotify API: {e}")
        return None

def enrich_track_data(sp, track_id):
    """Get image and preview URL for a track"""
    try:
        track = sp.track(track_id)
        
        # Get album image (largest available)
        image_url = None
        if track['album']['images']:
            image_url = track['album']['images'][0]['url']  # Largest image
        
        # Get preview URL (30s clip)
        preview_url = track.get('preview_url')
        
        # Get external Spotify URL
        external_url = track['external_urls']['spotify']
        
        return image_url, preview_url, external_url
    
    except Exception as e:
        return None, None, None

def enrich_dataset():
    """Enrich dataset with Spotify images and previews"""
    
    print("🎵 Enriching Music Dataset with Spotify Data...")
    print("="*60)
    
    # Load current dataset
    input_path = Path('data/processed/music_data_final.csv')
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return
    
    print(f"✓ Loading dataset from {input_path}...")
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    print(f"✓ Loaded {len(df):,} tracks")
    
    # Initialize Spotify API
    sp = initialize_spotify()
    if not sp:
        return
    
    # Add new columns if not exist
    if 'image_url' not in df.columns:
        df['image_url'] = None
    if 'preview_url' not in df.columns:
        df['preview_url'] = None
    if 'external_url' not in df.columns:
        df['external_url'] = None
    
    print("\n🔄 Fetching images and preview URLs from Spotify...")
    print("⏳ This will take some time (API rate limits)...\n")
    
    success_count = 0
    error_count = 0
    
    # Process tracks in batches to avoid rate limits
    batch_size = 50
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        for idx in batch.index:
            track_id = df.at[idx, 'track_id']
            
            # Skip if already enriched
            if pd.notna(df.at[idx, 'image_url']):
                continue
            
            try:
                image_url, preview_url, external_url = enrich_track_data(sp, track_id)
                
                if image_url:
                    df.at[idx, 'image_url'] = image_url
                    df.at[idx, 'preview_url'] = preview_url
                    df.at[idx, 'external_url'] = external_url
                    success_count += 1
                else:
                    error_count += 1
                
                # Progress update every 100 tracks
                if (success_count + error_count) % 100 == 0:
                    print(f"  Processed: {success_count + error_count:,} | Success: {success_count} | Failed: {error_count}")
            
            except Exception as e:
                error_count += 1
                if error_count % 100 == 0:
                    print(f"  ⚠️ Error at {idx}: {e}")
        
        # Rate limit protection - wait between batches
        if i + batch_size < len(df):
            time.sleep(2)  # 2 second pause between batches
        
        # Save progress every 500 tracks
        if (i + batch_size) % 500 == 0:
            print(f"\n💾 Saving progress... ({i + batch_size} tracks processed)")
            df.to_csv(input_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*60)
    print(f"✅ Enrichment complete!")
    print(f"   Total processed: {success_count + error_count:,}")
    print(f"   Successfully enriched: {success_count:,}")
    print(f"   Failed: {error_count:,}")
    print(f"   Success rate: {success_count/(success_count+error_count)*100:.1f}%")
    print("="*60)
    
    # Save final dataset
    print(f"\n💾 Saving enriched dataset to {input_path}...")
    df.to_csv(input_path, index=False, encoding='utf-8-sig')
    print("✓ Saved!")
    
    # Show statistics
    print("\n📊 Dataset Statistics:")
    print(f"   Tracks with images: {df['image_url'].notna().sum():,}")
    print(f"   Tracks with preview: {df['preview_url'].notna().sum():,}")
    print(f"   Tracks with Spotify link: {df['external_url'].notna().sum():,}")

if __name__ == '__main__':
    enrich_dataset()
