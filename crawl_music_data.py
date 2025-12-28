import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json
import pandas as pd
from pathlib import Path
import time

# ========================================
# SPOTIFY API CREDENTIALS
# ========================================
# Lấy từ: https://developer.spotify.com/dashboard
CLIENT_ID = '70fc0d3a209b4a2ba8140fca0c48e409'
CLIENT_SECRET = 'cdf1b5e778ad407eb5df3070b534388b'

def initialize_spotify():
    """Khởi tạo Spotify API client"""
    try:
        client_credentials_manager = SpotifyClientCredentials(
            client_id=CLIENT_ID, 
            client_secret=CLIENT_SECRET
        )
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        print("✓ Spotify API initialized successfully!")
        return sp
    except Exception as e:
        print(f"❌ Error initializing Spotify API: {e}")
        print("\nHướng dẫn:")
        print("1. Truy cập: https://developer.spotify.com/dashboard")
        print("2. Tạo app mới")
        print("3. Copy Client ID và Client Secret")
        print("4. Paste vào file này (dòng 11-12)")
        return None

def crawl_playlist_tracks(sp, playlist_id, playlist_name=""):
    """Crawl tracks từ một playlist"""
    tracks_data = []
    
    try:
        results = sp.playlist_tracks(playlist_id)
        tracks = results['items']
        
        while results['next']:
            results = sp.next(results)
            tracks.extend(results['items'])
        
        print(f"  Found {len(tracks)} tracks in playlist: {playlist_name}")
        
        for item in tracks:
            if item['track'] is None:
                continue
                
            track = item['track']
            track_id = track['id']
            
            if track_id is None:
                continue
            
            # Lấy audio features
            try:
                audio_features = sp.audio_features([track_id])[0]
            except:
                audio_features = {}
            
            # Extract data
            track_data = {
                'track_id': track_id,
                'name': track['name'],
                'artist': ', '.join([artist['name'] for artist in track['artists']]),
                'artist_id': track['artists'][0]['id'] if track['artists'] else None,
                'album': track['album']['name'],
                'album_id': track['album']['id'],
                'release_date': track['album']['release_date'],
                'popularity': track['popularity'],
                'duration_ms': track['duration_ms'],
                'explicit': track['explicit'],
                'preview_url': track['preview_url'],
                'external_url': track['external_urls']['spotify'],
                'image_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
            }
            
            # Add audio features
            if audio_features:
                track_data.update({
                    'danceability': audio_features.get('danceability'),
                    'energy': audio_features.get('energy'),
                    'key': audio_features.get('key'),
                    'loudness': audio_features.get('loudness'),
                    'mode': audio_features.get('mode'),
                    'speechiness': audio_features.get('speechiness'),
                    'acousticness': audio_features.get('acousticness'),
                    'instrumentalness': audio_features.get('instrumentalness'),
                    'liveness': audio_features.get('liveness'),
                    'valence': audio_features.get('valence'),
                    'tempo': audio_features.get('tempo'),
                    'time_signature': audio_features.get('time_signature'),
                })
            
            tracks_data.append(track_data)
        
        return tracks_data
    
    except Exception as e:
        print(f"  Error crawling playlist {playlist_id}: {e}")
        return []

def get_artist_genres(sp, tracks_data):
    """Lấy genres cho mỗi track từ artist info"""
    print("\nFetching artist genres...")
    
    artist_ids = list(set([t['artist_id'] for t in tracks_data if t.get('artist_id')]))
    artist_genres = {}
    
    # Batch requests (50 artists per request)
    for i in range(0, len(artist_ids), 50):
        batch = artist_ids[i:i+50]
        try:
            artists = sp.artists(batch)['artists']
            for artist in artists:
                if artist:
                    artist_genres[artist['id']] = ', '.join(artist['genres'])
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            print(f"  Error fetching artist batch: {e}")
    
    # Add genres to tracks
    for track in tracks_data:
        artist_id = track.get('artist_id')
        track['genres'] = artist_genres.get(artist_id, '')
    
    print(f"✓ Added genres for {len(artist_genres)} artists")
    return tracks_data

def crawl_multiple_playlists(sp, playlist_ids):
    """Crawl nhiều playlists"""
    all_tracks = []
    
    for i, (playlist_id, playlist_name) in enumerate(playlist_ids, 1):
        print(f"\n[{i}/{len(playlist_ids)}] Crawling: {playlist_name}")
        tracks = crawl_playlist_tracks(sp, playlist_id, playlist_name)
        all_tracks.extend(tracks)
        time.sleep(0.5)  # Rate limiting
    
    return all_tracks

def save_data(tracks_data, output_dir='data'):
    """Lưu dữ liệu crawled"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save JSON
    json_file = output_path / 'music_data_raw.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(tracks_data, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Saved: {json_file}")
    
    # Save CSV
    df = pd.DataFrame(tracks_data)
    
    # Remove duplicates based on track_id
    df = df.drop_duplicates(subset=['track_id'])
    
    csv_file = output_path / 'music_data.csv'
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"✓ Saved: {csv_file}")
    print(f"✓ Total unique tracks: {len(df)}")
    
    return df

def search_and_crawl_tracks(sp, queries, limit_per_query=50):
    """Search tracks theo keywords thay vì playlists"""
    all_tracks = []
    
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Searching: {query}")
        try:
            results = sp.search(q=query, type='track', limit=limit_per_query, market='US')
            tracks = results['tracks']['items']
            print(f"  Found {len(tracks)} tracks")
            
            for track in tracks:
                if track is None or track['id'] is None:
                    continue
                
                track_id = track['id']
                
                # Lấy audio features (skip nếu lỗi 403 - app không có quyền)
                audio_features = {}
                try:
                    result = sp.audio_features([track_id])
                    if result and result[0]:
                        audio_features = result[0]
                except Exception as e:
                    # Skip audio features nếu lỗi (403 = không có quyền)
                    pass
                
                # Extract data
                track_data = {
                    'track_id': track_id,
                    'name': track['name'],
                    'artist': ', '.join([artist['name'] for artist in track['artists']]),
                    'artist_id': track['artists'][0]['id'] if track['artists'] else None,
                    'album': track['album']['name'],
                    'album_id': track['album']['id'],
                    'release_date': track['album']['release_date'],
                    'popularity': track['popularity'],
                    'duration_ms': track['duration_ms'],
                    'explicit': track['explicit'],
                    'preview_url': track['preview_url'],
                    'external_url': track['external_urls']['spotify'],
                    'image_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
                }
                
                # Add audio features
                if audio_features:
                    track_data.update({
                        'danceability': audio_features.get('danceability'),
                        'energy': audio_features.get('energy'),
                        'key': audio_features.get('key'),
                        'loudness': audio_features.get('loudness'),
                        'mode': audio_features.get('mode'),
                        'speechiness': audio_features.get('speechiness'),
                        'acousticness': audio_features.get('acousticness'),
                        'instrumentalness': audio_features.get('instrumentalness'),
                        'liveness': audio_features.get('liveness'),
                        'valence': audio_features.get('valence'),
                        'tempo': audio_features.get('tempo'),
                        'time_signature': audio_features.get('time_signature'),
                    })
                
                all_tracks.append(track_data)
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"  Error searching '{query}': {e}")
    
    return all_tracks

def main():
    print("="*60)
    print("SPOTIFY MUSIC DATA CRAWLER")
    print("="*60)
    
    # Initialize Spotify
    sp = initialize_spotify()
    if sp is None:
        return
    
    # ========================================
    # SEARCH QUERIES
    # ========================================
    search_queries = [
        # Genres
        'genre:pop', 'genre:rock', 'genre:hip-hop', 'genre:jazz', 'genre:electronic',
        'genre:country', 'genre:r-n-b', 'genre:indie', 'genre:classical', 'genre:metal',
        
        # Popular artists
        'Taylor Swift', 'Ed Sheeran', 'The Weeknd', 'Drake', 'Ariana Grande',
        'Billie Eilish', 'Post Malone', 'Dua Lipa', 'Bad Bunny', 'Harry Styles',
        
        # Decades
        'year:2024', 'year:2023', 'year:2022', 'year:2021', 'year:2020',
        'year:2015-2019', 'year:2010-2014', 'year:2000-2009', 'year:1990-1999',
        
        # Moods
        'happy energetic', 'sad acoustic', 'chill relax', 'workout', 'party',
    ]
    
    # Crawl data
    print(f"\nSearching {len(search_queries)} queries...")
    all_tracks = search_and_crawl_tracks(sp, search_queries, limit_per_query=50)
    
    if not all_tracks:
        print("\n❌ No data collected!")
        return
    
    # Get genres
    all_tracks = get_artist_genres(sp, all_tracks)
    
    # Save data
    df = save_data(all_tracks)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total tracks crawled: {len(all_tracks)}")
    print(f"Unique tracks: {len(df)}")
    print(f"Date range: {df['release_date'].min()} to {df['release_date'].max()}")
    print(f"Avg popularity: {df['popularity'].mean():.2f}")
    print("="*60)

if __name__ == "__main__":
    main()
