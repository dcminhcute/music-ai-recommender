import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import Counter

print("="*60)
print("MUSIC DATA VISUALIZATION")
print("="*60)

# Cấu hình matplotlib
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# Load dữ liệu đã clean
input_file = 'data/processed/music_data_final.csv'
df = pd.read_csv(input_file, encoding='utf-8-sig')
print(f"✓ Loaded {len(df)} tracks")

# Tạo folder visualizations
viz_dir = Path('visualizations')
viz_dir.mkdir(exist_ok=True)

# ========================================
# BIỂU ĐỒ 1: PHÂN BỐ POPULARITY
# ========================================
print("\n[1/6] Creating popularity distribution...")

plt.figure(figsize=(10, 6))
plt.hist(df['popularity'], bins=40, color='steelblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Track Popularity', fontsize=14, fontweight='bold')
plt.xlabel('Popularity Score (0-100)')
plt.ylabel('Number of Tracks')
plt.axvline(df['popularity'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {df["popularity"].mean():.1f}')
plt.axvline(df['popularity'].median(), color='green', linestyle='--', linewidth=2, 
           label=f'Median: {df["popularity"].median():.1f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
output_file = viz_dir / '01_popularity_distribution.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {output_file}")

# ========================================
# BIỂU ĐỒ 2: TOP 15 ARTISTS
# ========================================
print("[2/6] Creating top artists chart...")

artist_counts = df['artist'].value_counts().head(15)

plt.figure(figsize=(12, 8))
plt.barh(range(len(artist_counts)), artist_counts.values, color='coral', edgecolor='black')
plt.yticks(range(len(artist_counts)), artist_counts.index)
plt.xlabel('Number of Tracks', fontsize=12)
plt.title('Top 15 Artists by Track Count', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

for i, count in enumerate(artist_counts.values):
    plt.text(count + 0.5, i, f'{count}', va='center', fontsize=9, fontweight='bold')

plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
output_file = viz_dir / '02_top_artists.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {output_file}")

# ========================================
# BIỂU ĐỒ 3: TOP GENRES
# ========================================
print("[3/6] Creating genre distribution...")

all_genres = []
for genres_str in df['genres'].dropna():
    if genres_str != 'Unknown':
        genres_list = [g.strip() for g in str(genres_str).split(',')]
        all_genres.extend(genres_list)

genre_counts = Counter(all_genres)
top_genres = dict(genre_counts.most_common(15))

plt.figure(figsize=(12, 8))
plt.barh(range(len(top_genres)), list(top_genres.values()), color='skyblue', edgecolor='black')
plt.yticks(range(len(top_genres)), list(top_genres.keys()))
plt.xlabel('Number of Tracks', fontsize=12)
plt.title('Top 15 Music Genres', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

for i, count in enumerate(top_genres.values()):
    plt.text(count + 5, i, f'{count}', va='center', fontsize=10, fontweight='bold')

plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
output_file = viz_dir / '03_genre_distribution.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {output_file}")

# ========================================
# BIỂU ĐỒ 4: AUDIO FEATURES CORRELATION
# ========================================
print("[4/6] Creating audio features correlation heatmap...")

audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 
                  'tempo', 'popularity']

corr_df = df[audio_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Audio Features Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
output_file = viz_dir / '04_audio_features_correlation.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {output_file}")

# ========================================
# BIỂU ĐỒ 5: MOOD DISTRIBUTION
# ========================================
print("[5/6] Creating mood distribution...")

if 'mood' in df.columns:
    mood_counts = df['mood'].value_counts()
    
    plt.figure(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    plt.pie(mood_counts.values, labels=mood_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors, textprops={'fontsize': 11})
    plt.title('Track Distribution by Mood', fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    output_file = viz_dir / '05_mood_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_file}")

# ========================================
# BIỂU ĐỒ 6: ENERGY VS VALENCE SCATTER
# ========================================
print("[6/6] Creating energy vs valence scatter plot...")

plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['valence'], df['energy'], 
                     c=df['popularity'], cmap='viridis', 
                     alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, label='Popularity')
plt.xlabel('Valence (Happiness)', fontsize=12)
plt.ylabel('Energy', fontsize=12)
plt.title('Music Mood Map: Energy vs Valence', fontsize=14, fontweight='bold')

# Add quadrant labels
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
plt.text(0.75, 0.75, 'Happy\n& Energetic', ha='center', va='center', 
         fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
plt.text(0.25, 0.75, 'Sad\n& Energetic', ha='center', va='center', 
         fontsize=10, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
plt.text(0.75, 0.25, 'Happy\n& Calm', ha='center', va='center', 
         fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
plt.text(0.25, 0.25, 'Sad\n& Calm', ha='center', va='center', 
         fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.grid(True, alpha=0.3)
plt.tight_layout()
output_file = viz_dir / '06_energy_valence_map.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {output_file}")

# ========================================
# SUMMARY
# ========================================
print("\n" + "="*60)
print("VISUALIZATION COMPLETE")
print("="*60)
print(f"Created 6 visualizations in: {viz_dir}")
print(f"  1. Popularity Distribution")
print(f"  2. Top 15 Artists")
print(f"  3. Genre Distribution")
print(f"  4. Audio Features Correlation")
print(f"  5. Mood Distribution")
print(f"  6. Energy vs Valence Map")
print("="*60)
