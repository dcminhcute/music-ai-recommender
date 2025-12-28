"""
Generate dynamic track images based on audio features
Creates beautiful gradient images with mood-based colors
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import colorsys

def get_mood_colors(energy, valence, danceability):
    """Get gradient colors based on audio features"""
    
    # Base hue from valence (0=sad/purple, 1=happy/yellow)
    hue = 0.15 + (valence * 0.5)  # 0.15-0.65 range (purple to yellow)
    
    # Saturation from energy (higher energy = more saturated)
    saturation = 0.4 + (energy * 0.5)  # 0.4-0.9
    
    # Lightness from danceability
    lightness1 = 0.3 + (danceability * 0.2)  # Darker base
    lightness2 = 0.5 + (danceability * 0.3)  # Lighter top
    
    # Convert HSL to RGB
    rgb1 = tuple(int(x * 255) for x in colorsys.hls_to_rgb(hue, lightness1, saturation))
    rgb2 = tuple(int(x * 255) for x in colorsys.hls_to_rgb(hue, lightness2, saturation))
    
    return rgb1, rgb2

def get_genre_accent_color(genre):
    """Get accent color based on genre"""
    genre_colors = {
        'pop': (255, 105, 180),      # Hot pink
        'rock': (220, 20, 60),        # Crimson
        'hip-hop': (138, 43, 226),    # Blue violet
        'jazz': (184, 134, 11),       # Dark goldenrod
        'electronic': (0, 255, 255),  # Cyan
        'classical': (147, 112, 219), # Medium purple
        'country': (210, 180, 140),   # Tan
        'r-n-b': (199, 21, 133),     # Medium violet red
        'indie': (60, 179, 113),      # Medium sea green
        'metal': (105, 105, 105),     # Dim gray
    }
    
    genre_lower = str(genre).lower()
    for key, color in genre_colors.items():
        if key in genre_lower:
            return color
    
    return (255, 255, 255)  # Default white

def create_gradient(width, height, color1, color2):
    """Create vertical gradient"""
    base = Image.new('RGB', (width, height), color1)
    top = Image.new('RGB', (width, height), color2)
    
    mask = Image.new('L', (width, height))
    mask_data = []
    for y in range(height):
        mask_data.extend([int(255 * (y / height))] * width)
    mask.putdata(mask_data)
    
    base.paste(top, (0, 0), mask)
    return base

def add_text_with_shadow(draw, position, text, font, text_color, shadow_color):
    """Add text with shadow for better readability"""
    x, y = position
    # Shadow
    draw.text((x+2, y+2), text, font=font, fill=shadow_color)
    # Main text
    draw.text((x, y), text, font=font, fill=text_color)

def generate_track_image(track_name, artist, genres, energy, valence, danceability, popularity):
    """Generate a beautiful track image"""
    
    # Image dimensions
    width, height = 400, 400
    
    # Get colors
    color1, color2 = get_mood_colors(energy, valence, danceability)
    
    # Create gradient background
    img = create_gradient(width, height, color1, color2)
    draw = ImageDraw.Draw(img)
    
    # Add decorative elements based on energy
    if energy > 0.7:
        # High energy - add dynamic lines
        for i in range(5):
            y_pos = int(height * (0.2 + i * 0.15))
            draw.rectangle([0, y_pos, width, y_pos + 2], fill=(255, 255, 255, 100))
    
    # Add circular accent based on genre
    genre_color = get_genre_accent_color(genres)
    circle_size = int(100 + popularity)
    circle_alpha = 30
    circle_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    circle_draw = ImageDraw.Draw(circle_layer)
    circle_draw.ellipse(
        [width - circle_size, -50, width + 50, circle_size],
        fill=(*genre_color, circle_alpha)
    )
    img = Image.alpha_composite(img.convert('RGBA'), circle_layer).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Try to load font, fallback to default
    try:
        title_font = ImageFont.truetype("arial.ttf", 36)
        artist_font = ImageFont.truetype("arial.ttf", 24)
        genre_font = ImageFont.truetype("arial.ttf", 18)
    except:
        title_font = ImageFont.load_default()
        artist_font = ImageFont.load_default()
        genre_font = ImageFont.load_default()
    
    # Add text content
    padding = 20
    
    # Track name (wrap if too long)
    track_display = track_name if len(track_name) <= 25 else track_name[:22] + "..."
    add_text_with_shadow(
        draw, (padding, height - 120), track_display,
        title_font, (255, 255, 255), (0, 0, 0)
    )
    
    # Artist name
    artist_display = artist if len(artist) <= 30 else artist[:27] + "..."
    add_text_with_shadow(
        draw, (padding, height - 75), artist_display,
        artist_font, (230, 230, 230), (0, 0, 0)
    )
    
    # Genre badge
    genre_display = str(genres).split(',')[0].strip() if pd.notna(genres) else "Music"
    genre_display = genre_display[:15]
    add_text_with_shadow(
        draw, (padding, height - 40), f"♪ {genre_display}",
        genre_font, (200, 200, 200), (0, 0, 0)
    )
    
    # Add mood indicator in top-left
    mood_emoji = "😊" if valence > 0.6 else "😢" if valence < 0.4 else "😐"
    energy_bars = "█" * int(energy * 5)
    add_text_with_shadow(
        draw, (padding, padding), f"{mood_emoji} {energy_bars}",
        artist_font, (255, 255, 255), (0, 0, 0)
    )
    
    return img

def process_dataset_images(csv_path, output_dir, sample_size=100):
    """Generate images for a sample of tracks"""
    
    print(f"🎨 Generating track images...")
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sample tracks (prioritize popular ones)
    df_sorted = df.sort_values('popularity', ascending=False)
    sample = df_sorted.head(sample_size)
    
    print(f"Generating {len(sample)} images...")
    
    for idx, row in sample.iterrows():
        try:
            img = generate_track_image(
                track_name=row['name'],
                artist=row['artist'],
                genres=row['genres'],
                energy=row.get('energy', 0.5),
                valence=row.get('valence', 0.5),
                danceability=row.get('danceability', 0.5),
                popularity=row.get('popularity', 50)
            )
            
            # Save with track_id as filename
            track_id = row['track_id']
            img.save(output_path / f"{track_id}.png")
            
            if (idx + 1) % 10 == 0:
                print(f"  Generated {idx + 1}/{len(sample)} images...")
                
        except Exception as e:
            print(f"  ⚠️  Error generating image for {row['name']}: {e}")
            continue
    
    print(f"\n✅ Generated {len(list(output_path.glob('*.png')))} images")
    print(f"📁 Saved to: {output_path}")

if __name__ == '__main__':
    # Generate images for top 100 popular tracks
    process_dataset_images(
        csv_path='data/processed/music_data_final.csv',
        output_dir='data/track_images',
        sample_size=100
    )
