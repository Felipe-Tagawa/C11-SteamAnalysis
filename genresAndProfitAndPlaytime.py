# ==============================>
# 1. Imports and Configs
# ==============================>

import json
import os

import kagglehub as kh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy import stats

# ==============================>
# 2. Download and Database Read
# ==============================>

path = kh.dataset_download("fronkongames/steam-games-dataset")
json_path = os.path.join(path, 'games.json')

with open(json_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# ==============================>
# 3. DataFrame Conversion
# ==============================>

games_list = []
for app_id, game in dataset.items():
    game_data = {
        'AppID': app_id,
        'Name': game.get('name', ''),
        'RequiredAge': int(game.get('required_age', 0) or 0),
        'ReleaseDate': game.get('release_date', ''),
        'Price': float(game.get('price', 0) or 0),
        'DlcCount': int(game.get('dlc_count', 0) or 0),
        'DetailedDescription': game.get('detailed_description', ''),
        'AboutTheGame': game.get('about_the_game', ''),
        'Reviews': game.get('reviews', ''),
        'HeaderImage': game.get('header_image', ''),
        'Website': game.get('website', ''),
        'Achievements': int(game.get('achievements', 0) or 0),
        'Recommendations': int(game.get('recommendations', 0) or 0),
        'Notes': game.get('notes', ''),
        'SupportedLanguages': game.get('supported_languages', ''),
        'Developers': game.get('developers', ''),
        'Publishers': game.get('publishers', ''),
        'EstimatedOwners': game.get('estimated_owners', ''),
        'Positive': int(game.get('positive', 0) or 0),
        'Negative': int(game.get('negative', 0) or 0),
        'MetacriticScore': game.get('metacritic_score') if game.get('metacritic_score') not in [None, ""] else np.nan,
        'PeakCCU': int(game.get('peak_ccu', 0) or 0),
        'AveragePlaytimeForever': int(game.get('average_playtime_forever', 0) or 0),
        'AveragePlaytime2Weeks': int(game.get('average_playtime_2weeks', 0) or 0),
        'MedianPlaytimeForever': int(game.get('median_playtime_forever', 0) or 0),
        'MedianPlaytime2Weeks': int(game.get('median_playtime_2weeks', 0) or 0),
        'Genres': ', '.join(game.get('genres', [])) if isinstance(game.get('genres'), list) else '',
        'Categories': ', '.join(game.get('categories', [])) if isinstance(game.get('categories'), list) else '',
        'Screenshots': len(game.get('screenshots', [])) if isinstance(game.get('screenshots'), list) else 0,
        'Movies': len(game.get('movies', [])) if isinstance(game.get('movies'), list) else 0
    }
    games_list.append(game_data)

df = pd.DataFrame(games_list)
print(f"📊 DataFrame: {df.shape[0]} lines and {df.shape[1]} columns")

# ==============================>
# 4. Save CSV
# ==============================>

df.to_csv("steam_games.csv", encoding="utf-8", index=False)



# Question 4 : Which genres are the most profitable (median price) and which have the best engagement (playtime)?

df_genres = df[['Genres', 'Price', 'AveragePlaytimeForever']].dropna()
df_genres = df_genres[df_genres['AveragePlaytimeForever'] > 0]  # <--- nova linha
df_genres['Genres'] = df_genres['Genres'].str.split(', ')
df_genres = df_genres.explode('Genres')

# by genre
genre_stats = df_genres.groupby('Genres').agg(
    MedianPrice=('Price', 'median'),
    MedianPlaytime=('AveragePlaytimeForever', 'median'),
    Count=('Genres', 'count')
).reset_index()

# only genres with enough games to be relevant
genre_stats = genre_stats[genre_stats['Count'] >= 20]

# median price by genre
plt.figure(figsize=(10,6))
top_price = genre_stats.sort_values('MedianPrice', ascending=False).head(15)
sns.barplot(y='Genres', x='MedianPrice', data=top_price, palette='viridis')
plt.title('Top 15 Genres by Median Price')
plt.xlabel('Median Price (USD)')
plt.ylabel('Genre')
plt.show()

# median playtime by genre
plt.figure(figsize=(10,6))
top_playtime = genre_stats.sort_values('MedianPlaytime', ascending=False).head(15)
sns.barplot(y='Genres', x='MedianPlaytime', data=top_playtime, palette='magma')
plt.title('Top 15 Genres by Median Playtime')
plt.xlabel('Median Playtime (minutes)')
plt.ylabel('Genre')
plt.show()

print("""
Observation:
- Genres with higher median price tend to be strategy or simulation games.
- Genres with longer playtime usually include RPGs and sandbox games.
- Engagement (playtime) is not necessarily tied to higher prices.
""")
