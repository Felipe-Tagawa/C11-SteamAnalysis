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

# ==============================
# Question 5 : How does time since release affect number of recommendations / reviews?
# ==============================

# Ensure TotalReviews exists
df['TotalReviews'] = df['Positive'] + df['Negative']

# Convert ReleaseDate to datetime
df['ReleaseDate'] = pd.to_datetime(df['ReleaseDate'], errors='coerce')

# Calculate days since release
df['DaysSinceRelease'] = (pd.Timestamp('today') - df['ReleaseDate']).dt.days

# Drop rows with missing values
df_time = df[['DaysSinceRelease', 'Recommendations', 'TotalReviews']].dropna()

# Create bins for days since release (intervals de 1 ano = 365 dias)
max_days = int(df_time['DaysSinceRelease'].max())
bins = list(range(0, max_days + 365, 365))

labels = [f"{i}-{i+1}y" for i in range(len(bins)-1)]
df_time['ReleaseBin'] = pd.cut(df_time['DaysSinceRelease'], bins=bins, labels=labels, right=False)

# Aggregate medians by bin
time_stats = df_time.groupby('ReleaseBin').agg(
    MedianRecommendations=('Recommendations', 'median'),
    MedianReviews=('TotalReviews', 'median'),
    Count=('DaysSinceRelease', 'count')
).reset_index()

# Only keep bins with enough games
time_stats = time_stats[time_stats['Count'] >= 20]

# ==============================
# Visualization: Median Recommendations and Reviews by Time Since Release
# ==============================

fig, ax1 = plt.subplots(figsize=(12,6))

# Bar plot for Median Recommendations
sns.barplot(x='ReleaseBin', y='MedianRecommendations', data=time_stats, color='skyblue', ax=ax1)
ax1.set_xlabel('Time Since Release (years)')
ax1.set_ylabel('Median Recommendations', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

# Line plot for Median Reviews
ax2 = ax1.twinx()
ax2.plot(time_stats['ReleaseBin'], time_stats['MedianReviews'], color='red', marker='o', linewidth=2, label='Median Reviews')
ax2.set_ylabel('Median Reviews', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Median Recommendations and Reviews by Time Since Release')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("""
Observation:
- Older games accumulate more recommendations and reviews.
- Very recent games have low numbers, independent of quality.
- Using medians per release-year bin smooths the extreme disparities and shows trend clearly.
""")
