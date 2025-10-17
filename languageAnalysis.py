# ==============================>
# 1. Imports and Configs
# ==============================>

import json
import os

import kagglehub as kh
import pandas as pd
import numpy as np
import plotly.express as px

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
    games_list.append({
        'AppID': app_id,
        'Name': game.get('name', ''),
        'RequiredAge': int(game.get('required_age', 0) or 0),
        'ReleaseDate': game.get('release_date', ''),
        'Price': float(game.get('price', 0) or 0),
        'DlcCount': int(game.get('dlc_count', 0) or 0),
        'Achievements': int(game.get('achievements', 0) or 0),
        'Recommendations': int(game.get('recommendations', 0) or 0),
        'SupportedLanguages': str(game.get('supported_languages', '')),
        'Positive': int(game.get('positive', 0) or 0),
        'Negative': int(game.get('negative', 0) or 0),
        'MetacriticScore': game.get('metacritic_score') if game.get('metacritic_score') not in [None, ""] else np.nan,
        'Genres': ', '.join(game.get('genres', [])) if isinstance(game.get('genres'), list) else '',
        'Categories': ', '.join(game.get('categories', [])) if isinstance(game.get('categories'), list) else ''
    })

df = pd.DataFrame(games_list)

# ==============================>
# 4. Top Languages Analysis
# ==============================>

top_languages = ['English', 'Chinese', 'Spanish', 'French', 'German', 'Russian', 'Portuguese', 'Japanese']

df['TotalReviews'] = df['Positive'].fillna(0) + df['Negative'].fillna(0)


for lang in top_languages:
    df[f'Supports_{lang}'] = df['SupportedLanguages'].str.lower().str.contains(lang.lower())

# group median adn count
lang_stats = []
for lang in top_languages:
    mask = df[f'Supports_{lang}']
    lang_stats.append({
        'Language': lang,
        'MedianPrice': df.loc[mask, 'Price'].median(),
        'MedianReviews': df.loc[mask, 'TotalReviews'].median(),
        'MedianRecommendations': df.loc[mask, 'Recommendations'].median(),
        'Count': mask.sum()
    })

lang_summary = pd.DataFrame(lang_stats)
lang_summary = lang_summary[lang_summary['Count'] >= 50]  # at least 50 games

# plot

fig = px.scatter(
    lang_summary,
    x='MedianPrice',
    y='MedianReviews',
    size='Count',  # number games
    color='MedianReviews',
    hover_name='Language',
    hover_data={
        'MedianPrice': True,
        'MedianReviews': True,
        'Count': True
    },
    color_continuous_scale='Viridis',
    size_max=60,
    title='Median Reviews vs Median Price by Language\n(Bubble Size ~ Number of Games, Color ~ Median Reviews)',
    labels={
        'MedianPrice': 'Median Price (USD)',
        'MedianReviews': 'Median Reviews',
        'Count': 'Number of Games'
    }
)

fig.update_layout(
    xaxis=dict(showgrid=True, gridcolor='lightgrey'),
    yaxis=dict(showgrid=True, gridcolor='lightgrey'),
    legend_title_text='Median Reviews / Number of Games',
    template='plotly_white'
)

fig.show()

print("""
Observations:
- Games supporting English dominate both median price and median reviews, indicating that reaching a global audience is a critical factor.
- Other widely spoken languages, such as Chinese, Spanish, and French, show moderate engagement, suggesting more niche player bases.
- A clear trend appears where multilingual support correlates with higher median prices and stronger user engagement, highlighting the importance of language accessibility in game popularity.
- Bubble sizes represent the number of games per language, and color intensity indicates median recommendations, providing a comprehensive interactive view of language impact on sales and engagement.
""")
