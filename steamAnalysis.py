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

# ==============================>

# 5. Questions

# ==============================>

# ==================================================================================================

# Question 1 : What kind of correlation exists between Price and MetaScore?


# More elegant style configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create figure with custom background
fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
ax.set_facecolor('#f8f9fa')

# Columns to show in the interactive plot
cols_to_keep = ['Price', 'MetacriticScore', 'Name', 'Developers', 'Publishers']

# Remove null values from Price and MetacriticScore
df_corr = df[cols_to_keep].dropna(subset=['Price', 'MetacriticScore'])
df_corr = df_corr[(df_corr['Price'] > 0) & (df_corr['MetacriticScore'] > 0)]

print("="*60)
print("ANALYSIS AND CORRELATION - Price vs Metacritic Score")
print("="*60)

# ============================================
# Statistical Analysis (Correlation)
# ============================================

# Pearson Correlation (Linear)
pearson_corr, pearson_pval = stats.pearsonr(df_corr['Price'], df_corr['MetacriticScore'])

print(f"1. PEARSON CORRELATION (Linear):")
print(f"    Coefficient (r): {pearson_corr:.4f} | P-value: {pearson_pval:.4e}")

print("\n" + "="*60)
print("QUICK INTERPRETATION")
print("="*60)
print(f"Pearson's r ({pearson_corr:.3f}) indicates a very weak linear correlation.")
print(f"Price explains only {(pearson_corr**2)*100:.2f}% of the variation in Metacritic scores.")
print("="*60)

# ============================================
# Visualization 1: Metacritic Score Distribution (Matplotlib)
# ============================================

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create figure with custom background
fig1, ax1 = plt.subplots(figsize=(10, 6), facecolor='white')
ax1.set_facecolor('#f8f9fa')

# Metacritic Score Distribution (Histogram - frequency of data)
ax1.hist(
    df_corr['MetacriticScore'],
    bins=30, # splits the distribution into 30 intervals
    color='#1f77b4',
    edgecolor='white',
    alpha=0.9 # sets the transparency level of the bars
)

# Add mean line as reference
mean_score = df_corr['MetacriticScore'].mean()
ax1.axvline(                # draws a vertical reference line at the mean score on the histogram
    mean_score,
    color='red',
    linestyle='dashed',     # sets the line style to dashed (---) instead of a solid line
    linewidth=2,
    label=f"Mean: {mean_score:.2f}"
)

ax1.set_xlabel('Metacritic Score', fontsize=12, fontweight='600')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='600')
ax1.set_title('Metacritic Score Distribution', fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, axis='y')
ax1.legend()

plt.tight_layout()
plt.show()

fig1.savefig("steam_games_distribution.png")

# ============================================
# Visualization 2: Price vs Metacritic Rating (Plotly)
# ============================================

fig2 = px.scatter(
    df_corr,
    x='Price',
    y='MetacriticScore',
    title='Price vs. Metacritic Rating',
    opacity=0.6,
    color_discrete_sequence=['coral'],
    # Data to show when the mouse stays on it
    hover_data=['Name', 'Developers', 'Publishers']
)

fig2.update_layout(
    xaxis_title="Price",
    yaxis_title="Metacritic Score"
)

fig2.show()

# Transform plot into png
fig2.write_image("steam_games_distribution per metacritic rating.png")

# End of Question 1

# ==================================================================================================

# Question 2 :

# End of Question 2

# ==================================================================================================

# Question 3 :

# End of Question 3

# ==================================================================================================

# Question 4 :

# End of Question 4

# ==================================================================================================

# Question 5 :

# End of Question 5

# ==================================================================================================

# Question 6 :

# End of Question 6

# ==================================================================================================

# Question 7 :

# End of Question 7

# ==================================================================================================

# Question 8 :

# End of Question 8

# ==================================================================================================

# Question 9 :

# End of Question 9

# ==================================================================================================

# Question 10 :

# End of Question 10

# ==================================================================================================
