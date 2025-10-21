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

# ============================================
# Statistical Analysis (Correlation)
# ============================================

# Pearson Correlation (Linear)
pearson_corr, pearson_pval = stats.pearsonr(df_corr['Price'], df_corr['MetacriticScore'])

print(f"1. PEARSON CORRELATION (Price vs Metacritic Score):")

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
    color_discrete_sequence=["gold"],
    # Data to show when the mouse stays on it
    hover_data=['Name', 'Developers', 'Publishers']
)

fig2.update_layout(
    xaxis_title="Price",
    yaxis_title="Metacritic Score"
)

fig2.show()

# End of Question 1

# ==================================================================================================

# Question 2 : What kind of correlation exists between Price and RequiredAge?

# Ensure essential columns for ReviewRatio exist
df['TotalReviews'] = df['Positive'] + df['Negative']
# Calculate Review Ratio (Positive reviews / Total reviews), np.nan if false
df['ReviewRatio'] = np.where(df['TotalReviews'] > 0, df['Positive'] / df['TotalReviews'], np.nan)

# Filter out games with no age rating or insufficient reviews for relevance
df_filtered = df[(df['RequiredAge'] >= 0) & (df['TotalReviews'] > 100)].copy()

# ===================================================
# STATISTICAL ANALYSIS: Pearson Correlation
# ===================================================

pearson_corr_age, pearson_pval_age = stats.pearsonr(df_filtered['Price'], df_filtered['RequiredAge'])

print()
print(f"2. PEARSON CORRELATION (Price vs RequiredAge):")

print("\n" + "="*60)
print("QUICK INTERPRETATION")
print("="*60)
if abs(pearson_corr_age) < 0.1:
    strength = "very weak"
elif abs(pearson_corr_age) < 0.3:
    strength = "weak"
elif abs(pearson_corr_age) < 0.5:
    strength = "moderate"
else:
    strength = "strong"

print(f"Pearson's r ({pearson_corr_age:.3f}) indicates a {strength} linear correlation.")
print(f"Price explains only {(pearson_corr_age**2)*100:.2f}% of the variation in Required Age ratings.")
print("="*60)

# ===================================================
# DATA AGGREGATION FOR VISUALIZATION
# ===================================================

df_filtered['RequiredAge'] = df_filtered['RequiredAge'].astype('category')

# --- 1. Simple Grouping by Required Age ---
age_analysis = df_filtered.groupby('RequiredAge', observed=True).agg( # Agregate
    # Median Price (less sensitive to price outliers than mean)
    MedianPrice=('Price', 'median'),
    # Mean of Positive Review Ratio
    AverageReviewRatio=('ReviewRatio', 'mean'),
    GameCount=('AppID', 'count')
).reset_index() # Index begin with 0

# Filter out groups with too few games for robust analysis
min_games_per_age = 20
age_analysis = age_analysis[age_analysis['GameCount'] >= min_games_per_age]

# ===================================================
# VISUALIZATION (Twin Axes)
# ===================================================

fig, ax1 = plt.subplots(figsize=(10, 6))

# Axis 1: Median Price (Bars)
color_price = 'darkblue'
ax1.set_xlabel('Required Age Rating', fontsize=12)
ax1.set_ylabel('Median Price (USD)', color=color_price, fontsize=12)
ax1.bar(
    age_analysis['RequiredAge'].astype(str),
    age_analysis['MedianPrice'],
    color=color_price,
    alpha=0.6,
    label='Median Price'
)
ax1.tick_params(axis='y', labelcolor=color_price)
ax1.set_ylim(0, age_analysis['MedianPrice'].max() * 1.2)

# Axis 2: Average Positive Review Ratio (Line Plot)
ax2 = ax1.twinx()
# Creates a second Y-axis (ax2) that shares the same X-axis (ax1).
# This allows plotting two metrics (Price and Review Ratio) with different scales on the same chart.

color_ratio = 'darkred'
# Defines the color 'darkred' for the elements of this secondary axis,
# ensuring visual contrast with the primary axis (Price, which is 'darkblue').

ax2.set_ylabel('Average Positive Review Ratio', color=color_ratio, fontsize=12)
# Sets the label for the second Y-axis (on the right) to 'Average Positive Review Ratio'
# and applies the defined color ('darkred') and font size.

ax2.plot(
# Starts plotting a line on the secondary axis (ax2).
    age_analysis['RequiredAge'].astype(str),
    # Defines the data for the X-axis: the required age groups (converted to strings).
    age_analysis['AverageReviewRatio'],
    # Defines the data for the Y-axis: the average positive review ratio values.
    color=color_ratio,
    # Applies the 'darkred' color to the line.
    marker='o',
    # Adds a circle marker ('o') at each data point on the line.
    linestyle='-',
    # Sets the line style to solid ('-').
    linewidth=2,
    # Sets the thickness of the line to 2 pixels.
    label='Average Review Ratio'
    # Defines the label for this line, which will be used in the legend.
)

ax2.tick_params(axis='y', labelcolor=color_ratio)
# Applies the 'darkred' color to the tick labels (values) on the secondary Y-axis,
# visually linking the numbers to the data line.

ax2.set_ylim(0.5, 1.0)
# Sets the limits for the secondary Y-axis from 0.5 (50%) to 1.0 (100%),
# focusing on the most relevant range for positive review ratios.

fig.suptitle('Median Price and Quality by Required Age', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
plt.show()

# End of Question 2

# ==================================================================================================

# Question 3 : Correlation between Positive Reviews and Recommendations

# --- 1. Data Filtering for Correlation ---
cols_corr_recommend = ['Positive', 'Recommendations', 'Name']
df_corr_recommend = df[cols_corr_recommend].copy() # Keep 2 different dataframes

# Drop rows with missing values in either column to ensure valid correlation
# The parameter 'inplace=True' modifies the original DataFrame directly,
# without creating a new copy. If set to False (default), the method returns
# a modified copy instead, leaving the original DataFrame unchanged.
df_corr_recommend.dropna(subset=['Positive', 'Recommendations'], inplace=True)

# Keep only rows where both values are positive (avoid zeros or negatives that distort correlation)
df_corr_recommend = df_corr_recommend[
    (df_corr_recommend['Positive'] > 0) &
    (df_corr_recommend['Recommendations'] > 0)
]

# --- 2. Statistical Analysis (Pearson Correlation) ---
# Pearson requires at least 2 valid data points
if df_corr_recommend.shape[0] >= 2:

    # Compute Pearson correlation coefficient and p-value
    pearson_corr_recommend, pearson_pval_recommend = stats.pearsonr(
        df_corr_recommend['Positive'],
        df_corr_recommend['Recommendations']
    )

    # --- 3. Interpretation ---
    # Coefficient of determination (r^2), representing the % of explained variation
    r_squared = (pearson_corr_recommend ** 2) * 100

    # Define correlation strength based on absolute value of r
    correlation_strength = 'Very Weak'
    if abs(pearson_corr_recommend) >= 0.5:
        correlation_strength = 'Strong'
    elif abs(pearson_corr_recommend) >= 0.3:
        correlation_strength = 'Moderate'
    elif abs(pearson_corr_recommend) > 0.1:
        correlation_strength = 'Weak'

    # Determine direction (positive or negative correlation)
    direction = 'positive' if pearson_corr_recommend > 0 else 'negative'

    print()
    print(f"3. PEARSON CORRELATION (Positive Review vs Recommendations):")

    # Print formatted summary
    print("\n" + "=" * 60)
    print("QUICK INTERPRETATION (Positive Reviews vs. Recommendations)")
    print("=" * 60)
    print(f"1. PEARSON COEFFICIENT (r): {pearson_corr_recommend:.4f}")
    print(f"2. INTERPRETATION: {correlation_strength} {direction} linear correlation.")
    print(f"3. R-SQUARED: Positive Reviews explain {r_squared:.2f}% of the variation in Recommendations.")
    print("=" * 60)

    # --- 4. Visualization (Scatter Plot) ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot showing relationship between Positive Reviews and Recommendations
    ax.scatter(
        df_corr_recommend['Positive'],
        df_corr_recommend['Recommendations'],
        alpha=0.4,     # transparency for better readability
        s=10,          # marker size
        color='darkblue'
    )

    # Use logarithmic scales to handle large numeric ranges
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Add linear regression trendline (in red, dashed)

    sns.regplot(  # Create a regression plot (scatter + regression linear fit)
        x='Positive',  # Column on the x-axis represents number of positive reviews
        y='Recommendations',  # Column on the y-axis represents number of recommendations
        data=df_corr_recommend,  # DataFrame containing the columns to plot
        scatter=False,  # Disable scatter points (no need)
        color='red',  # Color of the regression line
        line_kws={  # Keyword arguments for line styling
            'linestyle': '--',  # Dashed line style for visual distinction
            'linewidth': 1.5,  # Slightly thicker line for better visibility
            'label': f"Linear Trend (r={pearson_corr_recommend:.2f})"  # Add label with Pearson's r value
        },
        ax=ax  # Plot on the existing matplotlib Axes object
    )

    # Configure plot titles and labels
    ax.set_title('Positive Reviews vs. Recommendations (Log Scale)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Number of Positive Reviews (Log Scale)', fontsize=12)
    ax.set_ylabel('Number of Recommendations (Log Scale)', fontsize=12)

    plt.grid(True, alpha=0.3)
    plt.show()


# End of Question 3

# ==================================================================================================

# Question 4 : Which genres are the most profitable (median price) and which have the best engagement (playtime)?

df_genres = df[['Genres', 'Price', 'AveragePlaytimeForever']].dropna()
df_genres['Genres'] = df_genres['Genres'].str.split(', ')
df_genres = df_genres.explode('Genres')

# aggregate by genre
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

# medin playtime by genre
plt.figure(figsize=(10,6))
top_playtime = genre_stats.sort_values('MedianPlaytime', ascending=False).head(15)
sns.barplot(y='Genres', x='MedianPlaytime', data=top_playtime, palette='magma')
plt.title('Top 15 Genres by Median Playtime')
plt.xlabel('Median Playtime (minutes)')
plt.ylabel('Genre')
plt.show()

print("""
Genres with higher median price tend to be strategy or simulation games.
Genres with longer playtime usually include RPGs and sandbox games.
This indicates engagement is not necessarily tied to higher prices.
""")

# End of Question 4

# ==================================================================================================

# Question 5 : How does time since release affect the number of recommendations/reviews?

# End of Question 5

# ==================================================================================================

# Question 6 : Does platform support (Windows/Mac/Linux) impact sales or ratings?

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
