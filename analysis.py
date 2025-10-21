import json
import os
import kagglehub as kh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ==============================>
# 1. Data Download and Loading
# ==============================>

# Downloads dataset and gets path to games.json
path = kh.dataset_download("fronkongames/steam-games-dataset")
json_path = os.path.join(path, 'games.json')

with open(json_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# ==============================>
# 2. DataFrame Conversion (Simplified using list comprehension)
# ==============================>

def safe_int(val):
    """Converts value to int, defaulting to 0 if None or conversion fails."""
    try:
        return int(val) if val is not None else 0
    except ValueError:
        return 0

def safe_float(val):
    """Converts value to float, defaulting to 0.0 if None or conversion fails."""
    try:
        return float(val) if val is not None else 0.0
    except ValueError:
        return 0.0

games_list = []
for app_id, game in dataset.items():
    game_data = {
        'AppID': app_id,
        'Name': game.get('name', ''),
        'RequiredAge': safe_int(game.get('required_age', 0)),
        'ReleaseDate': game.get('release_date', ''),
        'Price': safe_float(game.get('price', 0)),
        'DlcCount': safe_int(game.get('dlc_count', 0)),
        'Recommendations': safe_int(game.get('recommendations', 0)),
        'Achievements': safe_int(game.get('achievements', 0)),
        'EstimatedOwners': game.get('estimated_owners', ''),
        'Positive': safe_int(game.get('positive', 0)),
        'Negative': safe_int(game.get('negative', 0)),
        'MetacriticScore': game.get('metacritic_score') if game.get('metacritic_score') not in [None, ""] else np.nan,
        'MedianPlaytimeForever': safe_int(game.get('median_playtime_forever', 0)),
        'AveragePlaytimeForever': safe_int(game.get('average_playtime_forever', 0)),
        'Genres': ', '.join(game.get('genres', [])),
        # Simplified handling for list fields, assuming they are lists or empty strings
        'Screenshots': len(game.get('screenshots', [])),
        'Movies': len(game.get('movies', []))
    }
    games_list.append(game_data)

df = pd.DataFrame(games_list)
print(f"📊 DataFrame: {df.shape[0]} lines and {df.shape[1]} columns")

# ==============================>
# 3. Save CSV
# ==============================>

df.to_csv("steam_games_simplified.csv", encoding="utf-8", index=False)

# ==================================================================================================
# Question 7: Do games with achievements have better engagement and player retention?
# ==================================================================================================

print("\n" + "="*80)
print("Question 7: Do games with achievements have better engagement and player retention?")
print("="*80)

# Filter games with and without achievements
with_ach = df[df['Achievements'] > 0].copy()
without_ach = df[df['Achievements'] == 0].copy()

# Calculate Review Ratio for both groups
with_ach['ReviewRatio'] = with_ach['Positive'] / (with_ach['Positive'] + with_ach['Negative'])
without_ach['ReviewRatio'] = without_ach['Positive'] / (without_ach['Positive'] + without_ach['Negative'])

# Removing games with no reviews
# Remove lines where review ratio is nan
with_ach = with_ach[with_ach['ReviewRatio'].notna()]
without_ach = without_ach[without_ach['ReviewRatio'].notna()]

# Remove lines where review ratio is infinite
with_ach = with_ach[~with_ach['ReviewRatio'].isin([float('inf'), float('-inf')])]
without_ach = without_ach[~without_ach['ReviewRatio'].isin([float('inf'), float('-inf')])]

# Filter only games with playtime > 0
with_ach_playtime = with_ach[with_ach['MedianPlaytimeForever'] > 0]
without_ach_playtime = without_ach[without_ach['MedianPlaytimeForever'] > 0]

# Calculate engagement metrics
metrics_with = {
    'Median Playtime': with_ach_playtime['MedianPlaytimeForever'].median(),
    'Avg Recommendations': with_ach['Recommendations'].mean(),
    'Avg Review Ratio': with_ach['ReviewRatio'].mean()
}

metrics_without = {
    'Median Playtime': without_ach_playtime['MedianPlaytimeForever'].median(),
    'Avg Recommendations': without_ach['Recommendations'].mean(),
    'Avg Review Ratio': without_ach['ReviewRatio'].mean()
}

print("\nGames WITH Achievements:")
for key, value in metrics_with.items():
    print(f"  {key}: {value:,.2f}")

print("\nGames WITHOUT Achievements:")
for key, value in metrics_without.items():
    print(f"  {key}: {value:,.2f}")

# ------ Bar Charts ------
categories = ['With Achievements', 'Without Achievements']
colors = ['#2ecc71', '#e74c3c']

plt.figure(figsize=(14, 10))

# 1. Median Playtime
plt.subplot(2, 2, 1)
playtime_values = [metrics_with['Median Playtime'], metrics_without['Median Playtime']]  # list with both playtime values
bars1 = plt.bar(categories, playtime_values, color=colors)  # creating a bar graph and saving the bars
plt.title('Median Playtime Comparison', fontsize=12, fontweight='bold')
plt.ylabel('Minutes', fontsize=10)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()  # Height for each bar in the graph

    # Writing the value above the bar
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:,.0f}', ha='center', va='bottom', fontsize=9)

# 2. Average Recommendations
plt.subplot(2, 2, 2)
rec_values = [metrics_with['Avg Recommendations'], metrics_without['Avg Recommendations']]  # list with both recommendation values
bars2 = plt.bar(categories, rec_values, color=colors)  # creating a bar graph and saving the bars
plt.title('Average Recommendations Comparison', fontsize=12, fontweight='bold')
plt.ylabel('Recommendations', fontsize=10)

# Add value labels on bars
for bar in bars2:
    height = bar.get_height()  # Height for each bar in the graph

    # Writing the value above the bar
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:,.0f}', ha='center', va='bottom', fontsize=9)

# 3. Average Review Ratio
plt.subplot(2, 2, 3)
ratio_values = [metrics_with['Avg Review Ratio'], metrics_without['Avg Review Ratio']]  # list with both review ratio values
bars3 = plt.bar(categories, ratio_values, color=colors)  # creating a bar graph and saving the bars
plt.title('Average Review Ratio Comparison', fontsize=12, fontweight='bold')
plt.ylabel('Ratio (0-1)', fontsize=10)
plt.ylim(0, 1)

# Add value labels on bars
for bar in bars3:
    height = bar.get_height()  # Height for each bar in the graph

    # Writing the value above the bar
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 4. Number of Games
plt.subplot(2, 2, 4)
count_values = [len(with_ach), len(without_ach)]  # list with both game count values
bars4 = plt.bar(categories, count_values, color=colors)  # creating a bar graph and saving the bars
plt.title('Number of Games in Each Category', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=10)

# Add value labels on bars
for bar in bars4:
    height = bar.get_height()  # Height for each bar in the graph

    # Writing the value above the bar
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', ha='center', va='bottom', fontsize=9)

# Final adjustments
plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig('question7_achievements_comparison.png', dpi=300, bbox_inches='tight')  # Save the figure with high resolution
plt.show()  # Display the plot

# ==================================================================================================
# Question 8: Is it worth investing in DLCs? Do they really increase game lifespan and revenue?
# ==================================================================================================

print("\n" + "="*80)
print("Question 8: Is it worth investing in DLCs? Do they really increase game lifespan and revenue?")
print("="*80)

# Convert owners to numeric
def owners_to_number(owners):
    if type(owners) != str or '-' not in owners:
        return np.nan

    parts = owners.replace(',', '').split('-')
    return (int(parts[0]) + int(parts[1])) / 2

# Segment games by DLC presence
with_dlc = df[df['DlcCount'] > 0].copy()
without_dlc = df[df['DlcCount'] == 0].copy()

# converting owners to a numeric value instead of a string
with_dlc['EstimatedOwnersNumeric'] = with_dlc['EstimatedOwners'].apply(owners_to_number)
without_dlc['EstimatedOwnersNumeric'] = without_dlc['EstimatedOwners'].apply(owners_to_number)

# Calculating Review Ratio
with_dlc['ReviewRatio'] = with_dlc['Positive'] / (with_dlc['Positive'] + with_dlc['Negative'])
without_dlc['ReviewRatio'] = without_dlc['Positive'] / (without_dlc['Positive'] + without_dlc['Negative'])

# Cleaning data
# Replacing infinite values with nan and then removing the lines without values in review ratio or owners
with_dlc = with_dlc.replace([np.inf, -np.inf], np.nan).dropna(subset=['ReviewRatio', 'EstimatedOwnersNumeric'])
without_dlc = without_dlc.replace([np.inf, -np.inf], np.nan).dropna(subset=['ReviewRatio', 'EstimatedOwnersNumeric'])

# Filter games with playtime > 0
with_dlc_playtime = with_dlc[with_dlc['MedianPlaytimeForever'] > 0]
without_dlc_playtime = without_dlc[without_dlc['MedianPlaytimeForever'] > 0]

# Calculate metrics
metrics_with_dlc = {
    'Median Playtime': with_dlc_playtime['MedianPlaytimeForever'].median(),
    'Avg Price': with_dlc['Price'].mean(),
    'Avg Estimated Owners': with_dlc['EstimatedOwnersNumeric'].mean(),
    'Avg Review Ratio': with_dlc['ReviewRatio'].mean()
}

metrics_without_dlc = {
    'Median Playtime': without_dlc_playtime['MedianPlaytimeForever'].median(),
    'Avg Price': without_dlc['Price'].mean(),
    'Avg Estimated Owners': without_dlc['EstimatedOwnersNumeric'].mean(),
    'Avg Review Ratio': without_dlc['ReviewRatio'].mean()
}

print("\nGames WITH DLCs:")
for key, value in metrics_with_dlc.items():
    print(f"  {key}: {value:,.2f}")

print("\nGames WITHOUT DLCs:")
for key, value in metrics_without_dlc.items():
    print(f"  {key}: {value:,.2f}")

# ------ Bar Charts ------
categories_dlc = ['With DLC', 'Without DLC']
colors_dlc = ['#3498db', '#e67e22']

plt.figure(figsize=(14, 10))

# 1. Median Playtime
plt.subplot(2, 2, 1)
playtime_values_dlc = [metrics_with_dlc['Median Playtime'], metrics_without_dlc['Median Playtime']]  # list with both playtime values
bars1 = plt.bar(categories_dlc, playtime_values_dlc, color=colors_dlc)  # creating a bar graph and saving the bars
plt.title('Median Playtime Comparison (DLC)', fontsize=12, fontweight='bold')
plt.ylabel('Minutes', fontsize=10)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()  # Height for each bar in the graph

    # Writing the value above the bar
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:,.0f}', ha='center', va='bottom', fontsize=9)

# 2. Average Price
plt.subplot(2, 2, 2)
price_values = [metrics_with_dlc['Avg Price'], metrics_without_dlc['Avg Price']]  # list with both price values
bars2 = plt.bar(categories_dlc, price_values, color=colors_dlc)  # creating a bar graph and saving the bars
plt.title('Average Price Comparison (DLC)', fontsize=12, fontweight='bold')
plt.ylabel('Price (USD)', fontsize=10)

# Add value labels on bars
for bar in bars2:
    height = bar.get_height()  # Height for each bar in the graph

    # Writing the value above the bar
    plt.text(bar.get_x() + bar.get_width()/2., height, f'${height:,.2f}', ha='center', va='bottom', fontsize=9)

# 3. Average Estimated Owners
plt.subplot(2, 2, 3)
owners_values = [metrics_with_dlc['Avg Estimated Owners'], metrics_without_dlc['Avg Estimated Owners']]  # list with both estimated owner values
bars3 = plt.bar(categories_dlc, owners_values, color=colors_dlc)  # creating a bar graph and saving the bars
plt.title('Average Estimated Owners Comparison (DLC)', fontsize=12, fontweight='bold')
plt.ylabel('Estimated Owners', fontsize=10)

# Add value labels on bars
for bar in bars3:
    height = bar.get_height()  # Height for each bar in the graph

    # Writing the value above the bar
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:,.0f}', ha='center', va='bottom', fontsize=9)

# 4. Average Review Ratio
plt.subplot(2, 2, 4)
ratio_values_dlc = [metrics_with_dlc['Avg Review Ratio'], metrics_without_dlc['Avg Review Ratio']]  # list with both review ratio values
bars4 = plt.bar(categories_dlc, ratio_values_dlc, color=colors_dlc)  # creating a bar graph and saving the bars
plt.title('Average Review Ratio Comparison (DLC)', fontsize=12, fontweight='bold')
plt.ylabel('Ratio (0-1)', fontsize=10)
plt.ylim(0, 1)

# Add value labels on bars
for bar in bars4:
    height = bar.get_height()  # Height for each bar in the graph

    # Writing the value above the bar
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Final adjustments
plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig('question8_dlc_comparison.png', dpi=300, bbox_inches='tight')  # Save the figure with high resolution
plt.show()  # Display the plot

# ==================================================================================================
# Question 9: What price range maximizes the relationship between player satisfaction and sales volume?
# Which price should I put in my indie game?
# ==================================================================================================

print("\n" + "=" * 80)
print("Question 9: Price range and relationship between satisfaction and sales volume")
print("=" * 80)

# Converting EstimatedOwners to numeric instead of string
df["EstimatedOwnersNumeric"] = df["EstimatedOwners"].apply(owners_to_number)

# Ensure necessary columns exist
df["ReviewRatio"] = df["Positive"] / (df["Positive"] + df["Negative"])
df["ReviewRatio"] = df["ReviewRatio"].replace([np.inf, -np.inf], np.nan)

# Creating price ranges
df["PriceRange"] = pd.cut(df["Price"], bins=[0, 5, 10, 20, 40, float("inf")], labels=["$0-5", "$5-10", "$10-20", "$20-40", "$40+"])

# Filter only valid data
valid_data = df.dropna(subset=["ReviewRatio", "EstimatedOwnersNumeric", "Recommendations", "PriceRange"])

# Calculate averages by price range
# Grouping by price range and aggregating the mean of ReviewRatio, owners, and Recommendations
stats = valid_data.groupby("PriceRange", observed=True).agg({
    "ReviewRatio": "mean",
    "EstimatedOwnersNumeric": "mean",
    "Recommendations": "mean"
}).reset_index() # Turn PriceRange back to a normal column

print("\nMetrics by price range:")
print(stats)

# --- Bar charts (4 subplots) ---
plt.figure(figsize=(12, 10))

# 1. Average Review Ratio
plt.subplot(2, 2, 1)
plt.bar(stats["PriceRange"], stats["ReviewRatio"], color="skyblue")
plt.title("Average Satisfaction (Review Ratio)")
plt.ylabel("Review Ratio (0-1)")

# 2. Average estimated owners
plt.subplot(2, 2, 2)
plt.bar(stats["PriceRange"], stats["EstimatedOwnersNumeric"], color="orange")
plt.title("Average Estimated Owners")
plt.ylabel("Average number of owners")

# 3. Average recommendations
plt.subplot(2, 2, 3)
plt.bar(stats["PriceRange"], stats["Recommendations"], color="green")
plt.title("Average Recommendations")
plt.ylabel("Average number of recommendations")

# 4. Normalized comparison of three metrics
stats_norm = stats.copy()
stats_norm["OwnersNorm"] = stats["EstimatedOwnersNumeric"] / stats["EstimatedOwnersNumeric"].max()
stats_norm["RecsNorm"] = stats["Recommendations"] / stats["Recommendations"].max()

plt.subplot(2, 2, 4)
plt.plot(stats["PriceRange"], stats["ReviewRatio"], marker="o", label="Satisfaction", color="blue")
plt.plot(stats["PriceRange"], stats_norm["OwnersNorm"], marker="s", label="Owners (norm.)", color="orange")
plt.plot(stats["PriceRange"], stats_norm["RecsNorm"], marker="^", label="Recommendations (norm.)", color="green")
plt.title("Normalized Comparison by Price Range")
plt.ylabel("Normalized Value (0-1)")
plt.legend()

plt.tight_layout()
plt.show()

# Determine the ideal range (sweet spot)
stats["BalanceScore"] = (
        stats["ReviewRatio"] * 0.4 +
        (stats["EstimatedOwnersNumeric"] / stats["EstimatedOwnersNumeric"].max()) * 0.3 +
        (stats["Recommendations"] / stats["Recommendations"].max()) * 0.3
)

best_range = stats.loc[stats["BalanceScore"].idxmax()]

# printing best stats to the game: ideal price , review ratio, avarage owners and recommendations
print(f"\nIdeal price range: {best_range['PriceRange']}")
print(f"  Review Ratio: {best_range['ReviewRatio']:.3f}")
print(f"  Average owners: {best_range['EstimatedOwnersNumeric']:,.0f}")
print(f"  Average recommendations: {best_range['Recommendations']:,.0f}")

# ==================================================================================================

# Question 10: Which supported languages generate greater global reach without compromising ratings?

# TODO: Step 1 - Parse SupportedLanguages column
# Extract individual languages from the text (may contain HTML tags)
# Focus on: English, Portuguese-Brazil, Spanish, Chinese, Russian, German, French, Japanese

# TODO: Step 2 - Create binary columns for key languages
# df['HasEnglish'], df['HasPortuguese'], df['HasSpanish'], etc.

# TODO: Step 3 - Calculate Review Ratio (if not already done in Question 9)

# TODO: Step 4 - Analyze PT-BR impact specifically
# Compare games with PT-BR vs without: EstimatedOwners, ReviewRatio, Recommendations

# TODO: Step 5 - Find top 5 languages correlated with high Recommendations
# For each language, calculate mean Recommendations and mean ReviewRatio

# TODO: Step 6 - Create heatmap
# Rows: Languages, Columns: Avg Review Ratio, Avg Estimated Owners (normalized)
# Color intensity shows performance

# TODO: Step 7 - Analyze language combinations
# Which combinations (e.g., EN + PT-BR + ES) yield best results?

# TODO: Step 8 - Print strategic insights about localization priorities for Brazilian developers

# End of Question 10

# ==================================================================================================