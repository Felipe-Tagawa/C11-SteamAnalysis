import kagglehub as kh
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc # biblioteca matemática

path = kh.dataset_download("fronkongames/steam-games-dataset")

print("Path to dataset files:", path)

json_path = os.path.join(path, 'games.json')

with open(json_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

games_list = []
for app_id, game in dataset.items():
    game_data = {
        'AppID': app_id,
        'Name': game.get('name', ''),
        'ReleaseDate': game.get('release_date', ''),
        'Price': float(game.get('price', 0) or 0),
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

print(f"DataFrame criado: {df.shape}")
print(df.head())

# salva corretamente o arquivo csv
df.to_csv("steam_games.csv", encoding="utf-8", index=False)
print(f"✅ Arquivo salvo: steam_games.csv ({df.shape[0]} jogos, {df.shape[1]} colunas)")
