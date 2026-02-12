import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

# =========================
# LECTURE DES CLOSE
# =========================
prices = {}

for file in glob.glob("*=X.csv"):
    pair = os.path.basename(file).replace("=X.csv", "")

    df = pd.read_csv(
        file,
        sep=";",
        decimal=",",
        index_col=0,
        parse_dates=True
    )

    if "Close" not in df.columns:
        continue

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    prices[pair] = df["Close"]

# DataFrame des prices
prices_df = pd.DataFrame(prices)

# Garder uniquement dates communes
prices_df = prices_df.dropna()

if prices_df.shape[1] < 2:
    raise ValueError("Pas assez de paires pour calculer une corrélation")

# =========================
# MATRICE DE CORRÉLATION
# =========================
corr = prices_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Matrice de corrélation des prix FX (Close)")
plt.tight_layout()
plt.show()
