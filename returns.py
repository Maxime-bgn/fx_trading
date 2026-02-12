import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

START_DATE = "2020-01-01"
TRADING_DAYS = 252

# =========================
# LECTURE + RETURNS
# =========================
returns = {}

for file in glob.glob("*=X.csv"):
    pair = file.replace("=X.csv", "")

    df = pd.read_csv(
        file,
        sep=";",
        decimal=",",
        index_col=0,
        parse_dates=True
    )

    if "Close" not in df.columns:
        continue

    prices = pd.to_numeric(df["Close"], errors="coerce").dropna()
    prices = prices[prices.index >= START_DATE]

    if len(prices) < 200:
        continue

    r = np.log(prices / prices.shift(1)).dropna()
    returns[pair] = r

returns_df = pd.DataFrame(returns)

# =========================
# SAUVEGARDE DES RENDEMENTS
# =========================
returns_df.to_csv("fx_returns.csv", sep=";", decimal=",")

print("[OK] fx_returns.csv sauvegardé")

# =========================
# SHARPE RATIOS
# =========================
sharpe = (returns_df.mean() / returns_df.std()) * np.sqrt(TRADING_DAYS)
sharpe = sharpe.sort_values(ascending=False)

print("\n=== RATIO DE SHARPE (2020–2025) ===")
print(sharpe.round(2))

# =========================
# PERFORMANCE CUMULÉE
# =========================
cum_perf = np.exp(returns_df.cumsum())

plt.figure(figsize=(12, 6))
for col in cum_perf.columns:
    plt.plot(cum_perf.index, cum_perf[col], label=col)

plt.title("Performance cumulée FX – Base 1 (2020–2025)")
plt.ylabel("Performance")
plt.legend(ncol=3, fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()


# BAR CHART SHARPE

plt.figure(figsize=(12, 6))
plt.bar(sharpe.index, sharpe.values)
plt.axhline(0, linewidth=1)
plt.title("Ratios de Sharpe FX (2020–2025)")
plt.ylabel("Sharpe Ratio")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
