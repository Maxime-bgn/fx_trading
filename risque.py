import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

TRADING_DAYS = 252
TARGET = 0.0  # seuil downside

# =========================
# LOAD RETURNS
# =========================
returns_df = pd.read_csv(
    "fx_returns.csv",
    sep=";",
    decimal=",",
    index_col=0,
    parse_dates=True
)

returns_df = returns_df.dropna(axis=1, thresh=200)


# FONCTIONS DRAWDOWN

def max_drawdown_signed(r):
    cum = np.exp(r.cumsum())
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()   # négatif

def max_drawdown_positive(r):
    cum = np.exp(r.cumsum())
    peak = cum.cummax()
    dd = (peak - cum) / peak
    return dd.max()   # positif

def semi_deviation(r, target=0.0):
    downside = r[r < target]
    return downside.std() * np.sqrt(TRADING_DAYS)


#Var

def var_gaussian(r, alpha=0.95):
    mu = r.mean()
    sigma = r.std()
    z = norm.ppf(alpha)
    return -(mu - z * sigma)  # VaR positive (perte)

# CALCUL DES METRIQUES
stats = {}

for pair in returns_df.columns:
    r = returns_df[pair].dropna()
    if len(r) < 200:
        continue

    ret_ann = r.mean() * TRADING_DAYS
    vol_ann = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = ret_ann / vol_ann if vol_ann > 0 else np.nan
    var_95 = var_gaussian(r)

    semi = semi_deviation(r)
    sortino = ret_ann / semi if semi > 0 else np.nan

    mdd_signed = max_drawdown_signed(r)
    mdd_pos = max_drawdown_positive(r)

    calmar = ret_ann / mdd_pos if mdd_pos > 0 else np.nan

    stats[pair] = {
        "Return_ann": ret_ann,
        "Vol_ann": vol_ann,
        "Sharpe": sharpe,
        "Semi_dev": semi,
        "Sortino": sortino,
        "VaR_95_daily": var_95,
        "MaxDD_signed": mdd_signed,
        "MaxDD_positive": mdd_pos,
        "Calmar": calmar
    }

risk_df = pd.DataFrame(stats).T.sort_values("Sortino", ascending=False)

# OUTPUT
print("\n=== METRIQUES DE RISQUE FX ===")
print(risk_df.round(3))

risk_df.to_csv("fx_risk_metrics.csv", sep=";", decimal=",")
print("\n[OK] fx_risk_metrics.csv sauvegardé")
plt.figure(figsize=(14, 8))
plt.axis("off")

table = plt.table(
    cellText=risk_df.round(3).values,
    colLabels=risk_df.columns,
    rowLabels=risk_df.index,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.3)

plt.title("Métriques de risque FX (2020–2025)", pad=20)
plt.tight_layout()
plt.show()