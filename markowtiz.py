import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

TRADING_DAYS = 252
MAX_WEIGHT = 0.20

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
returns_df = returns_df.dropna()

mu = returns_df.mean() * TRADING_DAYS
Sigma = returns_df.cov() * TRADING_DAYS

n = len(mu)
w0 = np.ones(n) / n
bounds = [(0, MAX_WEIGHT)] * n
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

# =========================
# FONCTIONS PORTFOLIO
# =========================
def port_return(w): 
    return np.dot(w, mu)

def port_vol(w): 
    return np.sqrt(w.T @ Sigma @ w)

def neg_sharpe(w):
    return -port_return(w) / port_vol(w)

# =========================
# MINIMUM VARIANCE
# =========================
res_minvar = minimize(
    port_vol, w0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)
w_minvar = res_minvar.x

# =========================
# MAXIMUM SHARPE
# =========================
res_maxsharpe = minimize(
    neg_sharpe, w0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)
w_maxsharpe = res_maxsharpe.x

# =========================
# FRONTIÈRE EFFICIENTE
# =========================
target_returns = np.linspace(
    port_return(w_minvar),
    port_return(w_maxsharpe),
    30
)

frontier_vol = []
frontier_ret = []

for R in target_returns:
    cons = (
        constraints,
        {"type": "eq", "fun": lambda w, R=R: port_return(w) - R}
    )

    res = minimize(
        port_vol, w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons
    )

    if res.success:
        frontier_vol.append(port_vol(res.x))
        frontier_ret.append(port_return(res.x))

# =========================
# GRAPHE EXPLICATIF
# =========================
plt.figure(figsize=(10, 6))

plt.plot(frontier_vol, frontier_ret, "b--", label="Frontière efficiente")
plt.scatter(port_vol(w_minvar), port_return(w_minvar),
            c="green", s=80, label="Minimum Variance")
plt.scatter(port_vol(w_maxsharpe), port_return(w_maxsharpe),
            c="red", s=80, label="Maximum Sharpe")

plt.xlabel("Risque (Volatilité)")
plt.ylabel("Rendement attendu")
plt.title("Markowitz — Frontière efficiente FX (2020–2025)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
weights_df = pd.DataFrame({
    "Min_Variance": w_minvar,
    "Max_Sharpe": w_maxsharpe
}, index=returns_df.columns)

print("\n=== ALLOCATIONS PAR PAIRE ===")
print(weights_df.round(3))

weights_df.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Poids optimaux par paire FX")
plt.ylabel("Poids")
plt.axhline(0, linewidth=1)
plt.xticks(rotation=45, ha="right")
plt.grid(True)
plt.tight_layout()
plt.show()