#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# Sélection des paires FX par Machine Learning — Groupe 487
# =============================================================================

# -- 0. Imports & configuration -----------------------------------------------

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

warnings.filterwarnings("ignore")

FX_RETURNS   = "./fx_returns.csv"   # chemin vers le fichier de returns.py
OUTPUT_DIR   = "./outputs_ml"
TRADING_DAYS = 252
FORWARD_H    = 30     # horizon du label (jours de trading)
SHARPE_SEUIL = 0.3    # seuil pour classer "Active"
N_ACTIVE     = 6      # nombre de paires à sélectionner

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Chargement des données ------------------------------------------------

returns_df = pd.read_csv(FX_RETURNS, sep=";", decimal=",", index_col=0, parse_dates=True)
returns_df = returns_df.dropna(how="all")

print(f"{len(returns_df.columns)} paires | {len(returns_df)} jours")
print(list(returns_df.columns))


# -- 2. Définition du label ---------------------------------------------------
# Pour chaque paire et chaque date t, on calcule le Sharpe des 30 prochains jours.
# Si ce Sharpe dépasse 0.3 la paire est Active (1) sinon Inactive (0).

def make_label(r):
    r_future = r.shift(-1)
    # shift(-1) : on regarde les rendements futurs sans utiliser le jour t lui-meme
    def sharpe(x):
        s = x.std()
        return (x.mean() / s * np.sqrt(TRADING_DAYS)) if s > 1e-8 else 0.0
    forward_sharpe = r_future.rolling(FORWARD_H).apply(sharpe, raw=True).shift(-(FORWARD_H - 1))
    return (forward_sharpe > SHARPE_SEUIL).astype(int).rename("label")


# -- 3. Définition des features -----------------------------------------------
# 5 indicateurs calculés uniquement sur le passé (aucune fuite vers le futur)

FEATURE_COLS = ["sharpe_30", "vol_20", "vol_ratio", "win_rate_20", "autocorr_lag1"]

def make_features(r):
    f = pd.DataFrame(index=r.index)

    mu_30  = r.rolling(30).mean() * TRADING_DAYS
    sig_30 = r.rolling(30).std()  * np.sqrt(TRADING_DAYS)
    f["sharpe_30"]     = (mu_30 / sig_30.replace(0, np.nan)).clip(-5, 5)

    vol_20 = r.rolling(20).std() * np.sqrt(TRADING_DAYS)
    vol_60 = r.rolling(60).std() * np.sqrt(TRADING_DAYS)
    f["vol_20"]        = vol_20
    f["vol_ratio"]     = (vol_20 / vol_60.replace(0, np.nan)).clip(0, 5)

    f["win_rate_20"]   = r.rolling(20).apply(lambda x: (x > 0).mean(), raw=True)
    f["autocorr_lag1"] = r.rolling(30).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 2 else 0.0, raw=True
    ).fillna(0.0).clip(-1, 1)

    return f


# -- 4. Construction du dataset -----------------------------------------------
# On empile les features et le label de chaque paire dans un seul dataframe trié par date.

chunks = []
for pair in returns_df.columns:
    r = returns_df[pair].dropna()
    merged = make_features(r).join(make_label(r), how="inner").dropna()
    merged.insert(0, "pair", pair)
    if len(merged) >= 100:
        chunks.append(merged)

dataset = pd.concat(chunks).sort_index()
print(f"\nDataset : {len(dataset)} observations x {len(FEATURE_COLS)} features")
print(f"Labels positifs : {dataset['label'].mean():.1%}\n")
print(dataset[FEATURE_COLS + ["label"]].describe().round(3))
print("\nApercu du dataset :")
print(dataset.head(10))

# Graphique : proportion de labels positifs par paire
label_by_pair = dataset.groupby("pair")["label"].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(13, 4))
colors = ["#4CAF50" if v > 0.5 else "#EF5350" for v in label_by_pair.values]
ax.bar(label_by_pair.index, label_by_pair.values, color=colors)
ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="50%")
ax.set_ylim(0, 1)
ax.set_ylabel("% de periodes Active")
ax.set_title("Proportion de labels positifs par paire\n(vert = active plus de 50% du temps)", fontweight="bold")
ax.set_xticklabels(label_by_pair.index, rotation=45, ha="right")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "label_by_pair.png"), dpi=150, bbox_inches="tight")
plt.show()

# Graphique : distribution des features
fig, axes = plt.subplots(1, 5, figsize=(16, 4))
for ax, col in zip(axes, FEATURE_COLS):
    dataset[col].dropna().plot.hist(bins=50, ax=ax, color="#1976D2", alpha=0.8)
    ax.set_title(col, fontweight="bold")
    ax.set_xlabel("")
    ax.grid(alpha=0.3)
fig.suptitle("Distribution des features sur toutes les paires et toutes les dates",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_distributions.png"), dpi=150, bbox_inches="tight")
plt.show()


# -- 5. Walk-forward validation -----------------------------------------------
# On découpe la période en 4 tranches chronologiques.
# Le modèle apprend sur le passé et on teste sur ce qui suit immédiatement.

X = dataset[FEATURE_COLS].values
y = dataset["label"].values

tscv = TimeSeriesSplit(n_splits=4)
results_cv = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=15,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X[train_idx], y[train_idx])
    proba = rf.predict_proba(X[test_idx])[:, 1]
    pred  = rf.predict(X[test_idx])

    d_start = dataset.index[test_idx[0]].strftime("%Y-%m")
    d_end   = dataset.index[test_idx[-1]].strftime("%Y-%m")

    results_cv.append({
        "Fold": fold + 1,
        "Periode test": f"{d_start} -> {d_end}",
        "AUC": round(roc_auc_score(y[test_idx], proba), 3),
        "F1":  round(f1_score(y[test_idx], pred, zero_division=0), 3),
        "Accuracy": round(accuracy_score(y[test_idx], pred), 3),
    })

cv_df = pd.DataFrame(results_cv).set_index("Fold")
print("\nResultats walk-forward validation :")
print(cv_df)
print(f"\nAUC moyenne : {cv_df['AUC'].mean():.3f} +/- {cv_df['AUC'].std():.3f}")
print(f"F1 moyenne  : {cv_df['F1'].mean():.3f} +/- {cv_df['F1'].std():.3f}")


# -- 6. Modèle final & feature importance -------------------------------------
# On réentraine sur tout le dataset pour le modèle de production.

model = RandomForestClassifier(
    n_estimators=300, max_depth=6, min_samples_leaf=15,
    class_weight="balanced", random_state=42, n_jobs=-1
)
model.fit(X, y)

importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(importance.index, importance.values, color="#1976D2")
ax.set_xlabel("Importance relative")
ax.set_title("Feature Importance (Random Forest)", fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
plt.show()


# -- 7. Scoring final ---------------------------------------------------------
# Pour chaque paire, on donne les indicateurs d'aujourd'hui au modèle.
# Il estime une probabilité d'être performante le mois prochain.
# On prend les N_ACTIVE meilleures.

rows = []
for pair in returns_df.columns:
    sub = dataset[dataset["pair"] == pair][FEATURE_COLS].dropna()
    if sub.empty:
        continue
    proba = model.predict_proba(sub.iloc[[-1]])[:, 1][0]
    rows.append({"Paire": pair, "P(Active)": round(proba, 3)})

scores = (pd.DataFrame(rows)
          .sort_values("P(Active)", ascending=False)
          .reset_index(drop=True))
scores["Rang"]   = scores.index + 1
scores["Statut"] = ["ACTIVE" if i < N_ACTIVE else "inactive" for i in range(len(scores))]

colors = ["#4CAF50" if s == "ACTIVE" else "#B0BEC5" for s in scores["Statut"]]
fig, ax = plt.subplots(figsize=(13, 4))
ax.bar(scores["Paire"], scores["P(Active)"], color=colors)
ax.set_ylim(0, 1)
ax.set_ylabel("P(Active)")
ax.set_title(f"Scoring ML - top {N_ACTIVE} paires selectionnees (vert)", fontweight="bold")
ax.set_xticklabels(scores["Paire"], rotation=45, ha="right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pair_scores.png"), dpi=150, bbox_inches="tight")
plt.show()


# -- 8. Export ----------------------------------------------------------------

scores.to_csv(os.path.join(OUTPUT_DIR, "pair_scores.csv"), index=False)
print(f"\nResultats sauvegardes dans {OUTPUT_DIR}/")
print("\nPaires selectionnees :")
print(scores[scores["Statut"] == "ACTIVE"][["Rang", "Paire", "P(Active)"]].to_string(index=False))
