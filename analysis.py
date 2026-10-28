# ============================================================
# Mobile Price Classification - Analisi Esplorativa (EDA)
# Dataset: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. CARICAMENTO DATI
# ------------------------------------------------------------
# Nota: su Windows usare r"..." (raw string) per i percorsi,
# cosi' i backslash non vengono interpretati come caratteri speciali.

df_train = pd.read_csv(r"S:\Desktop\KIMI+cartella\train.csv")
df_test = pd.read_csv(r"S:\Desktop\KIMI+cartella\test.csv")

print("Train:", df_train.shape)   # (2000, 21) -> include il target
print("Test :", df_test.shape)    # (1000, 21) -> senza target, con 'id'

# Verifica della differenza di colonne tra i due file
print("Nel train ma non nel test:", set(df_train.columns) - set(df_test.columns))
print("Nel test ma non nel train:", set(df_test.columns) - set(df_train.columns))

# ------------------------------------------------------------
# 2. PRIMO CONTATTO
# ------------------------------------------------------------
print(df_train.head())      # prime 5 righe
print(df_train.info())      # tipi di dato + valori mancanti (qui: zero!)
print(df_train.describe())  # statistiche delle colonne numeriche

# ------------------------------------------------------------
# 3. ESPLORAZIONE DEL TARGET
# ------------------------------------------------------------
# Quante istanze per fascia di prezzo? (risultato: 500 ciascuna, bilanciato)
print(df_train["price_range"].value_counts().sort_index())

# ------------------------------------------------------------
# 4. CORRELAZIONE CON IL TARGET
# ------------------------------------------------------------
# Quanto ogni caratteristica "si muove insieme" al prezzo (-1 ... +1)
corr = (
    df_train.corr()["price_range"]
    .drop("price_range")
    .sort_values(ascending=False)
)
print(corr.round(3))
# Risultato chiave: ram = 0.917 (dominante), battery_power = 0.201

# ------------------------------------------------------------
# 5. VISUALIZZAZIONE
# ------------------------------------------------------------
# Boxplot: distribuzione della RAM per fascia di prezzo
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.boxplot(data=df_train, x="price_range", y="ram", ax=ax,
            hue="price_range", palette="viridis", legend=False)
ax.set_title("RAM per fascia di prezzo", fontsize=14, fontweight="bold")
ax.set_xlabel("Fascia di prezzo (0 = economico, 3 = premium)")
ax.set_ylabel("RAM (MB)")
fig.savefig("ram_boxplot.png", bbox_inches="tight")

# ESERCIZIO: ripeti il boxplot per 'battery_power' e confronta le sovrapposizioni

