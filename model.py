# ============================================================
# Mobile Price Classification - Modello ML
# Pipeline: preparazione dati -> split -> Random Forest -> valutazione
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------------------------------------------------
# 1. CARICAMENTO E PREPARAZIONE
# ------------------------------------------------------------
df_train = pd.read_csv(r"S:\Desktop\KIMI+cartella\train.csv")

# y = il target (risposte), X = le caratteristiche (domande)
# REGOLA: da X si toglie SOLO il target, mai le feature (data leakage!)
y = df_train["price_range"]
X = df_train.drop(columns=["price_range"])

# ------------------------------------------------------------
# 2. SPLIT TRAIN/VALIDATION (80% / 20%)
# ------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# 3. MODELLO: Random Forest
# ------------------------------------------------------------
# Pattern sklearn universale: crea -> fit -> predict -> valuta
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# ------------------------------------------------------------
# 4. VALUTAZIONE
# ------------------------------------------------------------
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy Random Forest: {acc:.3f}")
# Baseline di riferimento: Decision Tree = 0.833 | caso = 0.25

# Matrice di confusione: dove sbaglia il modello?
cm = confusion_matrix(y_test, y_pred)
print(cm)

fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
ax.set_title("Matrice di confusione - Random Forest", fontsize=13, fontweight="bold")
ax.set_xlabel("Previsione del modello")
ax.set_ylabel("Risposta vera")
fig.savefig("confusion_matrix.png", bbox_inches="tight")

# ------------------------------------------------------------
# 5. PREVISIONI SUL TEST SET DI KAGGLE (senza target)
# ------------------------------------------------------------
# df_test = pd.read_csv(r"S:\Desktop\KIMI+cartella\test.csv")
# predizioni = model.predict(df_test.drop(columns=["id"]))
# pd.DataFrame({"id": df_test["id"], "price_range": predizioni}).to_csv(
#     "submission.csv", index=False)

plt.show()
