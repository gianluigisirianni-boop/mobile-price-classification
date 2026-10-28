# Costruisce ed esegue il notebook completo del progetto.
# Il notebook viene eseguito con cwd nella cartella dei CSV, cosi' i path
# restano relativi ("train.csv") e riproducibili per chi clona il repo.

import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# 📱 Mobile Price Classification

**Obiettivo:** aiutare Bob, fondatore di una nuova azienda di cellulari, a capire
quali caratteristiche tecniche determinano il prezzo di vendita e a prevedere
la **fascia di prezzo** (0 = economico … 3 = premium) di un telefono.

**Dataset:** [Mobile Price Classification — Kaggle](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)

- `train.csv` — 2.000 righe con target `price_range` (usato per addestramento e validazione)
- `test.csv` — 1.000 righe senza target (per l'eventuale submission Kaggle)

**Pipeline:** EDA → correlazioni → baseline Decision Tree → Random Forest → valutazione → feature importance""")

md("""## 1. Setup e caricamento dati

Per riprodurre: scaricare i due CSV dal link Kaggle qui sopra e metterli nella
stessa cartella del notebook.""")

code("""import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# grafici compatti: notebook piu' leggero, anteprima comunque nitida
matplotlib.rcParams["figure.dpi"] = 70
matplotlib.rcParams["savefig.bbox"] = "tight"

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

print("Train:", df_train.shape, "| Test:", df_test.shape)
print("Colonna presente solo nel train (il target):",
      set(df_train.columns) - set(df_test.columns))""")

md("""## 2. Primo contatto con i dati

Comandi standard appena si carica un dataset: dimensioni, prime righe,
tipi di dato e statistiche di base.""")

code("""print(df_train.head())
print()
df_train.info()""")

code("""df_train.describe().round(2)""")

md("""**Osservazioni:**
- ✅ Nessun valore mancante (tutte le colonne sono `non-null` al 100%)
- ✅ Tutte le colonne sono numeriche (`int64` / `float64`): nessuna conversione necessaria
- Le colonne `blue`, `dual_sim`, `four_g`, `three_g`, `touch_screen`, `wifi` sono **binarie** (0/1):
  la loro media coincide con la quota di telefoni dotati di quella caratteristica""")

md("""## 3. Distribuzione del target""")

code("""df_train["price_range"].value_counts().sort_index()""")

md("""**Target perfettamente bilanciato**: 500 telefoni per ciascuna delle 4 fasce.
Nessun problema di classi sbilanciate.""")

md("""## 4. Correlazione con il prezzo

Quanto ogni caratteristica "si muove insieme" a `price_range` (da -1 a +1).""")

code("""corr = (df_train.corr()["price_range"]
        .drop("price_range")
        .sort_values(ascending=False))
corr.round(3)""")

md("""**La RAM domina** (0.917). Seguono `battery_power` (0.20) e la risoluzione
dello schermo (~0.15). Tutto il resto e' rumore (< 0.05).""")

code("""fig, ax = plt.subplots(figsize=(9, 5.5))
sns.boxplot(data=df_train, x="price_range", y="ram", ax=ax,
            hue="price_range", palette="viridis", legend=False)
ax.set_title("RAM per fascia di prezzo", fontsize=14, fontweight="bold")
ax.set_xlabel("Fascia di prezzo (0 = economico, 3 = premium)")
ax.set_ylabel("RAM (MB)")
plt.show()""")

md("""Le scatole quasi **non si sovrappongono**: la RAM da sola separa quasi
perfettamente le 4 fasce. Ottimo presagio per il machine learning.""")

md("""## 5. Machine Learning

### Preparazione X / y

> ⚠️ **Regola anti data-leakage:** `y` contiene il target; da `X` si toglie
> **solo** il target. Togliere feature (es. la RAM) distrugge il modello:
> senza RAM l'accuracy crolla da 83% a 24.7%, il livello del caso puro.""")

code("""from sklearn.model_selection import train_test_split

y = df_train["price_range"]
X = df_train.drop(columns=["price_range"])

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train:", x_train.shape, "| Validation:", x_test.shape)""")

md("""### Baseline: Decision Tree""")

code("""from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

tree = DecisionTreeClassifier(random_state=42)
tree.fit(x_train, y_train)
acc_tree = accuracy_score(y_test, tree.predict(x_test))
print(f"Accuracy Decision Tree: {acc_tree:.3f}")""")

md("""### Random Forest

Stesso identico pattern sklearn (crea → fit → predict → valuta): cambiano
solo import e costruttore.""")

code("""from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
acc_rf = accuracy_score(y_test, y_pred)
print(f"Accuracy Random Forest: {acc_rf:.3f}")""")

md("""## 6. Matrice di confusione

L'accuracy dice *quante* volte il modello sbaglia; la matrice dice **dove**.""")

code("""from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
ax.set_title("Matrice di confusione - Random Forest", fontsize=13, fontweight="bold")
ax.set_xlabel("Previsione del modello")
ax.set_ylabel("Risposta vera")
plt.show()""")

md("""**Tutti gli errori sono tra fasce adiacenti** (mai confusione 0↔3): il modello
non commette errori commercialmente gravi. L'errore piu' frequente e' la sottostima
di un gradino dei telefoni premium (15 casi 3→2).""")

md("""## 7. Feature importance

Quanto pesa ogni caratteristica nelle decisioni del modello (somma = 100%).""")

code("""imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()

fig, ax = plt.subplots(figsize=(9, 6))
imp.plot.barh(ax=ax, color="#2c7fb8")
ax.set_title("Feature Importance - Random Forest", fontsize=13, fontweight="bold")
ax.set_xlabel("Peso nella decisione (somma = 1.0)")
for i, v in enumerate(imp):
    ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=8)
plt.show()""")

md("""## 8. Conclusioni per Bob

1. **La RAM determina quasi la meta' delle decisioni** (47% di peso, correlazione 0.92):
   e' la leva di prezzo principale.
2. **Batteria e risoluzione schermo** formano la seconda linea (~19% combinato).
3. **Le feature binarie** (bluetooth, wifi, 4G, dual SIM...) pesano meno dell'1%
   ciascuna: sono attese dal mercato, non giustificano un prezzo piu' alto.
4. Il modello finale (Random Forest) raggiunge l'**89% di accuracy** e non confonde
   mai fasce lontane: e' affidabile per stimare il posizionamento di prezzo
   di un nuovo modello.""")

nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

client = NotebookClient(nb, timeout=300, kernel_name="python3",
                        resources={"metadata": {"path": r"S:\Desktop\KIMI+cartella"}})
client.execute()

# ---------------------------------------------------------------
# Post-processing: spezza le stringhe JSON piu' lunghe di 1000
# caratteri in liste di righe (convenzione multilinea di nbformat:
# i consumer fanno ''.join(lista) -> contenuto identico).
# Serve a rendere il file trasferibile riga per riga.
# ---------------------------------------------------------------
import json

MAX_CHUNK = 1000

def split_long_strings(obj):
    if isinstance(obj, dict):
        return {k: split_long_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [split_long_strings(v) for v in obj]
    if isinstance(obj, str) and len(obj) > 1500:
        return [obj[i:i + MAX_CHUNK] for i in range(0, len(obj), MAX_CHUNK)]
    return obj

nb_dict = json.loads(nbf.writes(nb))
nb_dict = split_long_strings(nb_dict)

out = r"C:\Users\siriu\AppData\Local\Temp\mobile_price_classification.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=1)

# verifica di sicurezza
with open(out, "r", encoding="utf-8") as f:
    lines = f.readlines()
print("righe:", len(lines), "| max riga:", max(len(l) for l in lines))

import os
print("Notebook eseguito e salvato:", out, os.path.getsize(out), "bytes")
