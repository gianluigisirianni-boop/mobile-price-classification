# Mobile Price Classification 📱

Analisi esplorativa (EDA) del dataset Kaggle *Mobile Price Classification*:
classificare telefoni cellulari in 4 fasce di prezzo (0 = economico … 3 = premium)
a partire dalle caratteristiche tecniche.

**Contesto:** Bob ha fondato un'azienda di cellulari e vuole capire quali
caratteristiche determinano il prezzo di vendita, per posizionare i suoi prodotti
contro i grandi produttori.

## Dataset

- Fonte: [Mobile Price Classification — Kaggle](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)
- `train.csv` — 2.000 righe, 20 caratteristiche + target `price_range`
- `test.csv` — 1.000 righe, senza target (struttura standard da competizione ML)

## Analisi esplorativa — risultati chiave

- **Dati puliti:** zero valori mancanti, tutte le colonne numeriche (int64/float64).
- **Target bilanciato:** 500 telefoni per ciascuna delle 4 fasce di prezzo —
  nessun problema di classi sbilanciate.
- **Colonne binarie:** `blue`, `dual_sim`, `four_g`, `three_g`, `touch_screen`, `wifi`
  (valori 0/1; la media coincide con la quota di telefoni dotati della feature).

### Correlazione con `price_range`

| Caratteristica | Correlazione | Lettura |
|---|---|---|
| `ram` | **0.917** | Caratteristica dominante |
| `battery_power` | 0.201 | Contributo debole ma reale |
| `px_width` / `px_height` | ~0.15 | Contributo debole |
| tutte le altre | < 0.05 | Rumore, irrilevanti da sole |

> **Conclusione principale:** l'analisi esplorativa mostra che la RAM domina la
> determinazione del prezzo (corr = 0.92) — il boxplot RAM per fascia mostra
> classi quasi separate — mentre le altre caratteristiche contribuiscono
> marginalmente ma in modo combinato.

## Come riprodurre

```bash
python analysis.py
```

Lo script scarica/richiede `train.csv` e genera i grafici dell'esplorazione.

## Modello ML — risultati

| Modello | Accuracy (validation) |
|---|---|
| Tirare a caso (baseline) | 25% |
| Decision Tree | 83,3% |
| **Random Forest** | **89,2%** |

La matrice di confusione mostra che **tutti gli errori avvengono tra fasce
adiacenti** (mai confusione tra fascia 0 e fascia 3): il modello non commette
errori commercialmente gravi. L'errore più frequente e' la sottostima di un
gradino dei telefoni premium (15 casi 3→2).

## Prossimi passi

- [x] Modello di classificazione (ML): baseline Decision Tree + Random Forest
- [x] Valutazione: accuracy, matrice di confusione
- [ ] Previsioni su `test.csv` (sezione 5 di `model.py`, pronta da decommentare)
- [ ] Eventuale tuning iperparametri (es. `n_estimators`)

## Dashboard interattiva (Streamlit + SQLite)

`streamlit run dashboard.py` — esplorazione self-service dei 2000 modelli:
4 KPI (modelli, batteria, RAM, % 4G) + filtri a cascata
(fascia di prezzo → limiti RAM dinamici) su database SQLite (`data/dati.db`).

**Impatto (stima):** un'analisi ad-hoc ("RAM media della fascia X con RAM
minima Y?") passa da ~3-5 minuti di codice pandas scritto a mano — accessibile
solo a chi sa programmare — a ~10 secondi e 2 click, per qualsiasi utente.
