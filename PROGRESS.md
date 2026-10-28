# Progressi sessione — segnalibro per la ripresa

## Fatto (sessione 1 - EDA)

* Caricamento train/test con pd.read\_csv + raw string r"..."

* Verifica colonne con differenza tra insiemi (price\_range solo nel train)

* Primo contatto: shape, head, info, describe

* Target bilanciato: value\_counts() -> 500 per classe

* Correlazioni: ram = 0.917 (dominante), battery\_power = 0.201

* Boxplot RAM per fascia di prezzo

* README.md e analysis.py salvati e testati

## Fatto (sessione 2 - ML completo)

* X/y corretti: y = price\_range, X = tutto il resto (RAM dentro!)
  ERRORE IMPARATO: mai togliere feature dal X; mai lasciare il target dentro X (data leakage)
  Dimostrazione: senza RAM l'accuracy crolla da 83% a 24.7% (caso puro)

* Split: train\_test\_split(test\_size=0.2, random\_state=42) -> 1600/400

* Decision Tree baseline: 0.833 | Random Forest: 0.892

* Matrice di confusione: errori solo tra fasce adiacenti, mai 0<->3

* model.py salvato e testato, README aggiornato

## PROGETTO COMPLETO — pronto per GitLab

Possibili estensioni future (nuovi esercizi):

1. Decommentare sezione 5 di model.py e generare submission.csv per Kaggle
2. Tuning: provare n\_estimators=200 o altri modelli (XGBoost, SVM)
3. Feature importance della Random Forest (model.feature\_importances\_)
4. Nuovo dataset: ai\_writing\_detection\_dataset.csv (gia' in cartella)

