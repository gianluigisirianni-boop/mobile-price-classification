import os
import re
import subprocess
import sys
from datetime import datetime


def genera_report():
    # Nome REALE del notebook del progetto
    notebook_name = "mobile_price_classification.ipynb"
    # Cartella di destinazione del report
    output_dir = "output"
    # Nome del file HTML che verra' creato
    output_name = os.path.join(output_dir, "report_mobile_classification.html")

    print(f"Lettura di {notebook_name} e generazione del report in corso...")

    if not os.path.exists(notebook_name):
        print(f"Errore: non trovo il file '{notebook_name}' in questa cartella!")
        return

    os.makedirs(output_dir, exist_ok=True)

    try:
        # sys.executable = lo stesso Python che sta eseguendo questo script:
        # nbconvert viene chiamato come modulo, senza dipendere dal comando
        # "jupyter" nel PATH (che su Windows spesso manca)
        comando = [
            sys.executable,
            "-m",
            "nbconvert",
            "--to",
            "html",
            "--execute",
            notebook_name,
            "--output",
            output_name,
        ]

        subprocess.run(comando, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Errore durante la conversione del notebook: {e}")
        print("Se manca nbconvert, installalo con:  python -m pip install nbconvert")
        return

    # --- Intestazione del report: titolo + data/ora di generazione ----------
    # datetime.now() viene valutato a OGNI esecuzione dello script,
    # quindi la data/ora nel report si aggiorna da sola a ogni run.
    adesso = datetime.now().strftime("%d/%m/%Y alle %H:%M:%S")
    intestazione = (
        '<div style="max-width: 960px; margin: 28px auto 4px auto; '
        'padding: 26px 34px; border-radius: 10px; '
        'background: linear-gradient(135deg, #173a5e 0%, #2c7fb8 100%); '
        "color: #ffffff; font-family: 'Segoe UI', Helvetica, Arial, sans-serif;\">"
        '<h1 style="margin: 0 0 8px 0; font-size: 1.75em; font-weight: 600; '
        'letter-spacing: 0.2px;">Analisi di Mercato Smartphone — Report Automatico</h1>'
        '<p style="margin: 0; font-size: 0.95em; opacity: 0.88;">'
        f"Generato automaticamente il {adesso}</p></div>"
    )

    with open(output_name, "r", encoding="utf-8") as f:
        html = f.read()

    # Inserisce l'intestazione subito dopo il tag <body ...>.
    # La lambda evita problemi con caratteri speciali nella sostituzione.
    # Il file HTML viene rigenerato da zero a ogni run, quindi non c'e'
    # rischio di intestazioni duplicate.
    html_nuovo, n = re.subn(
        r"<body[^>]*>", lambda m: m.group(0) + intestazione, html, count=1
    )
    if n == 0:  # fallback: nessun <body> trovato -> intestazione in cima al file
        html_nuovo = intestazione + html

    with open(output_name, "w", encoding="utf-8") as f:
        f.write(html_nuovo)

    print(f"Report generato con successo! File creato: {output_name}")
    print(f"Intestazione inserita con data/ora di generazione: {adesso}")


if __name__ == "__main__":
    genera_report()
