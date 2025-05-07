#!/usr/bin/env python3
"""
Script per correggere il file di submission "submission_dl.csv" forzando i valori negativi a 0.

Uso:
    python fix_submission.py [input_csv] [output_csv]

Se non specificato, input_csv = 'submission_dl.csv' e output_csv = 'submission_dl_fixed.csv'.
"""
import sys
import pandas as pd
import numpy as np

def fix_negative_predictions(input_path: str, output_path: str) -> None:
    """
    Carica un CSV di submission, forza i valori negativi della colonna 'Calories' a zero,
    e salva il risultato in un nuovo CSV.
    """
    # Carica il file di submission
    df = pd.read_csv(input_path)

    # Controlla presenza della colonna 'Calories'
    if 'Calories' not in df.columns:
        raise KeyError(f"Colonna 'Calories' non trovata in {input_path}")

    # Forza i valori negativi a zero
    df['Calories'] = np.maximum(df['Calories'], 0)

    # Salva il CSV corretto
    df.to_csv(output_path, index=False)
    print(f"File corretto salvato in: {output_path}")

if __name__ == '__main__':
    # Gestione argomenti da linea di comando
    if len(sys.argv) >= 3:
        input_csv = sys.argv[1]
        output_csv = sys.argv[2]
    elif len(sys.argv) == 2:
        input_csv = sys.argv[1]
        output_csv = 'submission_dl_fixed.csv'
    else:
        input_csv = 'submission_dl.csv'
        output_csv = 'submission_dl_fixed.csv'

    try:
        fix_negative_predictions(input_csv, output_csv)
    except Exception as e:
        print(f"Errore durante l'elaborazione: {e}")
        sys.exit(1)