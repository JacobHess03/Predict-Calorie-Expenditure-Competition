# Predict-Calorie-Expenditure-Competition

## Descrizione
Questo repository contiene il codice sviluppato per la competizione Kaggle "Predict Calorie Expenditure". Vengono presentati due approcci distinti per affrontare il problema della regressione e predizione del dispendio calorico basandosi sui dati forniti:

1.  Un pipeline di Machine Learning tradizionale basata su **XGBoost** con un'ampia fase di feature engineering, preprocessing e ottimizzazione degli iperparametri tramite ricerca automatica. (`main.py`)
2.  Un modello di **Deep Learning** basato sull'architettura **Wide & Deep** implementato con TensorFlow/Keras, che include strati di normalizzazione, skip-connection e l'uso di callback per un addestramento efficiente su CPU/GPU. (`mainDL.py`)

L'obiettivo è confrontare le performance di un modello basato su alberi potenziati rispetto a un modello neurale su questo specifico task di regressione.

## Struttura del Repository

```text
├── data/
│   ├── train.csv             # Dati di addestramento (con target Calories)
│   └── test.csv              # Dati di test (senza target)
├── main.py                   # Script ML (XGBoost pipeline)
├── mainDL.py                 # Script Deep Learning (Keras Wide & Deep)
├── submission.csv            # Esempio di file di submission generato da main.py
├── submission_dl.csv         # Esempio di file di submission generato da mainDL.py
├── requirements.txt          # Elenco delle dipendenze Python
└── README.md                 # Questo file
```
Requisiti

    Python 3.8+
    Le librerie elencate in requirements.txt

Installazione

Si raccomanda vivamente di utilizzare un ambiente virtuale per installare le dipendenze:

# Crea un ambiente virtuale (se non ne hai già uno)
```
python -m venv venv
```
# Attiva l'ambiente virtuale
# Su macOS/Linux:
```
source venv/bin/activate
```
# Su Windows:
```
venv\Scripts\activate
```
# Installa le dipendenze
```
pip install -r requirements.txt
```
Il file requirements.txt contiene le seguenti dipendenze:

```
pandas>=1.3
numpy>=1.21
scikit-learn>=1.0
xgboost>=1.6
matplotlib>=3.4
tensorflow>=2.8
```
Assicurati che i file train.csv e test.csv siano posizionati nella directory data/ prima di eseguire gli script.
Script main.py – Pipeline XGBoost

Questo script implementa una pipeline completa di Machine Learning utilizzando il modello XGBoost.

Caratteristiche Principali:

    Caricamento Dati: Legge train.csv e test.csv, conserva id.
    
    Feature Engineering:
        Calcola BMI (Weight / Height^2).
        Crea Age_Group categorici (0-18, 19-40, 41-65, 66-100).
        
    Preprocessing:
        Encoding Sex con LabelEncoder.
        (Opzionale) Rimozione outlier (z-score).
        Trasformazioni numeriche: StandardScaler, PowerTransformer, PolynomialFeatures (grado 2, solo interazioni).
        (Opzionale) Log-transform del target Calories.
        
    Ottimizzazione Modello: Esegue RandomizedSearchCV (XGBoost, 15 iterazioni, CV=3) per la ricerca iperparametri.
    
    Addestramento Finale: Addestra il miglior modello su tutto il set di training.
    
    Predizione & Submission: Genera predizioni sul test set e crea submission.csv.

Esecuzione:
```
python main.py
```
Output Atteso:

    In console: parametri ottimali, metriche di validation (R², RMSE).
    File generato: submission.csv.

Script mainDL.py – Modello Wide & Deep in Keras

Questo script utilizza TensorFlow/Keras per costruire e addestrare un modello di Deep Learning con architettura Wide & Deep.

Caratteristiche Principali:

    Caricamento Dati: Legge train.csv e test.csv, conserva id.
    
    Feature Engineering:
        Calcola BMI.
        Crea Age_Group.
        
    Preprocessing:
        Encoding Sex.
        Split Train/Validation (80%/20%).
        Normalizzazione dei dati numerici con layers.Normalization() (adattato su training).
        
    Architettura Modello (Wide & Deep):
        Input normalizzato.
        Deep Path: Dense(64, ReLU) → Dense(32, ReLU).
        Skip connection: concatenazione (Input normalizzato + Deep Path output).
        Output: Dense(1).
        
    Compilazione: Loss mse, Optimizer Adam (lr=1e-3), Metric RootMeanSquaredError.
    
    Callback: EarlyStopping (patience=10, val_loss), TensorBoard.
    
    Addestramento: Addestra fino a 100 epoche con EarlyStopping e validation.
    
    Visualizzazione: Plotta curve di training/validation con Matplotlib.
    
    Predizione & Submission: Genera predizioni sul test set e crea submission_dl.csv.

Esecuzione:
```
python mainDL.py
```
Output Atteso:

    In console: model.summary(), progressi dell'addestramento per epoca (loss, RMSE).
    Grafico: Curve di training/validation.
    ![mainDL](https://github.com/user-attachments/assets/9cb1cb4c-287c-46b4-a073-1efc6154e29f)

    File generato: submission_dl.csv.
