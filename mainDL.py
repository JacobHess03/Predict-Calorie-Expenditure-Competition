import pandas as pd
import numpy as np
import tensorflow as tf # Importazione esplicita di tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from time import strftime
import matplotlib.pyplot as plt

# 1. Caricamento dati
# Assicurati che i file train.csv e test.csv siano nella stessa directory dello script
try:
    df_train = pd.read_csv('train.csv')
    df_test  = pd.read_csv('test.csv')
    print("File CSV caricati con successo.")
except FileNotFoundError:
    print("Errore: Assicurati che i file 'train.csv' e 'test.csv' si trovino nella stessa directory dello script.")
    exit() # Esci dallo script se i file non vengono trovati


# 2. Conserva ID
train_ids = df_train.pop('id')
test_ids  = df_test.pop('id')

# 3. Feature engineering
for df in (df_train, df_test):
    df['BMI']       = df['Weight'] / (df['Height'] ** 2)
    # Gestisci eventuali valori infiniti o NaN risultanti dalla divisione per zero o valori nulli
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['BMI'].fillna(df['BMI'].median(), inplace=True) # Sostituisci NaN con la mediana del BMI

    # Assicurati che l'intervallo delle bins sia corretto e gestisci i casi limite/valori nulli in Age
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 40, 65, 100],
                             labels=[0, 1, 2, 3], right=True, include_lowest=True)
    # Converti in int, NaN diventa None. Sostituisci i potenziali None (se Age era NaN)
    df['Age_Group'] = df['Age_Group'].astype(float).fillna(-1).astype(int) # Usa -1 o un altro valore per Age non nei range/NaN

# 4. Encoding categorico
le = LabelEncoder()
df_train['Sex'] = le.fit_transform(df_train['Sex'])
df_test ['Sex'] = le.transform(df_test ['Sex'])

print("Feature engineering e encoding completati.")

# 5. Split train/validation
y = df_train.pop('Calories')
X = df_train

# Assicurati che X non contenga colonne non numeriche residue, ecc.
# print(X.dtypes) # Debugging: Controlla i tipi di dato

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Dati suddivisi: X_train shape {X_train.shape}, X_val shape {X_val.shape}")

# 6. Normalization layer
# È meglio creare il layer di normalizzazione all'interno della pipeline del modello
# o usarlo direttamente come primo strato del modello Sequenziale/funzionale.
# Tuttavia, se vuoi adattarlo prima, il tuo metodo è valido, ma richiede attenzione.

# Metodo alternativo più robusto (integrare nel modello):
# input_shape = X_train.shape[1]
# input_ = keras.Input(shape=(input_shape,), name='input_features')
# normalization_layer = layers.Normalization()
# normalization_layer.adapt(X_train) # Adatta il layer direttamente ai dati di training
# normalized = normalization_layer(input_)
# ... resto del modello con 'normalized' come input

# Mantenendo il tuo approccio per ora:
normalization_layer = layers.Normalization()
# Il build esplicito potrebbe non essere necessario, adapt() lo fa se non è ancora costruito.
# Ma se lo mantieni, la forma (None, X_train.shape[1]) è corretta.
# normalization_layer.build(input_shape=(None, X_train.shape[1])) # Questa linea non è strettamente necessaria prima di adapt
normalization_layer.adapt(np.array(X_train)) # Adatta il layer ai dati. Meglio passare un numpy array.

print("Layer di normalizzazione adattato.")

# 7. Costruzione Wide & Deep network
# Definizione della input_shape dopo lo split e prima dell'uso
input_shape = X_train.shape[1] # <<< Questa linea è qui e definisce input_shape

input_ = keras.Input(shape=(input_shape,), name='input_features')
normalized = normalization_layer(input_) # Applica la normalizzazione

# Deep path
deep = layers.Dense(64, activation='relu')(normalized)
deep = layers.Dense(32, activation='relu')(deep)

# Wide path (skip connection) - Concatena l'input normalizzato originale con l'output del deep path
# Assicurati che le dimensioni siano compatibili per la concatenazione
# normalized ha forma (None, input_shape)
# deep ha forma (None, 32)
# Per concatenarli correttamente lungo l'ultima dimensione (features),
# la forma è ok se l'input_shape è diversa da 32.
concat = layers.Concatenate()([normalized, deep]) # Concatena lungo l'asse delle feature (axis=-1 è default)

# Strato di output per la regressione
output = layers.Dense(1, name='calories_output')(concat)

# Definizione del modello Keras funzionale
model = keras.Model(inputs=input_, outputs=output)

print("\nModello Wide & Deep costruito.")

# 8. Compilazione
model.compile(
    loss='mse', # Mean Squared Error è una loss comune per la regressione
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), # Utilizzo corretto dell'ottimizzatore
    metrics=[tf.keras.metrics.RootMeanSquaredError()] # Utilizzo corretto della metrica
)
model.summary()

# 9. Callbacks: EarlyStopping + TensorBoard
def get_run_logdir(root_logdir='my_logs'):
    # Crea la directory se non esiste
    log_dir = Path(root_logdir) / strftime('run_%Y_%m_%d_%H_%M_%S')
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir) # Restituisce la stringa del percorso

run_logdir     = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(log_dir=run_logdir)
earlystop_cb   = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

print(f"\nCallback configurati. Log TensorBoard in: {run_logdir}")

# 10. Training
print("\nInizio addestramento del modello...")
history = model.fit(
    X_train, y_train,
    epochs=40,
    validation_data=(X_val, y_val),
    callbacks=[tensorboard_cb, earlystop_cb]
)

print("\nAddestramento completato.")

# 11. Plot learning curves
print("Generazione grafico learning curves...")
pd.DataFrame(history.history).plot(
    figsize=(8,5),
    grid=True,
    xlabel='Epoch'
)
plt.title('Training and Validation Metrics')
plt.show()
print("Grafico visualizzato.")

# 12. Predizione su test per submission
print("Generazione submission file...")
# Assicurati che l'ordine delle colonne e i tipi di dato in df_test siano gli stessi di X_train
# prima di fare la predizione, altrimenti la normalizzazione e il modello potrebbero non funzionare correttamente.
# Puoi riordinare le colonne di df_test per matchare X_train:
df_test_processed = df_test[X_train.columns]

preds = model.predict(df_test_processed).flatten() # Predici e appiattisci l'output (forma (N, 1) -> (N,))
submission = pd.DataFrame({'id': test_ids, 'Calories': preds})
submission.to_csv('submission_dl.csv', index=False)

print('Submission DL creata: submission_dl.csv')