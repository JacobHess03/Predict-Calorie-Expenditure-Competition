#  Importiamo le librerie necessarie
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import xgboost as xgb
from sklearn.datasets import load_diabetes

#  Creazione di un dataset sintetico con 4 feature e una variabile target
np.random.seed(42)
X = np.random.rand(100, 4) * 10  # 4 feature indipendenti
y = 3 * X[:, 0] + 2 * X[:, 1] - 1.5 * X[:, 2] + np.random.randn(100) * 2  # Relazione lineare con rumore

#  Suddivisione in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
#  Standardizzazione delle feature
# =========================
scaler = StandardScaler()  # Normalizza i dati con media=0 e deviazione standard=1
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
#  Rimozione della Multicollinearità con VIF
# =========================
# Calcoliamo il VIF per ogni feature
vif_data = pd.DataFrame()
vif_data["Feature"] = [f"X{i}" for i in range(X.shape[1])]
vif_data["VIF"] = [variance_inflation_factor(X_train_scaled, i) for i in range(X_train_scaled.shape[1])]

# Selezioniamo solo le feature con VIF < 10 (soglia accettabile per evitare multicollinearità)
selected_features = vif_data[vif_data["VIF"] < 10]["Feature"].index  # Manteniamo solo le feature con VIF accettabile
X_train_vif = X_train_scaled[:, selected_features]
X_test_vif = X_test_scaled[:, selected_features]

# =========================
#  Regolarizzazione (Ridge e Lasso)
# =========================
ridge = Ridge(alpha=1.0)  # Regolarizzazione L2, alpha=1 significa una leggera penalizzazione
ridge.fit(X_train_vif, y_train)
y_pred_ridge = ridge.predict(X_test_vif)
r2_ridge = r2_score(y_test, y_pred_ridge)  # Calcoliamo l'R^2

lasso = Lasso(alpha=0.1)  # Regolarizzazione L1, alpha più alto riduce più coefficienti a 0
lasso.fit(X_train_vif, y_train)
y_pred_lasso = lasso.predict(X_test_vif)
r2_lasso = r2_score(y_test, y_pred_lasso)  # Calcoliamo l'R^2

# =========================
#  Selezione delle Feature (`SelectKBest` e `RFE`)
# =========================
# SelectKBest: seleziona le 2 feature migliori in base alla statistica F
selector_kbest = SelectKBest(score_func=f_regression, k=2)
X_train_kbest = selector_kbest.fit_transform(X_train_vif, y_train)
X_test_kbest = selector_kbest.transform(X_test_vif)

# Recursive Feature Elimination (RFE): seleziona le 2 feature più importanti
model = LinearRegression()
selector_rfe = RFE(model, n_features_to_select=2)
X_train_rfe = selector_rfe.fit_transform(X_train_vif, y_train)
X_test_rfe = selector_rfe.transform(X_test_vif)

# =========================
#  Aggiunta di Termini Non Lineari (Regressione Polinomiale)
# =========================
poly = PolynomialFeatures(degree=2)  # Generiamo feature quadratiche
X_train_poly = poly.fit_transform(X_train_vif)
X_test_poly = poly.transform(X_test_vif)

# Creiamo un modello di regressione polinomiale
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)
r2_poly = r2_score(y_test, y_pred_poly)  # Calcoliamo l'R^2

# =========================
#  Risultati
# =========================
results = {
    "R^2 Ridge": r2_ridge,
    "R^2 Lasso": r2_lasso,
    "R^2 Polinomiale": r2_poly,
}
results



def elimina_variabili_vif_pvalue(X_train, y_train, vif_threshold=10.0, pvalue_threshold=0.05):
    """
    Rimuove variabili da X_train basandosi su VIF e p-value.
    
    - Elimina solo variabili con VIF > soglia e p-value > soglia.
    - Ricalcola VIF e p-value dopo ogni eliminazione.
    """
    
    # Copia dei dati per lavorare in sicurezza
    X_current = X_train.copy()
    
    # Aggiungi costante per statsmodels
    X_const = sm.add_constant(X_current)
    
    while True:
        # Modello OLS per calcolare p-value
        model = sm.OLS(y_train, X_const).fit()
        pvalues = model.pvalues.drop('const')  # escludi l'intercetta
        
        # Calcolo VIF
        vif = pd.DataFrame()
        vif["Feature"] = X_current.columns
        vif["VIF"] = [variance_inflation_factor(X_current.values, i) for i in range(X_current.shape[1])]
        
        # Unisco p-value e VIF
        stats = vif.copy()
        stats["p-value"] = pvalues.values
        
        # Trova candidati da eliminare: VIF alto + p-value alto
        candidates = stats[(stats["VIF"] > vif_threshold) & (stats["p-value"] > pvalue_threshold)]
        
        if candidates.empty:
            print("\nNessuna variabile da eliminare. Selezione completata.")
            break
        
        # Elimina la variabile con il VIF piÃ¹ alto tra i candidati
        worst_feature = candidates.sort_values(by="VIF", ascending=False)["Feature"].iloc[0]
        print(f"Rimuovo '{worst_feature}' con VIF = {candidates.loc[candidates['Feature'] == worst_feature, 'VIF'].values[0]:.2f} "
              f"e p-value = {candidates.loc[candidates['Feature'] == worst_feature, 'p-value'].values[0]:.4f}")
        
        # Aggiorna i dati
        X_current = X_current.drop(columns=[worst_feature])
        X_const = sm.add_constant(X_current)
    
    print("\nFeature finali selezionate:")
    print(X_current.columns.tolist())
    
    return X_current

def elimina_vif(X):
    
    # =============================================================================
    # Rimozione Iterativa della Multicollinearità con VIF
    # =============================================================================
    # Creiamo una copia di X su cui lavoreremo per rimuovere le feature
    X_vif_filtered = X.copy()

    # Definiamo la soglia VIF
    vif_threshold = 10 # Una soglia comune, puoi sperimentare con 5 o 10

    print(f"\n========================= Rimozione Iterativa Feature con VIF > {vif_threshold} =========================")

    # Inizializziamo variabili per il ciclo
    max_vif = float('inf') # Partiamo con un valore alto per entrare nel ciclo
    iteration = 0

    
    
    # Eseguiamo il ciclo finché il VIF massimo è sopra la soglia E ci sono più di 1 feature
    while max_vif > vif_threshold and X_vif_filtered.shape[1] > 1:
        iteration += 1
        print(f"\n--- Iterazione {iteration} ---")
        print(f"Features rimanenti: {X_vif_filtered.shape[1]}")

        # Calcola VIF per l'attuale set di feature
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_vif_filtered.columns

        # Converti il DataFrame corrente in un array numpy per variance_inflation_factor
        X_np = X_vif_filtered.values

        vif_list = []
        # Itera sugli indici delle colonne dell'array numpy
        for i in range(X_np.shape[1]):
            # Gestisce colonne con un solo valore unico (la varianza è zero, VIF infinito)
            if np.unique(X_np[:, i]).size <= 1:
                vif = float('inf') # Assegna infinito
            else:
                try:
                    # Calcola il VIF per la colonna i rispetto a tutte le altre colonne in X_np
                    vif = variance_inflation_factor(X_np, i)
                except Exception as e:
                    # Cattura altri possibili errori nel calcolo VIF
                    print(f"Errore nel calcolo VIF per colonna '{X_vif_filtered.columns[i]}' (indice {i}): {e}")
                    vif = np.nan # Assegna NaN in caso di errore

            vif_list.append(vif)

        vif_data["VIF"] = vif_list

        # Rimuovi righe con VIF NaN prima di trovare il massimo, per sicurezza
        vif_data = vif_data.dropna(subset=["VIF"])

        # Trova la feature con il VIF più alto nell'attuale set di feature
        if not vif_data.empty:
            # Ordina e prendi la prima riga (quella con il VIF massimo)
            max_vif_row = vif_data.sort_values(by="VIF", ascending=False).iloc[0]
            max_vif = max_vif_row["VIF"]
            feature_to_remove = max_vif_row["Feature"]

            # Se il VIF massimo è sopra la soglia, rimuovi quella feature
            if max_vif > vif_threshold:
                print(f"  - VIF Max: {max_vif:.2f} (Feature: '{feature_to_remove}'). Rimuovo...")
                # Rimuovi la colonna dal DataFrame X_vif_filtered
                X_vif_filtered = X_vif_filtered.drop(columns=[feature_to_remove])
            else:
                # Se il VIF massimo è sotto la soglia, esci dal ciclo
                print(f"  - VIF Max ({max_vif:.2f}) è sotto la soglia ({vif_threshold}). Processo terminato.")

        else: # Questo caso si verifica se rimangono 0 o 1 feature (gestito dalla condizione del while) o se tutti i VIF sono NaN
            print("  - Nessun dato VIF valido da analizzare o solo una feature rimasta. Processo terminato.")
            max_vif = 0 # Imposta max_vif a 0 per uscire dal ciclo

    # Una volta terminato il ciclo, X_vif_filtered contiene solo le feature con VIF <= threshold
    print("\n========================= Fine Rimozione Iterativa VIF =========================")
    print(f"Processo di rimozione VIF terminato.")
    print(f"Features finali selezionate ({X_vif_filtered.shape[1]}):")
    print(X_vif_filtered.columns.tolist())

    # Ora, aggiorniamo X con le features selezionate per le fasi successive
    X_selected = X_vif_filtered
    return X_selected




# import numpy as np
# import xgboost as xgb
# from sklearn.datasets import load_diabetes

# # 1) Carica il dataset
# diabetes = load_diabetes()
# X, y = diabetes.data, diabetes.target

# # 2) Crea il DMatrix
# dtrain = xgb.DMatrix(X, label=y, missing=np.nan)

# # 3) Parametri fissi
# params = {
#     'objective': 'reg:squarederror',
#     'eta': 0.1,
#     'max_depth': 4,
#     # 'alpha' e 'lambda' (reg_alpha/reg_lambda) verranno sovrascritti nel loop
# }

# # 4) Griglia per reg_alpha e reg_lambda
# param_grid = {
#     'alpha':    [0, 0.1, 0.5, 1],
#     'reg_lambda':[0, 0.1, 0.5, 1]
# }

# best_params = {}
# best_rmse = float("inf")

# # 5) Grid Search manuale
# for alpha in param_grid['alpha']:
#     for reg_lambda in param_grid['reg_lambda']:
#         # imposta i parametri correnti
#         params['alpha']      = alpha
#         params['lambda']     = reg_lambda   # in XGBoost 'lambda' mappa a reg_lambda
#         # oppure esplicitamente:
#         # params['reg_alpha']   = alpha
#         # params['reg_lambda']  = reg_lambda

#         cv_results = xgb.cv(
#             params,
#             dtrain,
#             num_boost_round=100,
#             nfold=5,
#             metrics="rmse",
#             early_stopping_rounds=10,
#             seed=42,
#             verbose_eval=False
#         )

#         mean_rmse = cv_results['test-rmse-mean'].min()
#         print(f"alpha={alpha}, reg_lambda={reg_lambda} → RMSE: {mean_rmse:.4f}")

#         if mean_rmse < best_rmse:
#             best_rmse   = mean_rmse
#             best_params = {'alpha': alpha, 'reg_lambda': reg_lambda}

# # 6) Risultati
# print(f"\nMigliori parametri: {best_params} con RMSE: {best_rmse:.4f}")




import seaborn as sns
import matplotlib.pyplot as plt
# Dataframe dell'errore sul target
df_error = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred_poly
})
# Calcola l'errore assoluto in log per ogni esempio
df_error["abs_log_error"] = df_error["log_error"].abs()

# Raggruppa per categoria 'Sex' e calcola errore medio
category = "Sex"  # Puoi cambiarlo in qualunque colonna categorica
grouped = df_error.groupby(category)["abs_log_error"].mean().sort_values(ascending=False)

# Visualizza il grafico
plt.figure(figsize=(8, 5))
sns.barplot(x=grouped.values, y=grouped.index)
plt.title(f"Errore logaritmico medio per categoria '{category}'")
plt.xlabel("Errore log medio (|log(y) - log(pred)|)")
plt.ylabel(category)
plt.show()