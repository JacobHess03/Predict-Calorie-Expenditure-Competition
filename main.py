import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures, PowerTransformer
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. Caricamento dati
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# 2. Conserva ID e rimuovi
train_ids = df_train.pop('id')
test_ids = df_test.pop('id')

# 3. Feature engineering base per entrambi
def feature_engineering(df):
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df['Age_Group'] = pd.cut(df['Age'], bins=[0,18,40,65,100], labels=[0,1,2,3]).astype(int)
    return df

df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)

# 4. Encoding categorico
le = LabelEncoder()
df_train['Sex'] = le.fit_transform(df_train['Sex'])
df_test['Sex'] = le.transform(df_test['Sex'])

# 5. Rimozione outlier rapido (z-score)
num_cols = ['Age','Height','Weight','BMI']
z = np.abs((df_train[num_cols] - df_train[num_cols].mean()) / df_train[num_cols].std())
mask = (z < 3).all(axis=1)
df_train, train_ids = df_train[mask], train_ids[mask]

# 6. Preparazione X, y e trasformazione target
X = df_train.drop(columns=['Calories'])
y = np.log1p(df_train['Calories'])

# 7. Split per validazione
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Trasformazioni: scaling + power + polinomiali
scaler = StandardScaler()
pt = PowerTransformer(method='yeo-johnson')
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# Fit trasformazioni su train
X_train_scaled = scaler.fit_transform(X_train)
X_train_pt = pt.fit_transform(X_train_scaled)
X_train_poly = poly.fit_transform(X_train_pt)

# Trasforma validation
X_val_scaled = scaler.transform(X_val)
X_val_pt = pt.transform(X_val_scaled)
X_val_poly = poly.transform(X_val_pt)

# 9. Definizione modello e ricerca iperparametri leggera
xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=4)
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.03, 0.05],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 1, 5],
    'reg_alpha': [0, 1, 10],
    'reg_lambda': [1, 10, 100]
}

rand_search = RandomizedSearchCV(
    xgb, param_distributions=param_dist,
    n_iter=15, cv=3, scoring='r2', verbose=1, random_state=42, n_jobs=4
)
rand_search.fit(X_train_poly, y_train)
best = rand_search.best_estimator_
print('Best params:', rand_search.best_params_)

# 10. Valutazione su validation
y_val_pred_log = best.predict(X_val_poly)
y_val_pred = np.expm1(y_val_pred_log)
y_val_true = np.expm1(y_val)
print('Validation R2:', r2_score(y_val_true, y_val_pred))
print('Validation RMSE:', np.sqrt(mean_squared_error(y_val_true, y_val_pred)))

# 11. Retrain su full train con trasformazioni complete
X_full_scaled = scaler.fit_transform(X)
X_full_pt = pt.fit_transform(X_full_scaled)
X_full_poly = poly.fit_transform(X_full_pt)
y_full = y
best.set_params(n_estimators=best.best_iteration if hasattr(best, 'best_iteration') else best.n_estimators)
best.fit(X_full_poly, y_full)

# 12. Prepara test e predici
X_test = df_test
X_test_scaled = scaler.transform(X_test)
X_test_pt = pt.transform(X_test_scaled)
X_test_poly = poly.transform(X_test_pt)
y_test_log = best.predict(X_test_poly)
y_test = np.expm1(y_test_log)

# 13. Creazione submission
submission = pd.DataFrame({'id': test_ids, 'Calories': y_test})
submission.to_csv('submission.csv', index=False)
print('Submission salvata con iperparametri ottimizzati.')
