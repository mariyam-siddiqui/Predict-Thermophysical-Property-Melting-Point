# %%
# !pip install optuna boruta shap xgboost rdkit

# %%
import pandas as pd
import numpy as np
import shap
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem import Descriptors
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import optuna

# %% ------------------------- FEATURE EXTRACTION -------------------------

def features(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return {'molecular_formula': None, 'num_atoms': None, 'num_bonds': None, 
                'mol_weight': None, 'single_bond_count': None, 
                'double_bond_count': None, 'triple_bond_count': None, 'aromatic_bond_count': None}
    
    bond_counts = defaultdict(int)
    for bond in mol.GetBonds():
        bond_counts[str(bond.GetBondType())] += 1
    
    descriptors = {}
    for name, func in Descriptors.descList:
        try:
            descriptors[name] = func(mol)
        except:
            descriptors[name] = None
    
    base = {
        'molecular_formula': CalcMolFormula(mol),
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
        'mol_weight': MolWt(mol),
        'single_bond_count': bond_counts.get('SINGLE', 0),
        'double_bond_count': bond_counts.get('DOUBLE', 0),
        'triple_bond_count': bond_counts.get('TRIPLE', 0),
        'aromatic_bond_count': bond_counts.get('AROMATIC', 0)
    }
    return {**base, **descriptors}

# %% ------------------------- DATA LOAD -------------------------

df = pd.read_csv("dataset/train.csv")

features_df = df['SMILES'].apply(features).apply(pd.Series)
df = pd.concat([df, features_df], axis=1)
df = df.drop(columns=[col for col in df.columns if df[col].nunique() == 1])

cols = [x for x in df.columns if x not in ['id','SMILES','Tm','molecular_formula']]
X = df[cols].fillna(0)
y = df["Tm"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# %% ------------------------- BORUTA FEATURE SELECTION -------------------------

# rf = RandomForestRegressor()
# boruta = BorutaPy(rf, n_estimators='auto')
# boruta.fit(X_train.values, y_train.values)

# selected_features = X_train.columns[boruta.support_].tolist()
# print(f"✅ Selected {len(selected_features)} features using Boruta")

# X_train_sel = X_train[selected_features]
# X_test_sel = X_test[selected_features]

X_train_sel = X_train
X_test_sel = X_test


# %% ------------------------- OPTUNA TUNING -------------------------

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 800, 2000),
        # 'max_depth': trial.suggest_int('max_depth', 0, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        # 'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        # 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        # 'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        # 'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True)
    }

    model = XGBRegressor(**params)
    model.fit(X_train_sel, y_train, eval_set=[(X_test_sel, y_test)], verbose=False)
    preds = model.predict(X_test_sel)
    mae = mean_absolute_error(y_test, preds)
    return mae

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best trial:")
print(study.best_trial.params)

# %% ------------------------- FINAL MODEL -------------------------

best_params = study.best_trial.params
best_model = XGBRegressor(**best_params)
best_model.fit(X_train_sel, y_train)

preds = best_model.predict(X_test_sel)

mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"✅ MAE: {mae:.4f}")
print(f"✅ MSE: {mse:.4f}")
print(f"✅ R2: {r2:.4f}")

# %% ------------------------- PREDICT TEST.CSV -------------------------

test = pd.read_csv("dataset/test.csv")
features_test = test['SMILES'].apply(features).apply(pd.Series)
test_full = pd.concat([test, features_test], axis=1)

# test_full = test_full.fillna(0)
Tm_pred = best_model.predict(features_test)
test_full["Tm_pred"] = Tm_pred

ans = pd.DataFrame({
    "id": test_full["id"],
    "Tm": test_full["Tm_pred"]
})

ans.to_csv("Submission_with_Boruta_Optuna.csv", index=False)
print("✅ Submission file saved as 'Submission_with_Boruta_Optuna.csv'")

# %%
