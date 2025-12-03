# %%
# !pip install rdkit
# !pip install shap
# !pip install xgboost

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap

# %%
df = pd.read_csv("dataset/train.csv")

# %%
df.head()

# %%
df.info()

# %%
df.columns

# %%
# Understanding the data 
# group 1 to 424
# id - name
# SMILES - molecular structure
# Tm - Melting Point

# %%
df.describe()

# %%
# Histogram of melting points

plt.hist(df["Tm"], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel("Temperature Bins")
plt.ylabel("Count")

plt.show()

# %%
# FEATURE ENGINEERING 

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem import Descriptors


from collections import defaultdict

all_bonds = set()

def features(smile):

    mol = Chem.MolFromSmiles(smile)

    if mol is None:
        return {
            'molecular_formula': None,
            'num_atoms': None,
            'num_bonds': None,
            'mol_weight': None,
            # 'total_bond_count': bond_counts,
            'single_bond_count' : None,
            'double_bond_count' : None,
            'triple_bond_count' : None,
            'aromatic_bond_count': None
        }
    
    bond_counts = defaultdict(int)
    for bond in mol.GetBonds():
        bond_type = str(bond.GetBondType())
        bond_counts[bond_type] += 1
        all_bonds.add(bond_type)

    single_bonds = bond_counts.get('SINGLE', 0)
    double_bonds = bond_counts.get('DOUBLE', 0)
    triple_bonds = bond_counts.get('TRIPLE', 0)
    aromatic_bonds = bond_counts.get('AROMATIC', 0)

    descriptors = {}

    for name, func in Descriptors.descList:
        try:
            descriptors[name] = func(mol)
        except:
            descriptors[name] = None
    
    base_features = {
        'molecular_formula': CalcMolFormula(mol),
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
        'mol_weight': MolWt(mol),
        # 'total_bond_count': total_bonds,
        'single_bond_count': single_bonds,
        'double_bond_count': double_bonds,
        'triple_bond_count': triple_bonds,
        'aromatic_bond_count': aromatic_bonds
    }

    all_features = {**base_features, **descriptors}

    return all_features
    

    # for bond in type_bonds:
    # print(f"Bond between Atoms {bond.GetBeginAtomIdx()} and {bond.GetEndAtomIdx()}, Type: {bond.GetBondType()}")




# %%
# Assume your DataFrame is named df and has a column 'smiles'
features_df = df['SMILES'].apply(features).apply(pd.Series)
df = pd.concat([df, features_df], axis=1)

# %%
# Drop Constant Features
constant_features = [col for col in df.columns if df[col].nunique()==1]

# %%
constant_features

# %%
df.columns

# %%
df = df.drop(columns = constant_features)


# %%
df.head()

# %%
cols = [x for x in df.columns if x not in ['id',
 'SMILES',
 'Tm',
'molecular_formula']]

# %%
from sklearn.model_selection import train_test_split

X = df[cols]
y = df["Tm"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle = True, random_state= 5)


# %%
# !pip install boruta

# %%
# BORUTA FEATURE SELECTION
from boruta import BorutaPy

rf = RandomForestRegressor()
boruta = BorutaPy(rf)
boruta.fit(X_train.values, y_train.values)

selected_features = X_train.columns[boruta.support_].tolist()
print(f"✅ Selected {len(selected_features)} features using Boruta")

X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]


# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
# model = RandomForestRegressor()

params = {
    'n_estimators' : 500,
    'learning_rate' : 0.05,
    'max_depth' : 50,
    'subsample' : 0.8,
    'colsample_bytree' : 0.8,
    "random_state" : 42,
    "n_jobs" : -1
}

model = XGBRegressor(
    n_estimators = 1000,
    learning_rate = 0.09,
    max_depth = None
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# %%

mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")

mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y_test, predictions)
print(f'R-squared: {r2}')

# %%
print(hasattr(model, "feature_names_in_"))

# %%
# Predict Values of test.csv

test = pd.read_csv("dataset/test.csv")

# %%
test.shape

# %%
test.head()

# %%
features_test = test['SMILES'].apply(features).apply(pd.Series)
test_full = pd.concat([test, features_test], axis = 1)

# %%
test = test.drop(columns=constant_features)




# %%
Tm_pred = model.predict(test_full[cols])
test_full["Tm_pred"] = Tm_pred

# %%
ans = pd.DataFrame()
ans["id"] = test_full["id"]
ans["Tm"] = test_full["Tm_pred"]

# %%
# ans.to_csv("Submission4_01112025.csv", index = False)
# %%
