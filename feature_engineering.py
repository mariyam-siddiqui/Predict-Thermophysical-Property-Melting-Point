# %%
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.Descriptors import MolWt

def features(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return {
            'molecular_formula': None,
            'num_atoms': None,
            'num_bonds': None,
            'mol_weight': None
        }
    return {
        'molecular_formula': CalcMolFormula(mol),
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
        'mol_weight': MolWt(mol)
    }
    
    # for bond in type_bonds:
    # print(f"Bond between Atoms {bond.GetBeginAtomIdx()} and {bond.GetEndAtomIdx()}, Type: {bond.GetBondType()}")
