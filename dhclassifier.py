import numpy as np
import pandas as pd
import joblib
import xgboost
from janitor.chemistry  import molecular_descriptors, morgan_fingerprint
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Draw import IPythonConsole
import pandas.io.formats.html

PandasTools.molSize = (300,300)

KeyDes=['ExactMolWt', 'NumHBA', 'NumHBD', 'MolLogP', 'NumRotatableBonds',
       'TPSA', 'FractionCSP3', 'HallKierAlph', 'Kappa1', 'Kappa2', 'LabuteASA',
       'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
       'NumAliphaticRings', 'NumAmideBonds', 'NumAromaticCarbocycles',
       'NumAromaticRings', 'NumAtomStereoCenters', 'NumBridgeheadAtoms',
       'NumHeterocycles', 'NumRings', 'NumSaturatedCarbocycles',
       'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumSpiroAtoms',
       'NumUnspecifiedAtomStereoCenters', 'qed', 'NumValenceElectrons',
       'BertzCT', 'NHOHCount', 'NOCount', 'MolMR', 'HeavyAtomCount',
       'fr_benzene', 'fr_bicyclic', 'fr_halogen']


mean=np.array([ 3.54710813e+02,  4.82845528e+00,  1.78644986e+00,  1.86165265e+00,
        4.90379404e+00,  8.49120407e+01,  4.23355939e-01, -1.76230659e+00,
        1.94759605e+01,  8.69804968e+00,  1.46530865e+02,  4.23577236e-01,
        5.93766938e-01,  1.01734417e+00,  4.79674797e-01,  1.02845528e+00,
        1.51192412e+00,  1.56260163e+00,  1.34959350e-01,  1.07723577e+00,
        2.52926829e+00,  2.82113821e-01,  4.16260163e-01,  6.98373984e-01,
        2.49322493e-02,  5.48509485e-01,  5.38579173e-01,  1.28857724e+02,
        7.05219482e+02,  2.16585366e+00,  5.79756098e+00,  9.05070391e+01,
        2.38850949e+01,  1.02926829e+00,  8.75609756e-01,  7.78319783e-01])

std=np.array([1.41948852e+02, 3.01223727e+00, 1.89503085e+00, 3.18217573e+00,
       3.77723585e+00, 6.02234420e+01, 2.55025000e-01, 1.31558262e+00,
       2.24732563e+01, 1.90167674e+01, 5.78689786e+01, 1.03253508e+00,
       9.09582229e-01, 1.37444535e+00, 9.06136366e-01, 8.96366600e-01,
       1.14353171e+00, 2.53957770e+00, 6.34421986e-01, 1.16664911e+00,
       1.64720773e+00, 7.73181461e-01, 7.59974674e-01, 1.09426785e+00,
       1.70870107e-01, 9.87838327e-01, 2.20889469e-01, 5.40773677e+01,
       4.10450765e+02, 2.44375591e+00, 3.54201740e+00, 3.79259986e+01,
       1.00314669e+01, 8.97700297e-01, 1.53179945e+00, 1.25179899e+00])

        
LR=joblib.load('./Trained_Models/LR.pkl')
RF=joblib.load('./Trained_Models/RF.pkl')
SVM=joblib.load('./Trained_Models/SVM.pkl')
XG=joblib.load('./Trained_Models/XG.pkl')



def DHClassifier(smiles, Standardize=False):
    if isinstance(smiles, str):
        smiles=[smiles]
    Compounds=pd.DataFrame(data=smiles, columns=['SMILES'])
    PandasTools.AddMoleculeColumnToFrame(Compounds, "SMILES", "mol")
    Descriptors=GenerateDescriptors(Compounds)
    StdDescriptors=StandardizeDescriptors(Descriptors)
    
    Compounds[['XG_Drug','XG_Herbicide']]=XG.predict_proba(StdDescriptors.values)
    Compounds[['LR_Drug','LR_Herbicide']]=LR.predict_proba(StdDescriptors.values)
    Compounds[['RF_Drug','RF_Herbicide']]=RF.predict_proba(StdDescriptors.values)
    Compounds[['SVM_Drug','SVM_Herbicide']]=SVM.predict_proba(StdDescriptors.values)

    if Standardize==True:
    	return pd.concat([Compounds, StdDescriptors], axis=1)
    if Standardize==False:
    	return pd.concat([Compounds, Descriptors], axis=1)

def StandardizeDescriptors(df):
    df=df.sub(mean)
    df=df.div(std)
    return df

def GenerateDescriptors(df):
    Descriptors1=molecular_descriptors(df=df, mols_column_name='mol')
    Add_Des=['qed','NumValenceElectrons','NumRadicalElectrons','BertzCT','NHOHCount','NOCount','NumRotatableBonds','MolLogP','MolMR','HeavyAtomCount','fr_benzene','fr_bicyclic','fr_halogen']
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(Add_Des)
    Descriptors2=[]
    for i in df['SMILES']:
        tmp=calculator.CalcDescriptors(Chem.MolFromSmiles(i))
        tmp=list(tmp)
        Descriptors2.append(tmp)
    AllDes=pd.concat([Descriptors1,pd.DataFrame(columns=Add_Des,data=Descriptors2)],axis=1)
    UsedDes=pd.DataFrame()
    for i in KeyDes:
        UsedDes[i]=AllDes[i]
    return UsedDes
