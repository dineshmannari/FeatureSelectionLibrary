import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class FeatureSelector:
    def __init__(self, th=1):
        self.th = th
        
    def preprocess(self, df):
        cols_with_qmark = []
        for itr, col in enumerate(df.columns):
            try:
                x = df[col].str.contains('\?').sum()
                cols_with_qmark.append(col)
                print(f"{col} - {x}")
            except:
                pass

        df = df.replace('?', np.nan)
        df = df.drop([13], axis=1)
        cols_with_qmark.remove(13)

        for itr, col in enumerate(cols_with_qmark):
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[10].fillna(df[col].mean())

        for itr, col in enumerate(df.columns):
            if len(df[col].unique().tolist()) == 1:
                df = df.drop([col], axis=1)

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        scaler = MinMaxScaler()
        X_norm = scaler.fit(X)
        X_norm = scaler.transform(X)
        self.X_norm_df = pd.DataFrame(X_norm)

        self.y = y.values
    
    def find_strong_relevance_features(self):
        F = self.X_norm_df.values
        C = self.y
        th = self.th
        
        D = F.shape[1]  # number of features
        SU = np.zeros(D)  # array to store the relevance values
        for i in range(D):
            if sum(F[:, i] == 1) == 0 or sum(F[:, i] == 0) == 0:
                SU[i] = 0
            else:
                SU[i] = sum(C == 1) / sum(F[:, i] == 1) - sum(C == 0) / sum(F[:, i] == 0)
        SUmax = max(SU)
        # Determine the threshold value
        rho0 = min(0.1 * SUmax, SU[D-1] / np.log2(D) - th)
        # Find the strong relevance features
        F_strong = []
        for i in range(D):
            if SU[i] >= rho0:
                F_strong.append(i)
        return F[:, F_strong], F_strong, SU

# Load your data
cols = [i for i in range(280)]
df = pd.read_csv('D:/MS sem2/DM/Project/arrhythmia.data', names=cols)

# Instantiate the feature selector
selector = FeatureSelector(th=1)

# Preprocess the data
selector.preprocess(df)

# Apply feature selection
selected_features, selected_indices, SU = selector.find_strong_relevance_features()

# Print the selected indices and the number of selected features
print(selected_indices)
print(len(selected_indices))
