import os
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'clv_pipeline.joblib')

class CLVEngine:
    def __init__(self):
        saved = joblib.load(MODEL_PATH)
        self.model = saved['model']
        self.scaler = saved['scaler']
        self.ohe = saved['ohe']
        self.num_cols = saved['num_cols']
        self.cat_cols = saved['cat_cols']

    def predict_clv(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df.columns = [c.replace(' ', '_').lower() for c in df.columns]
        
        for c in self.cat_cols:
            if c not in df.columns:
                df[c] = 'MISSING'
            df[c] = df[c].fillna('MISSING')
            
        for c in self.num_cols:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = df[c].fillna(0.0)
            
        enc = np.hstack([df[self.num_cols].values, self.ohe.transform(df[self.cat_cols])])
        scaled = self.scaler.transform(enc)
        log_preds = self.model.predict(scaled)
        return np.expm1(log_preds)
