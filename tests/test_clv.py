import pytest
import pandas as pd
import numpy as np
from src.clv_engine import CLVEngine

def test_clv_prediction_positive():
    engine = CLVEngine()
    sample = pd.read_csv('data/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv', nrows=5)
    preds = engine.predict_clv(sample)
    
    assert len(preds) == 5
    assert np.all(preds > 0.0)
