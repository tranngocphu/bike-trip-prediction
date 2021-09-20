import os
from joblib import load
from sklearn.base import BaseEstimator, TransformerMixin


# The following is_peak() function and AddPeakAndMinutes class
# are required to load saved transformer from *.joblib

def is_peak(t, windows):
    for window in windows:
        if t >= window[0] and t<= window[1]:
            return 1
    return 0


class AddPeakAndMinutes(BaseEstimator, TransformerMixin):
    def __init__(self, peak_hours=[[7,9], [17,19]], peak_months=[[6,10]]):
        self.peak_hours = peak_hours
        self.peak_months = peak_months
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        # add peak hour indicator
        X['PeakHour'] = X['Phour'].apply(lambda x: is_peak(x, self.peak_hours))
        # add peak month indicator
        X['PeakMonth'] = X['Pmonth'].apply(lambda x: is_peak(x, self.peak_months))
        # add minutes from midnight
        X['Minutes'] = X['Phour']*60 + X['Pmin']
        return X