
import pandas as pd
import numpy as np


class DataFrame:
    def __init__(self, filename):
        self.filename = filename
        self.df = pd.read_csv('{}'.format(self.filename), index_col=False)

    def empty_df(self):
        columns = ['x', 'y', 'z']
        self.df = pd.DataFrame(columns=columns)

    def get_df(self):
        return self.df

    def get_coord(self):
        coord = self.df[['x', 'y', 'z']].values
        if coord.dtype != 'float64':
            coord = np.asarray(coord, dtype='float64')
        return coord

    def get_uni_u(self):
        u = self.df['uni_u'].values
        if u.dtype != 'float64':
            u = np.asarray(u, dtype='float64')
        return u

    def get_uni_w(self):
        w = self.df['uni_w'].values
        if w.dtype != 'float64':
            w = np.asarray(w, dtype='float64')
        return w

    def get_bipolar_V(self):
        bipolar_V =self.df['bipolar_V'].values
        if bipolar_V.dtype != 'float64':
            bipolar_V = np.asarray(bipolar_V, dtype='float64')
        return bipolar_V

    def get_D(self):
        D = self.df['D'].values
        if D.dtype != 'float64':
            D = np.asarray(D, dtype='float64')
        return D

    @staticmethod
    def _string_to_nparray(string):
        a = string.split()  # split string to list
        b = list(map(float, a))  # convert string elements to float elements
        values = np.asarray(b)  # convert list to numpy array
        return values

    def update_df(self, column, values, idx=None):
        self.df['{}'.format(column)] = " "
        if idx is None:
            self.df['{}'.format(column)] = values
        else:
            self.df['{}'.format(column)].iloc[idx] = list(values)

    def _convert_to_string(self, column):
        self.df['{}'.format(column)] = self.df['{}'.format(column)].astype(str)

    def save_df(self):
        self.df.to_csv('{}'.format(self.filename), index=False, index_label=False)

