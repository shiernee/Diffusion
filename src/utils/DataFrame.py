
import pandas as pd
import numpy as np


class DataFrame:
    def __init__(self, filename):
        self.filename = filename
        self.df = pd.read_csv('{}'.format(self.filename), index_col=False)

    def empty_df(self):
        columns = self.df.columns
        self.df = pd.DataFrame(columns=columns)

    def get_df(self):
        return self.df

    def get_coord(self):
        return self.df[['x', 'y', 'z']].values

    def get_uni_u(self):
        return self.df['uni_u'].values

    def get_uni_w(self):
        return self.df['uni_w'].values

    def get_bipolar_V(self):
        return self.df['bipolar_V'].values

    def get_D(self):
        return self.df['D'].values

    def string_to_nparray(self, string):
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

    def convert_to_string(self, column):
        self.df['{}'.format(column)] = self.df['{}'.format(column)].astype(str)

    def save_df(self):
        self.df.to_csv('{}'.format(self.filename), index=False, index_label=False)

