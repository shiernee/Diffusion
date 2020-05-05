import os
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from src.utils.DataFrame import DataFrame
import numpy as np

if __name__ == '__main__':
    parent_file = path.dirname(path.dirname(path.abspath(__file__)))
    filename = os.path.join(parent_file, "data", "testcase", "database.csv")

    dataframe = DataFrame(filename)
    coord = dataframe.get_coord()
    no_pt = len(coord)

    D = 1 * np.ones([no_pt, ])
    # HK D = 0 * np.ones([no_pt, ])
    a = 0.1 * np.ones([no_pt, ])
    epsilon = 0.01 * np.ones([no_pt, ])
    beta = 0.5 * np.ones([no_pt, ])
    gamma = 1 * np.ones([no_pt, ])
    delta = 0 * np.ones([no_pt, ])

    dataframe.update_df('D', D)
    dataframe.update_df('a', a)
    dataframe.update_df('epsilon', epsilon)
    dataframe.update_df('beta', beta)
    dataframe.update_df('gamma', gamma)
    dataframe.update_df('delta', delta)
    dataframe.save_df()






