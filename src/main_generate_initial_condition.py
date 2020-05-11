import os
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

import numpy as np
from pathlib import Path
from src.utils.DataFrame import DataFrame
from Utils.Utils import xyz2r_phi_theta

def cos_theta(theta):
    return np.cos(theta), 'cos_theta'


def cos_x(x):
    return np.cos(x), 'cos_x'


def sin_theta_square_cos_phi(theta, phi):
    return np.sin(theta) ** 2 * np.cos(phi), 'sin_theta_square_cos_phi'


if __name__ == '__main__':
    parent_file = path.dirname(path.dirname(path.abspath(__file__)))
    filename = os.path.join(parent_file, "data", "testcase", "database.csv")
    param_file = os.path.join(parent_file, "data", "testcase", "param_template.csv")

    dataframe = DataFrame(filename)

    coord = dataframe.get_coord()
    x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
    r, phi, theta = xyz2r_phi_theta(x, y, z)

    # u0, init_cond = cos_x(x)
    # u0, init_cond = cos_theta(theta)
    u0, init_cond = sin_theta_square_cos_phi(theta, phi)
    w0 = np.zeros_like(u0)

    dataframe.update_df('uni_u', u0)
    dataframe.update_df('uni_w', w0)
    dataframe.save_df()

    path = Path(filename).parent
    filename = os.path.join(path, "README.txt")
    with open('{}'.format(filename), mode='a', newline='') as csv_file:
        csv_file.write('u0: {}\n'.format(init_cond))
        csv_file.write('w0: zeros\n')
    print('{}'.format(filename))



