import numpy as np
from pathlib import Path
from src.utils.DataFrame import DataFrame


def xyz2sph(cart_coord):
    x, y, z = cart_coord[:, 0], cart_coord[:, 1], cart_coord[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)

    phi = np.zeros_like(theta)
    idx = np.argwhere(x > 0.0).squeeze()
    phi[idx] = np.arctan(y[idx] / x[idx])

    idx = np.argwhere((x < 0.0) & (y >= 0)).squeeze()
    phi[idx] = np.arctan(y[idx] / x[idx]) + np.pi

    idx = np.argwhere((x < 0.0) & (y < 0)).squeeze()
    phi[idx] = np.arctan(y[idx] / x[idx]) - np.pi

    idx = np.argwhere((x == 0) & (y > 0)).squeeze()
    phi[idx] = np.pi / 2

    idx = np.argwhere((x == 0) & (y < 0)).squeeze()
    phi[idx] = - np.pi / 2

    idx = np.argwhere((x == 0.0) & (y == 0))
    phi[idx] = 0.0
    return r, phi, theta


def cos_theta(theta):
    return np.cos(theta), 'cos_theta'


def sin_theta_square_cos_phi(theta, phi):
    return np.sin(theta) ** 2 * np.cos(phi), 'sin_theta_square_cos_phi'

if __name__ == '__main__':

    filename = 'C:\\Users\sawsn\Desktop\Shiernee\Diffusion\data\\testcase\\database.csv'
    dataframe = DataFrame(filename)

    coord = dataframe.get_coord()
    r, phi, theta = xyz2sph(coord)

    # u0, init_cond = cos_theta(theta)
    u0, init_cond = sin_theta_square_cos_phi(theta, phi)
    w0 = np.zeros_like(u0)

    dataframe.update_df('uni_u', u0)
    dataframe.update_df('uni_w', w0)
    dataframe.save_df()

    path = Path(filename).parent
    with open('{}/README.txt'.format(path), mode='a', newline='') as csv_file:
        csv_file.write('u0: {}\n'.format(init_cond))
        csv_file.write('w0: zeros\n')
    print('/{}/README.txt'.format(path))



