from src.pointcloud.GeneratePoints import *
from src.utils.DataFrame import DataFrame
from pathlib import Path

if __name__ == '__main__':

    no_pt = 1000
    sph_radius = 1
    filename = 'C:\\Users\sawsn\Desktop\Shiernee\Diffusion\data\\testcase\\database.csv'
    path = Path(filename)

    gp = GenerateSphPoints(no_pt, sph_radius)
    gp.create_README_file(path=path.parent)

    dataframe = DataFrame(filename)
    dataframe.empty_df()

    dataframe.update_df('x', gp.cart_coord[:, 0])
    dataframe.update_df('y', gp.cart_coord[:, 1])
    dataframe.update_df('z', gp.cart_coord[:, 2])
    dataframe.save_df()

    # max_x, max_y = 1, 1
    #
    # GenerateNiederreiterDatasets(no_pt, max_x, max_y)
    #
    # GenerateRegular2DGrid(no_pt, max_x, max_y)
    #
    # GenerateScatter2DPoints(no_pt, max_x, max_y)
    #

