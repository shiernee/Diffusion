import os
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from src.pointcloud.GeneratePoints import *
from src.utils.DataFrame import DataFrame
from pathlib import Path


if __name__ == '__main__':

    no_pt = 5000

    sph_radius = 1
    parent_file = path.dirname(path.dirname(path.abspath(__file__)))
    filename = os.path.join(parent_file, "data", "testcase2", "database.csv")

    path = Path(filename)

    gp = GenerateSphPoints_NormalXYZ(no_pt, sph_radius)
    # gp = GenerateSphPoints_UniformPhiTheta(no_pt, sph_radius)
    # gp = GenerateRegular2DGrid(no_pt, max_x=1, max_y=1)
    # gp = GenerateScatter2DPoints(no_pt, max_x=1, max_y=1)
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

