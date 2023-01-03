
"""
Find model data along points by interpolating between points. 
Points can either be read from a file (if many), or given as input when run (if only two points)
-Given two or more points. 
- Given one point and a degree line

"""

from weathervis.config import *
from weathervis.utils import (
    filter_values_over_mountain,
    default_map_projection,
    default_mslp_contour,
    plot_by_subdomains,
    default_arguments,
    chunck_func_call,
    adjustable_colorbar_cax,
    nice_legend,
    nicegrid,
    setup_directory
)

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from weathervis.domain import *
from weathervis.check_data import *
from weathervis.get_data import *
from weathervis.calculation import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable  ##__N
from weathervis.checkget_data_handler import checkget_data_handler
import warnings
import gc
from util.make_cross_section import *


def plot_Vertical_cross_section():
    pass

def Vertical_cross_section(kwargs):
    param = ["air_temperature_ml","surface_geopotential","surface_air_pressure","specific_humidity_ml"]
    kwargs.m_level = np.arange(0, 65, 1) #[0,1,2]
    nya = (78.9243,11.9312)
    ands = (69.310, 16.120)
    coos2 = [((nya[1], nya[0]), (ands[1], ands[0]))]

    dmet, data_domain, bad_param = checkget_data_handler(all_param=param, model=kwargs.model,  date=kwargs.datetime,
                                                         step=kwargs.steps, m_level=kwargs.m_level,
                                                         point_name=kwargs.point_name, num_point=kwargs.num_point)
    parallels = dmet.standard_parallel_projection_lambert
    center_longitude = dmet.longitude_of_central_meridian_projection_lambert
    center_latitude = dmet.latitude_of_projection_origin_projection_lambert
    P = ml2pl(dmet.ap, dmet.b, dmet.surface_air_pressure)
    H = pl2alt(dmet.ap, dmet.b, dmet.surface_air_pressure, dmet.air_temperature_ml, dmet.specific_humidity_ml, P,dmet.surface_geopotential)
    heights = np.arange(0, 5000, 100)  # give the height interval that the field is goid to be interpolated to. currently lowest 5000 m in 100m steps
    dmet.air_temperature_ml = dmet.air_temperature_ml[0,:,:,:]
    P = P[0, :, :, :]
    HH = H[0, :, :, :]

    cross = CrossSection(
                {
                    "lat": dmet.latitude,
                    "rlat": dmet.y,
                    "lon": dmet.longitude,
                    "rlon": dmet.x,
                    "TT": dmet.air_temperature_ml,
                    "z": HH,
                    "p": P / 100.0
                },
                coo = coos2[0],
                pressure = heights,
                version="rotated",#"rotated",
                pollon=None,
                pollat=None,
                flip=True,
                int2z=True,
                model=kwargs.model,
                parallels=parallels, 
                center_longitude=center_longitude, 
                center_latitude = center_latitude 
            )  # polgam=180,

    x, zi = np.meshgrid(cross.lon, cross.pressure)
    fig, ax = plt.subplots()    
    pc = ax.contourf( x, zi, cross.TT, cmap="gnuplot2_r", extend="both" )
    plt.colorbar(pc)
    plt.show()

if __name__ == "__main__":
    args = default_arguments()

    chunck_func_call(func=Vertical_cross_section,chunktype= args.chunktype, chunk=args.chunks,  kwargs=args)
    gc.collect()

