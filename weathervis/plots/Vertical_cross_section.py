
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


def plot_Vertical_cross_section(cross):
    x, zi = np.meshgrid(cross.lon, cross.pressure)
    fig, ax = plt.subplots()    
    pc = ax.contourf( x, zi, cross.air_temperature_ml, cmap="gnuplot2_r", extend="both" )
    plt.colorbar(pc)
    plt.show()


def Vertical_cross_section(kwargs):
    #types_readfromfile or as input
    #types_Hgrid = nearest neigbour, interpolation of 4 closest: pre and pro process
    #types_position = exact, derived:    preprocess
    #types Zgrid = asmodel, interpolated:postprocess
    #types_Ttime = from 1 modelrun, from multiple modelruns glue: retrievalprocess
    #save retriaval or use saved retriaval: pre and post process.
    #the most optimal?
    param = ["air_temperature_ml","surface_geopotential","surface_air_pressure","specific_humidity_ml"]
            #,
            #"x_wind_ml", "y_wind_ml", "cloud_area_fraction_ml", "upward_air_velocity_ml",
            # "mass_fraction_of_cloud_ice_in_air_ml", "mass_fraction_of_cloud_condensed_water_in_air_ml",
            # "mass_fraction_of_graupel_in_air_ml", "mass_fraction_of_snow_in_air_ml",
            # "mass_fraction_of_rain_in_air_ml", "atmosphere_boundary_layer_thickness","air_pressure_at_sea_level"]
    kwargs.m_level = np.arange(0, 65, 1) #[0,1,2]
    nya = (78.9243,11.9312)
    ands = (69.310, 16.120)
    coos2 = [((nya[1], nya[0]), (ands[1], ands[0]))]

    
    dmet, data_domain, bad_param = checkget_data_handler(all_param=param,save=True,read_from_saved=True, model=kwargs.model,  date=kwargs.datetime,
                                                         step=kwargs.steps, m_level=kwargs.m_level,
                                                         point_name=kwargs.point_name, num_point=kwargs.num_point)
    
    
    parallels = dmet.standard_parallel_projection_lambert
    center_longitude = dmet.longitude_of_central_meridian_projection_lambert
    center_latitude = dmet.latitude_of_projection_origin_projection_lambert
    P = ml2pl(dmet.ap, dmet.b, dmet.surface_air_pressure)
    H = pl2alt(dmet.ap, dmet.b, dmet.surface_air_pressure, dmet.air_temperature_ml, dmet.specific_humidity_ml, P,dmet.surface_geopotential)
    heights = np.arange(0, 5000, 100)  # give the height interval that the field is goid to be interpolated to. currently lowest 5000 m in 100m steps


    cross = CrossSection(
                {
                    "lat": dmet.latitude,
                    "rlat": dmet.y,
                    "lon": dmet.longitude,
                    "rlon": dmet.x,
                    "air_temperature_ml": dmet.air_temperature_ml[0,:,:,:],
                    "specific_humidity_ml":dmet.specific_humidity_ml[0,:,:,:],
                    #"x_wind_ml":dmet.x_wind_ml[0,:,:,:],
                    #"y_wind_ml":dmet.y_wind_ml[0,:,:,:],
                    #"cloud_area_fraction_ml":dmet.cloud_area_fraction_ml[0,:,:,:],
                    #"upward_air_velocity_ml":dmet.upward_air_velocity_ml[0,:,:,:],
                    #"mass_fraction_of_cloud_ice_in_air_ml":dmet.mass_fraction_of_cloud_ice_in_air_ml[0,:,:,:],
                    #"mass_fraction_of_cloud_condensed_water_in_air_ml":dmet.mass_fraction_of_cloud_condensed_water_in_air_ml[0,:,:,:],
                    #"mass_fraction_of_graupel_in_air_ml":dmet.mass_fraction_of_graupel_in_air_ml[0,:,:,:],
                    #"mass_fraction_of_snow_in_air_ml":dmet.mass_fraction_of_snow_in_air_ml[0,:,:,:],
                    #"mass_fraction_of_rain_in_air_ml":dmet.mass_fraction_of_rain_in_air_ml[0,:,:,:],
                    #"atmosphere_boundary_layer_thickness":dmet.atmosphere_boundary_layer_thickness[0,:,:],
                    #"air_pressure_at_sea_level":dmet.air_pressure_at_sea_level[0,:,:],
                    #"surface_air_pressure":dmet.surface_air_pressure[0,:,:],
                    "z": H[0, :, :, :],
                    "p": P[0, :, :, :] / 100.0
                },
                coo = coos2[0],
                nbre=10,
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

    plot_Vertical_cross_section(cross)

if __name__ == "__main__":
    args = default_arguments()

    chunck_func_call(func=Vertical_cross_section,chunktype= args.chunktype, chunk=args.chunks,  kwargs=args)
    gc.collect()

