
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
import matplotlib

def plot_Vertical_cross_section(cross):

    #print(vars(cross))
    #print(cross.dim.point)
    #print(cross.latitude)
    print("in plot")
    print(np.shape(cross.pressure))
    print(np.shape(cross.z))

    mi = 230#np.min((cross.air_temperature_ml[cross.z < 5000].min()))
    ma = 280#np.max((cross.air_temperature_ml[cross.z < 5000].max()))
    norm = matplotlib.colors.Normalize(vmin=mi, vmax=ma)
    cross.pressure = cross.pressure.squeeze()
    cross.z = cross.z.squeeze()

    cross.air_temperature_ml = cross.air_temperature_ml.squeeze()
    #print(np.shape(cross.air_temperature_ml))
    #print(np.shape(cross.pressure))
    #print(cross.latitude)
    #    lx, tx = np.meshgrid(model_data.m_level, obs_datetime)
    m_level = np.arange(0, 65, 1)
    
    try:
        if len(cross.time) > 1:
            x_ax = cross.time
            cross.air_temperature_ml=cross.air_temperature_ml.T
        else:
            x_ax = cross.latitude
    except:
        x_ax = cross.latitude
    
    
    x, zi = np.meshgrid(x_ax,m_level ) #height not pressure
    #print(np.shape(x))
    #print(np.shape(zi))
    #print(np.shape(cross.air_temperature_ml))
    #print(cross.air_temperature_ml)
    #print(cross.pressure[:,0])
    fig, ax = plt.subplots()    
    #pc = ax.contourf( x, zi, cross.air_temperature_ml, cmap="gnuplot2_r", extend="both" )
    #pc = ax.pcolormesh( x, zi, cross.air_temperature_ml, cmap="gnuplot2_r" )

    pc = ax.pcolormesh(x_ax, cross.z, cross.air_temperature_ml, cmap="gnuplot2_r", shading='nearest', zorder=1,norm=norm)
    ax.set_ylim([0, 5000])
    plt.gca().invert_xaxis()
    #plt.gca().invert_yaxis()

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
    line = ((nya[1], nya[0]), (ands[1], ands[0]))
    #I want nearest neigbour along line for some points, and not interpolated. input is an array of [[lon,lat], [lon,lat]...etc]
    Ainas_nearestN=False
    if Ainas_nearestN: 
        lon, lat, distance = find_cross_points(line, nbre=50)
        lon=np.append(line[0][0],lon); lon=np.append(lon,line[1][0])
        lat=np.append(line[0][1],lat); lat=np.append(lat,line[1][1] )
        query_points = [[lon, lat] for lon, lat in zip(lon,lat)]
        kwargs.point_lonlat = query_points
        print(query_points)
        #["NYA_Base","NYA_Zepp"]

    dmet, data_domain, bad_param = checkget_data_handler(all_param=param,save="buffer_newM.nc",read_from_saved="buffer_newM.nc", 
                                                        model=kwargs.model,  date=kwargs.datetime,
                                                        step=kwargs.steps,   m_level=kwargs.m_level,
                                                        point_name=kwargs.point_name, 
                                                        point_lonlat=kwargs.point_lonlat,
                                                        num_point=kwargs.num_point)
    print(dmet)
    print(dir(dmet))
    #exit(1)
    parallels = dmet.standard_parallel_projection_lambert
    center_longitude = dmet.longitude_of_central_meridian_projection_lambert
    center_latitude = dmet.latitude_of_projection_origin_projection_lambert
    P = ml2pl(ap=dmet.ap, b=dmet.b, surface_air_pressure= dmet.surface_air_pressure, dmet=dmet)

    H = pl2alt(ap= dmet.ap, b=dmet.b, surface_air_pressure= dmet.surface_air_pressure, air_temperature_ml= dmet.air_temperature_ml, specific_humidity_ml= dmet.specific_humidity_ml, pressure= P,surface_geopotential=dmet.surface_geopotential, dmet=dmet)
    setattr(dmet, "pressure", P)
    setattr(dmet, "z", H)
    

    print("done calc")
   
    Marvins_interp=True
    if Marvins_interp:
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
                    coo = line,
                    nbre=50,
                    pressure = np.arange(0,65), #heights  #Need to be the same as height for data or what u want to interp on
                    version="rotated",#"rotated",
                    pollon=None,
                    pollat=None,
                    flip=True,
                    int2z=False,
                    int2p=False,
                    model=kwargs.model,
                    parallels=parallels, 
                    center_longitude=center_longitude, 
                    center_latitude = center_latitude 
                )  # polgam=180,
        cross.longitude=cross.lon
        cross.latitude=cross.lat
    else:
        cross=dmet

    plot_Vertical_cross_section(cross)

if __name__ == "__main__":
    args = default_arguments()

    chunck_func_call(func=Vertical_cross_section,chunktype= args.chunktype, chunk=args.chunks,  kwargs=args)
    gc.collect()

