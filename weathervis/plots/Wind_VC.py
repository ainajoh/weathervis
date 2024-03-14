
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
    setup_directory,
    point_name2point_lonlat
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
from weathervis.plots.OLR_map import plot_OLR,OLR_map #OLR_map.param
from matplotlib import gridspec
import matplotlib.ticker
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.size': 15})

def pre_defined_points_retrieved(line, kwargs):
    lon, lat, distance = find_cross_points(line, nbre=50)
    lon=np.append(line[0][0],lon); lon=np.append(lon,line[1][0])
    lat=np.append(line[0][1],lat); lat=np.append(lat,line[1][1] )
    query_points = [[lon, lat] for lon, lat in zip(lon,lat)]
    kwargs.point_lonlat = query_points
    kwargs.domain_name = None
    return kwargs

def move_cross_with_the_flow_or_observation(kwargs, line, nbre, speed=30, tbra=1,file=None ):
    one_modelrun=True
    ##########follow airmass###########
    #speed = 30 #"m/s" what speed to follow airmass
    #tbra = 1 #time resolution in hour if u want to follow airmass
    lon, lat, distance = find_cross_points(line, nbre=nbre)
    spand_hours = (distance*1000/speed)/(60*60) #or define ur own
    print(spand_hours)
    if one_modelrun:
        list_ = np.arange(0,round(spand_hours+tbra),tbra, dtype=int)/int(tbra)#f"{0}:{round(spand_hours)}:{tbra}"   #or alter modelrun to get closest
        kwargs.steps = list_.astype(int)
    else:
        #NOT IN USE OR TESTED YET
        step_max=6
        list_ = np.arange(0,round(step_max+tbra),tbra, dtype=int)/int(tbra)#f"{0}:{round(spand_hours)}:{tbra}"   #or alter modelrun to get closest
        #kwargs.steps = list_.astype(int)
        #for i in range(0,spand_hours/step_max): #if max hours = 12 we go through this loop two times
        #    print(i)
        #h = index.hour-index.hour%6
        #new_initial_modelrun = index.strftime('%Y%m%d') + str(h).zfill(2)
        #datetime=new_initial_modelrun
        #modeldattime = pd.to_datetime(datetime, format="%Y%m%d%H") # The chosen Modelrun datetime
        #datetimes = np.append(datetimes, datetime)  #save the modelrundatetime in an array
 
        #delta_t = index-modeldattime              #Difference beween the modelrun datetime and observation datetime
        #step = delta_t.round('60min').components.hours            # The lead time hour closest to our observation
        #steps = np.append(steps, step)            #save the steps in an array
    point_per_timestep = spand_hours/((nbre+1)*tbra)
    factor_to_one_point_per_timestep = 1/point_per_timestep
    point_per_timestep=[round(point_per_timestep*factor_to_one_point_per_timestep),round(1*factor_to_one_point_per_timestep)] #passing two points per timestep 
    time_index= np.arange(0,round(spand_hours+tbra),tbra, dtype=int)/tbra
    time_index = np.repeat(time_index[::point_per_timestep[0]].astype(int), point_per_timestep[1], axis=0)
    query_index = np.arange(0,nbre+1)
    join_pt = dict(zip(query_index, time_index))
    return kwargs, join_pt

def plot_map(cross, dmet, data_domain, kwargs):
    crs = default_map_projection(dmet)
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9), subplot_kw={'projection': crs})

    scale = find_scale(dmet.lonlat)
    plot_OLR(figax=[fig1,ax1], model = kwargs.model, datetime= kwargs.datetime,dmet=dmet,scale=scale,steps=kwargs.steps,lonlat=dmet.lonlat, save=False)
    ax1.scatter(cross.longitude, cross.latitude,transform=ccrs.PlateCarree(), marker = "o", color="blue",  edgecolor="blue", zorder=3 )    

    ax1.add_feature(cfeature.GSHHSFeature(scale="auto"))
    ax1.set_extent(dmet.lonlat)
    plt.show()
def plot_Vertical_cross_section(cross):
    """Use less"""
    #CAOi = CAO_index(cross.air_temperature_ml, cross.pressure, cross.SST,cross.air_pressure_at_sea_level, p_level=850)
    #print(CAOi)
    #potential_temperatur(temperature, pressure)
    #lapserate(T_ml, z, srf_T = None)
    #BL_height_sl
    #relative_humidity(temp,q,p)
    # wind_dir(xwind,ywind,


    # Humidity and wind plot
    #fig = plt.figure(figsize=(14, 12))
    #gs = gridspec.GridSpec(2, 1)
    #wdir = wind_dir(cross.x_wind_ml,cross.y_wind_ml, cross.alpha)
    #u, v = windfromspeed_dir(cross.WS, wdir)
    ########################################################
    #print(vars(cross))
    #print(cross.dim.point)
    #print(cross.latitude)
    print("in plot")



    mi = 230#np.min((cross.air_temperature_ml[cross.z < 5000].min()))
    ma = 280#np.max((cross.air_temperature_ml[cross.z < 5000].max()))
    norm = matplotlib.colors.Normalize(vmin=mi, vmax=ma)
    cross.pressure = cross.pressure.squeeze()
    cross.z = cross.z.squeeze()

    cross.air_temperature_ml = cross.air_temperature_ml.squeeze()
    cross.turbulent_kinetic_energy_ml = cross.turbulent_kinetic_energy_ml.squeeze()

    #print(np.shape(cross.air_temperature_ml))
    #print(np.shape(cross.pressure))
    #print(cross.latitude)
    #    lx, tx = np.meshgrid(model_data.m_level, obs_datetime)
    m_level = np.arange(0, 65, 1)
    
    try:
        if len(cross.time) > 1:
            x_ax = cross.time
            cross.air_temperature_ml=cross.air_temperature_ml.T
            #cross.turbulent_kinetic_energy_ml=cross.turbulent_kinetic_energy_ml.T
        else:
            x_ax = cross.latitude
    except:
        x_ax = cross.latitude
    
    
    #x, zi = np.meshgrid(x_ax,m_level ) #height not pressure
    x, zi = np.meshgrid(x_ax,cross.pressure ) #height not pressure

    #print(np.shape(x))
    #print(np.shape(zi))
    #print(np.shape(cross.air_temperature_ml))
    #print(cross.air_temperature_ml)
    #print(cross.pressure[:,0])
    fig, ax = plt.subplots(figsize=(14, 6))    
    #pc = ax.contourf( x_ax, cross.pressure, cross.air_temperature_ml, cmap="gnuplot2", extend="both" )

    #pc = ax.pcolormesh(x_ax, cross.z, cross.air_temperature_ml, cmap="gnuplot2", shading='nearest', zorder=1,norm=norm)
    #pc = ax.pcolormesh(x_ax, cross.pressure, cross.turbulent_kinetic_energy_ml, cmap="Reds", shading='nearest', zorder=1)#,norm=norm)
    print(np.shape(cross.ri))
    #pc = ax.pcolormesh(x_ax, cross.pressure, cross.turbulent_kinetic_energy_ml, cmap="Reds", shading='nearest', zorder=1)#,norm=norm)
    #norm = matplotlib.colors.Normalize(vmin=-500, vmax=500)
    lvl = [0.25, 0, -0.25, 0.5, 0.75, 1, 1.25, 1.5]
    pc = ax.pcolormesh(x_ax, cross.pressure, cross.ri,vmax=0.25, vmin=-5,shading='nearest', zorder=1, cmap="cool")#,norm=norm)
    #pc.cmap.set_over('gray')
    ax.text(78, 2150, 'Boundary Layer Height',fontsize=15, rotation=10)

    CS = ax.contour(x_ax, cross.pressure, cross.pt, colors="k",levels = 12)
    #fmt = matplotlib.ticker.StrMethodFormatter("x:1.0f")  #fmt=r'$\theta=$%1.0f'
    fmt = matplotlib.ticker.StrMethodFormatter(r"$\theta=${x:,g}")  #fmt=r'$\theta=$%1.0f'

    ax.clabel(CS, CS.levels, inline=False,fontsize=10, fmt=fmt)
    print( cross.atmosphere_boundary_layer_thickness )
    PBLH = ax.plot(x[-1, :], cross.atmosphere_boundary_layer_thickness, linewidth=5, color="k")
    
    plt.gca().invert_xaxis()
    cbar_ri=plt.colorbar(pc)
    cbar_ri.set_label('Richardson #',fontsize=15)
    ax.set_ylabel("Height [m]")
    ax.set_xlabel("Latitudes")
    
    ax.add_patch(Rectangle((79.5, -650), 4.5, 300, color='#70B6F4', alpha=0.8, clip_on=False)) #blue
    ax.add_patch(Rectangle((66.9, -650), 12.6, 300, color='#FD7173', alpha=0.8, clip_on=False)) #red
    plt.savefig("Ri.png",
                bbox_inches="tight",
                dpi=300,
            )

    #fig2, ax2 = plt.subplots(figsize=(14, 6))
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    tb = ax2.pcolormesh(x_ax, cross.pressure, cross.turbulent_kinetic_energy_ml, cmap="Reds", shading='nearest', zorder=1)#,norm=norm)
    PBLH = ax2.plot(x[-1, :], cross.atmosphere_boundary_layer_thickness, linewidth=5, color="k")
    
    #CS = ax2.contour(x_ax,cross.pressure, cross.cloud_area_fraction_ml, colors="white",levels = 12)
    #ax2.clabel(CS, CS.levels, inline=True,fontsize=10)  #cloud_area_fraction_ml

    plt.gca().invert_xaxis()
    cbar_tke= plt.colorbar(tb)
    cbar_tke.set_label(r'Turbulent Kinetic Energy $[m^2⋅s^{-2}]$',fontsize=15)
    cbar_tke.ax.tick_params(labelsize=15)
    ax2.set_ylabel("Height [m]")
    ax2.set_xlabel("Latitudes")
    #plt.savefig("TKE.svg")
    ax2.text(78, 2150, 'Boundary Layer Height',fontsize=15, rotation=10)
    #ax2.text('Boundary Layer Height', xy=(2, 8), xytext=(4, 10), fontsize=12)
    #plt.savefig("TKE.png",
    #            dpi=300,
    #        )
    #ax2.hlines(y=-1, colors="red")
    #ax2.add_patch(Rectangle((-0.2, -0.35), 11.2, 0.7, color='C1', alpha=0.8)) 500

    ax2.add_patch(Rectangle((79.5, -650), 4.5, 300, color='#70B6F4', alpha=0.8, clip_on=False)) #blue
    ax2.add_patch(Rectangle((66.9, -650), 12.6, 300, color='#FD7173', alpha=0.8, clip_on=False)) #red
    #ax2.add_patch(Rectangle((70, 80), 500, 1000, color='r', alpha=0.8))
    #70B6F4   #FD7173
    plt.savefig("TKE.png",
                bbox_inches="tight",
                dpi=300,
            )
    
    ##################################################################################

    fig3, ax3 = plt.subplots(figsize=(14, 6))    
    #pc3 = ax3.pcolormesh(x_ax, cross.pressure, cross.ri, vmax=20, vmin=-20, shading='nearest', zorder=1, cmap="PiYG")#,norm=norm)
    # norm=matplotlib.colors.LogNorm()
    pc3 = ax3.pcolormesh(x_ax, cross.pressure, cross.ri, vmax=20, vmin=-20, norm=matplotlib.colors.LogNorm(),shading='nearest', zorder=1, cmap="PiYG")#,norm=norm)

    CS = ax3.contour(x_ax, cross.pressure, cross.pt, colors="k",levels = 12)
    fmt = matplotlib.ticker.StrMethodFormatter(r"$\theta=${x:,g}")  #fmt=r'$\theta=$%1.0f'
    ax3.clabel(CS, CS.levels, inline=False,fontsize=10, fmt=fmt)
    
    PBLH = ax3.plot(x[-1, :], cross.atmosphere_boundary_layer_thickness, linewidth=5, color="k")
    ax3.text(78, 2150, 'Boundary Layer Height',fontsize=15, rotation=10)

    plt.gca().invert_xaxis()
    cbar_ri2=plt.colorbar(pc3, extend="both")
    cbar_ri2.set_label('Richardson #',fontsize=15)
    ax3.set_ylabel("Height [m]")
    ax3.set_xlabel("Latitudes")

    ax3.add_patch(Rectangle((79.5, -650), 4.5, 300, color='#70B6F4', alpha=0.8, clip_on=False)) #blue
    ax3.add_patch(Rectangle((66.9, -650), 12.6, 300, color='#FD7173', alpha=0.8, clip_on=False)) #red
    plt.savefig("Ri3.png",
                bbox_inches="tight",
                dpi=300,
            )
##################################################################################
    fig4, ax4 = plt.subplots(figsize=(14, 6))

    CS = ax4.contour(x_ax, cross.pressure, cross.pt, colors="k",levels = 12)
    fmt = matplotlib.ticker.StrMethodFormatter(r"$\theta=${x:,g}")  #fmt=r'$\theta=$%1.0f'
    ax4.clabel(CS, CS.levels, inline=False,fontsize=10, fmt=fmt)

    tb = ax4.pcolormesh(x_ax, cross.pressure, cross.turbulent_kinetic_energy_ml, cmap="Reds", shading='nearest', zorder=1)#,norm=norm)
    
    PBLH = ax4.plot(x[-1, :], cross.atmosphere_boundary_layer_thickness, linewidth=5, color="k")
    ax4.text(78, 2150, 'Boundary Layer Height',fontsize=15, rotation=10)

    lvl = [0]
    pc = ax4.contour(x_ax, cross.pressure, cross.ri, levels=lvl, shading='nearest', zorder=1, colors=["green"], linewidths=5,)#,norm=norm) #colors="white
    ax4.text(76, 750, 'Ri=0',fontsize=15, rotation=10, color="green")

    plt.gca().invert_xaxis()
    cbar_tke= plt.colorbar(tb)
    cbar_tke.set_label(r'Turbulent Kinetic Energy $[m^2⋅s^{-2}]$',fontsize=15)
    cbar_tke.ax.tick_params(labelsize=15)
    ax4.set_ylabel("Height [m]")
    ax4.set_xlabel("Latitudes")

    ax4.add_patch(Rectangle((79.5, -650), 4.5, 300, color='#70B6F4', alpha=0.8, clip_on=False)) #blue
    ax4.add_patch(Rectangle((66.9, -650), 12.6, 300, color='#FD7173', alpha=0.8, clip_on=False)) #red

    plt.savefig("TKE4.png",
                bbox_inches="tight",
                dpi=300,
            )
##################################################################################
    #fig5, ax5 = plt.subplots(figsize=(14, 6))



    #plt.show()
def Vertical_cross_section(kwargs, point_time_file=None):
    """
    python plots/Vertical_cross_section.py --datetime 2020031000 --model AromeArctic --domain_name Svalbard --steps 6
    """
    #param = ["air_temperature_ml","surface_geopotential","surface_air_pressure","specific_humidity_ml",
    #         "air_pressure_at_sea_level","x_wind_ml","y_wind_ml", "SST", "turbulent_kinetic_energy_ml"]
    param = ["SIC","toa_outgoing_longwave_flux","air_pressure_at_sea_level","SST","cloud_area_fraction_ml", "air_temperature_ml","surface_geopotential","surface_air_pressure","specific_humidity_ml","turbulent_kinetic_energy_ml","atmosphere_boundary_layer_thickness", "y_wind_ml", "x_wind_ml"]
    #param.extend(OLR_map.param) #get more parameters that was used in the OLR.map 
    param = list(set(param))
    kwargs.m_level = np.arange(20, 64, 1)

    print(kwargs.point_name)
    line_endpoints = ("ice_cross", "end_cross") #or kwargs.point_name 
    line = tuple(point_name2point_lonlat(line_endpoints))  #has to be on the form ((lon,lat),(lon,lat)), so a tuple. Would be nice to include many poinst here..?
    kwargs.point_name = None; kwargs.point_lonlat = None   #We want to retrieve the entire model
    print(line)
    join_pt = {}
    nbre = 100 # horsiontal resolutioN of crosssection: number of points between start and end points
    move_data_with_flow = False
    int2z=True
    int2p=False
    heights_4interpolation = np.arange(0, 5000, 100)  # height/pressure interval that the field is goid to be interpolated to.
    pressure_4interpolation = np.arange(500, 1000, 10)[::-1]  # height/pressure interval that the field is goid to be interpolated to.

    if move_data_with_flow:
        kwargs,join_pt = move_cross_with_the_flow_or_observation(kwargs, line, nbre, speed=30, tbra=1 )
   
    #########follow airmass###########
    #kwargs = pre_defined_points_retrieved(line, kwargs): "allwind_A202003100600.nc"    "SSC_23.nc       # If u want to retrieve only the points NB: Avoid if u have many points > 20
    dmet, data_domain, bad_param = checkget_data_handler(all_param=param,save=False,read_from_saved="paper1cross_line.nc",#"TKE_A2020031000+07.nc",#"wind_A2020031000+6.nc", #"buffer_newM.nc", 
                                                        model=kwargs.model,  date=kwargs.datetime,
                                                        step=kwargs.steps,   m_level=kwargs.m_level,
                                                        point_name=kwargs.point_name, 
                                                        point_lonlat=kwargs.point_lonlat,
                                                        num_point=kwargs.num_point,
                                                        domain_name = kwargs.domain_name[0])

   
    dmet.pressure = ml2pl(ap=dmet.ap, b=dmet.b, surface_air_pressure= dmet.surface_air_pressure, dmet=dmet)
    dmet.z = pl2alt(ap= dmet.ap, b=dmet.b, surface_air_pressure= dmet.surface_air_pressure, air_temperature_ml= dmet.air_temperature_ml, specific_humidity_ml= dmet.specific_humidity_ml, pressure=  dmet.pressure,surface_geopotential=dmet.surface_geopotential, dmet=dmet)
    dmet.pt = potential_temperatur(dmet.air_temperature_ml, dmet.pressure)
    print(dmet.model)
    dmet.ri = richardson( dmet.pt,dmet.z,dmet.x_wind_ml, dmet.y_wind_ml )

    


    param = param + ["pt", "ri"]
    param_4cross = param

    required_vars4cross = {"lat": dmet.latitude,"rlat": dmet.y,"lon": dmet.longitude, "rlon": dmet.x, 
                            "z":dmet.z, "p":dmet.pressure/100. }
    
    vars4cross = dict(zip(param_4cross, [getattr(dmet, p) for p in param]))
    vars4cross = {**required_vars4cross, **vars4cross}
   
    if len(join_pt) == 0:
        for k in vars4cross.keys():
            if len(np.shape(vars4cross[k])) >2:
                vars4cross[k] = vars4cross[k][0,...].squeeze() 
                print(np.shape(vars4cross[k]))
    if int2z:
        intepolate_height = heights_4interpolation
        flip=True
    elif int2p:
        intepolate_height = pressure_4interpolation
        flip=False
    else:
        intepolate_height = np.arange(0,65)
        flip=False

    cross = CrossSection(
                    vars4cross,
                    coo = line,
                    nbre = nbre,
                    pressure = intepolate_height, #heights  #Need to be the same as height for data or what u want to interp on
                    version="rotated",#"rotated",
                    pollon=None,
                    pollat=None,
                    flip=flip,
                    int2z=int2z,
                    int2p=int2p,
                    parallels=dmet.standard_parallel_projection_lambert, 
                    center_longitude=dmet.longitude_of_central_meridian_projection_lambert, 
                    center_latitude = dmet.latitude_of_projection_origin_projection_lambert,
                    join_pt = join_pt
                )  # polgam=180,

    cross.longitude=cross.lon #6
    cross.latitude=cross.lat 
    print(cross.latitude)
    #exit(1)
    #print(cross.pressure)
    #exit(1)
    #cross.pt = potential_temperatur(cross.air_temperature_ml, cross.pressure*100)
    #print(cross.pt)
    for k in vars(cross).keys(): #vars4cross.keys():
        ddd = getattr(cross,k)
        if len(np.shape(ddd)) == 3:
            print("4d")
            print(k)
            p_array = np.zeros(np.shape(ddd)[1:])
            for p, t in join_pt.items():
                p_array[:,p] = getattr(cross,k)[t,:,p]
                print(np.shape(p_array))
            setattr(cross,k, p_array)
    
    plot_Vertical_cross_section(cross)
    plot_map(cross, dmet, data_domain, kwargs)



if __name__ == "__main__":
    args = default_arguments()
    chunck_func_call(func=Vertical_cross_section,chunktype= args.chunktype, chunk=args.chunks,  kwargs=args)
    gc.collect()


