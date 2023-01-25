# %%
#python Vertical_cross_section_withcmet.py --datetime 2022032500 --model AromeArctic#
from weathervis.config import *
from weathervis.utils import filter_values_over_mountain, default_map_projection, default_mslp_contour, plot_by_subdomains
import cartopy.crs as ccrs
from weathervis.domain import *  # require netcdf4
from weathervis.check_data import *
from weathervis.get_data import *
from weathervis.calculation import *
import matplotlib.pyplot as plt
import warnings
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable ##__N
from weathervis.checkget_data_handler import *
import gc
import matplotlib.colors
import numpy as np
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=UserWarning) # suppress matplotlib warning

def map_plot(data_domain, model_data, adj_obs_data, raw_obs_data, domain_name):
    eval(f"data_domain.{domain_name}()")  # get domain info
    # MAP PLOTTING ROUTNE ######################
    crs = default_map_projection(model_data)
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9), subplot_kw={'projection': crs})
    ax1.scatter(model_data.longitude, model_data.latitude,transform=ccrs.PlateCarree(), marker = "o", color="blue",  edgecolor="blue", zorder=3 )
    ax1.scatter(raw_obs_data.lon, raw_obs_data.lat, transform=ccrs.PlateCarree(), marker= "x", color="red", zorder=2)
    #ax1.scatter(raw_obs_data["Lon[d]"], raw_obs_data["Lat[d]"], transform=ccrs.PlateCarree(), marker= "x", color="green", zorder=3)

    ax1.add_feature(cfeature.GSHHSFeature(scale="auto"))
    ax1.set_extent(data_domain.lonlat)
    plt.show()

def VC_plot(model_data, adj_obs_data, raw_obs_data):
    #Find normalized values for plotting on the same scale
    num_p = len(model_data.time) #number of points retrieved
    mi = np.min((raw_obs_data['T[K]'][:num_p].values.min(), model_data.air_temperature_ml.min()))
    ma = np.max((raw_obs_data['T[K]'][:num_p].values.max(), model_data.air_temperature_ml.max()))
    norm = matplotlib.colors.Normalize(vmin=mi, vmax=ma)
    #norm = matplotlib.colors.Normalize(vmin=200, vmax=350)

    colormap = "coolwarm"       #cividis_r
    obs_datetime = adj_obs_data.index[:num_p]
    lx, tx = np.meshgrid(model_data.m_level, obs_datetime)
    fig, ax = plt.subplots(figsize=(12, 3))
    y_axis = "pressure"  #if u want axis to be in meters
    
    if y_axis == "altitude":
        cf = ax.pcolormesh(tx, model_data.altitude, model_data.air_temperature_ml, cmap=colormap, shading='nearest', zorder=1,norm=norm)
    else:
        cf = ax.pcolormesh(tx, model_data.pressure/100, model_data.air_temperature_ml, cmap=colormap, shading='nearest', zorder=1,norm=norm)

    fig.colorbar(cf, ax=ax)

    # CMET Zp[m]
    pres_cmet = adj_obs_data['P[Pa]'][:num_p].values / 100.
    Z_cmet = adj_obs_data['Zp[m]'][:num_p].values
    T_cmet = adj_obs_data['T[K]'][:num_p].values
    RH_cmet = adj_obs_data['RH[%]'][:num_p].values
    if y_axis == "altitude":
        cmetf = ax.scatter(obs_datetime, Z_cmet, c=T_cmet, marker="8", s=250, norm=norm, cmap=colormap, edgecolor="k",
                          zorder=3)
    else:
        cmetf = ax.scatter(obs_datetime, pres_cmet, c=T_cmet, marker="8", s=250, norm=norm, cmap=colormap, edgecolor="k",
                          zorder=3)
    
    raw_obs_data = raw_obs_data[raw_obs_data.index <= adj_obs_data.index[:num_p].values[-1]]
    pres_cmet = raw_obs_data['P[Pa]'].values / 100  # [:num_p].values / 100.
    Z_cmet = raw_obs_data['Zp[m]'].values  # [:num_p].values / 100.
    T_cmet = raw_obs_data['T[K]'].values# [:num_p].values
    RH_cmet = raw_obs_data['RH[%]'].values
    obs_datetime = raw_obs_data.index  # [:num_p]
    if y_axis == "altitude":
        cmetf = ax.scatter(obs_datetime, Z_cmet, c=T_cmet, s=100, norm=norm, cmap=colormap, zorder=2)  # , edgecolor="k")
        cmetf = ax.scatter(obs_datetime, Z_cmet, color="white", zorder=3, marker=".", s= 0.1)  # , edgecolor="k")
    else:
        cmetf = ax.scatter(obs_datetime, pres_cmet, c=T_cmet, s=100, norm=norm, cmap=colormap, zorder=2)  # , edgecolor="k")
        cmetf = ax.scatter(obs_datetime, pres_cmet, color="white", zorder=3, marker=".", s= 0.1)  # , edgecolor="k")
    
    if y_axis != "altitude":
        ax.invert_yaxis()


    # plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    plt.gcf().autofmt_xdate()
    plt.show()

def VC_getcmet(filename,julianday_adjustment, possible_lonlatname):
    #####################################
    # READ AND ADJUST CMET BALOON DATA:
    #####################################
    raw_obs_data = pd.read_csv(filename, sep=r'\s*[,]\s*',  header=0, index_col=0) #sep="\s*[,]\s*"
    raw_obs_data.columns = raw_obs_data.columns.str.replace(' ', '')  # removes any aditional spaces

    #merge all difference lonlat measurment until one. 
    ############################################################
    for lonlatnames in possible_lonlatname:
        if lonlatnames[0] in raw_obs_data.columns:
            print(lonlatnames)
            raw_obs_data["lon"] = raw_obs_data[lonlatnames[0]]
            raw_obs_data["lat"] = raw_obs_data[lonlatnames[1]]
            break
    for lonlatnames in possible_lonlatname:
        if lonlatnames[0] in raw_obs_data.columns:
            
            print(lonlatnames)
            raw_obs_data["lon"].fillna(raw_obs_data[lonlatnames[0]], inplace=True)
            raw_obs_data["lat"].fillna(raw_obs_data[lonlatnames[1]], inplace=True)
    raw_obs_data = raw_obs_data[raw_obs_data["lon"].notna()].reset_index()  # removes all missing lonlat positions
    ##########################################################

    #Fix error with actual datetime by making a new datetime column using adjustments set in julianday_adjustment
    raw_obs_data["datetime"] = pd.to_datetime("2022", format='%Y') + \
                       pd.to_timedelta(raw_obs_data.JulianDay + julianday_adjustment,
                                       unit='d')  # convert Julianday to datetime- 7 days offset in dataset

    raw_obs_data.index = raw_obs_data["datetime"]  # makes new index a datetime index
    # We pick the time closest to an hour ( in order to compare with the hour by hour dataset of the model). 
    idx = (raw_obs_data["datetime"].dt.round("H").sub(raw_obs_data["datetime"]).abs().groupby(raw_obs_data["datetime"].dt.round("H"),
                                                                              sort=False).idxmin())
    adj_obs_data = raw_obs_data.loc[idx]
    adj_obs_data = adj_obs_data[adj_obs_data["lon"].notna()]
  
    return adj_obs_data, raw_obs_data

def VC_get_model_data(obs_data, datetime, model):

    #####################################
    # READ AND ADJUST AROME ARCTIC MODEL DATA
    #####################################

    #Settings:
    #####################################
    if datetime==None: #if model datetime is not given when running this script then:
        #Choose the initial model run hour based on when the obervations starts
        #  (the model gives a forecast every 6 hours, and we want to choose the latest forecast from the observation date)
        firsthour = int(obs_data["datetime"].dt.strftime("%H").values[0])  #first observation hour
        hour = str(int(firsthour)-int(firsthour)%6).zfill(2)               #the hour closest to the latest modelrun
        datetime = obs_data["datetime"].dt.strftime("%Y%m%d").values[0] + hour  # merging date and hour for the modelrun we want

    modeldattime = pd.to_datetime(datetime, format="%Y%m%d%H") # The chosen Modelrun datetime

    param = ["air_pressure_at_sea_level","surface_air_pressure","surface_geopotential", "air_temperature_ml", "specific_humidity_ml"] #parameters we want
    m_level =  np.arange(30, 65, 1) #model levels we want
    ######################################


    dmet_old= False      #Just a counter for the for loop in order to glue all the model data from different times together
    num_p = len(adj_obs_data.index)
    steps=[]
    datetimes=[datetime]
    #Find the time and position of the observation
    use_latest_modelrun=True #explenation bellow
    # We can either choose to use just one initial modelrun with different leadtime to match the observation -->use_latest_modelrun=False
    #  Or we can choose to use the closest modelrun to the observation and fill in with leadtime  -->use_latest_modelrun=True
    
    #For loop that retrieves data and update the settings to get the modeldata closest to the observation
    for index, row in obs_data.iloc[:num_p].iterrows():  #Go through all the adjusted observation dataset
        point_lonlat = [row["lon"], row["lat"]]   #Location of observation
        if use_latest_modelrun:  #updates modelrun datetime with closest date to the observation
            h = index.hour-index.hour%6
            new_initial_modelrun = index.strftime('%Y%m%d') + str(h).zfill(2)
            datetime=new_initial_modelrun
            modeldattime = pd.to_datetime(datetime, format="%Y%m%d%H") # The chosen Modelrun datetime
            datetimes = np.append(datetimes, datetime)  #save the modelrundatetime in an array
 
        delta_t = index-modeldattime              #Difference beween the modelrun datetime and observation datetime
        step = delta_t.round('60min').components.hours            # The lead time hour closest to our observation
        steps = np.append(steps, step)            #save the steps in an array
        
        #retrieve data
        ######################################################
        dmet, data_domain, bad_param = checkget_data_handler(all_param=param, model=model,  date=datetime,
                                                         step=step,m_level=m_level,point_lonlat=point_lonlat,num_point=1,
                                                         domain_name=None )
        ######################################################
        
        xy_shape = np.shape(dmet.air_temperature_ml)[-2:]  # To save the shape for later adjustments
        
        #glue togather all the retrievals done in the for loop
        ######################################################
        if dmet_old: #if we have an older retrieval then do: 
            for prm in dmet.param: #for every parameter retrieve
                if prm != "pressure": #as long as it is not the pressure param: we glue them together
                    setattr(dmet, prm,  np.array(np.append( np.array([getattr(dmet, prm)]), np.array([getattr(dmet_old, prm)])))   )
        dmet_old = deepcopy(dmet)  #update old retieval
        #####################################################
    #We have to adjust the shape as the "glued" data lost the initial shape
    setattr(dmet, "ap", getattr(dmet, "ap").reshape( num_p, len(m_level)))
    setattr(dmet, "b", getattr(dmet, "b").reshape( num_p, len(m_level)))
    setattr(dmet, "air_temperature_ml", getattr(dmet, "air_temperature_ml").reshape( num_p, len(m_level), xy_shape[0], xy_shape[1] ))
    setattr(dmet, "specific_humidity_ml", getattr(dmet, "specific_humidity_ml").reshape( num_p, len(m_level), xy_shape[0], xy_shape[1] ))
    setattr(dmet, "surface_air_pressure", getattr(dmet, "surface_air_pressure").reshape( num_p, xy_shape[0], xy_shape[1] ))
    setattr(dmet, "air_pressure_at_sea_level", getattr(dmet, "air_pressure_at_sea_level").reshape( num_p, xy_shape[0], xy_shape[1] ))
    setattr(dmet, "surface_geopotential", getattr(dmet, "surface_geopotential").reshape( num_p, xy_shape[0], xy_shape[1] ))

    #Calculate pressure from modellevels
    pressurefromml = ml2pl( dmet.ap, dmet.b, dmet.surface_air_pressure)
    setattr(dmet, "pressure", pressurefromml)

    #Calculate Altitude from modellevels
    altitudefromml = ml2alt_gl( dmet.air_temperature_ml, dmet.specific_humidity_ml, dmet.ap, dmet.b, dmet.surface_air_pressure,  inputlevel="half", returnlevel="full") #ml2pl( dmet.ap, dmet.b, dmet.surface_air_pressure)
    #pl2alt_half2full_gl( air_temperature_ml, specific_humidity_ml, p)
    setattr(dmet, "altitude", altitudefromml)
    #setattr(dmet, "altitude", np.tile(getattr(dmet, "altitude"), (10,1)) )

    #Depending on num_point(how many points closest to the observation) in the retrieval
    #  we want to either calculate the mean of these points or just get one point in order to plot
    for prm in param: 
        if xy_shape[0]==1 and xy_shape[1]==1: 
            setattr(dmet, prm, getattr(dmet, prm).squeeze(axis=(-2,-1)))
        else: 
            setattr(dmet, prm, getattr(dmet, prm).mean(axis=(-2,-1)) )
    for prm in ["pressure","altitude"]:
        if xy_shape[0]==1 and xy_shape[1]==1: 
            setattr(dmet, prm, getattr(dmet, prm).squeeze(axis=(-2,-1))) #or consider squeeze
        else: 
            setattr(dmet, prm, getattr(dmet, prm).mean(axis=(-2,-1)) )
    dmet.steps=steps
    dmet.datetimes=datetimes
    
    return dmet, data_domain

if __name__ == "__main__":
    ##############################################################################################
    # step 1: Makesure your cmet files is in weathervis/data/islas2022/ with names of as used bellow
    # (cmet1.csv, cmet2.csv et)
    # step 2: run from terminal:
    #        $ python Vertical_cross_section_withcmet.py --cmet cmet2
    ##############################################################################################
    import argparse
    parser = argparse.ArgumentParser()

    #model settings
    parser.add_argument("--model", default="AromeArctic", help="MEPS or AromeArctic")
    parser.add_argument("--domain_name", default=["Svalbard"], nargs="+", type= none_or_str)
    parser.add_argument("--datetime", help="YYYYMMDDHH for modelrun", default=None,  type=str)
    parser.add_argument("--cmet", help="cmet1, cmet2.. etc", default="cmet1",  type=str)

    args = parser.parse_args()

    #cmet settings
    filename=f"{package_path}/data/islas2022/{args.cmet}.csv"   #write the path to your cmet data files
    #NB adjust lat and lon name in priority order or only one name.
    possible_lonlatname= np.array([["Lon[d]","Lat[d]"], ["Lon[deg]","Lat[deg]"],["LonMsg[d]","LatMsg[d]"]])
    error_julianday = {"cmet1":6, "cmet2":7, "cmet3":-1, "cmet4":12, "cmet5":17, "cmet6":17}
    #NB! DATE NEED ADJUSTMENTS
    for nom in error_julianday.keys():
        if nom in filename:
            julianday_adjustment= error_julianday[nom]
            break
    print(julianday_adjustment)


    ##############################################################################################
    adj_obs_data, raw_obs_data  = VC_getcmet(filename=filename,
                                             julianday_adjustment=julianday_adjustment,
                                             possible_lonlatname=possible_lonlatname)
    model_data, data_domain = VC_get_model_data(adj_obs_data, args.datetime, args.model)
    VC_plot(model_data=model_data, adj_obs_data=adj_obs_data, raw_obs_data=raw_obs_data )
    map_plot(model_data=model_data, adj_obs_data=adj_obs_data, raw_obs_data=raw_obs_data, domain_name=args.domain_name[0], data_domain=data_domain)