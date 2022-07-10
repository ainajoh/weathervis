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
    mi = np.min((raw_obs_data['RH[%]'][:num_p].values.min(), model_data.relative_humidity_pl.min()))
    ma = np.max((raw_obs_data['RH[%]'][:num_p].values.max(), model_data.relative_humidity_pl.max()))
    norm = matplotlib.colors.Normalize(vmin=mi, vmax=ma)
    norm = matplotlib.colors.Normalize(vmin=10, vmax=100)

    colormap = "cividis_r"
    #norm = matplotlib.colors.Normalize(vmin=0, vmax=100) for a fixed value between plots
    obs_datetime = adj_obs_data.index[:num_p]
    lx, tx = np.meshgrid(model_data.pressure, obs_datetime)
    fig, ax = plt.subplots(figsize=(12, 3))
    pp = model_data.pressure
    cf = ax.pcolormesh(tx, lx, model_data.relative_humidity_pl * 100, norm=norm, cmap=colormap, shading='nearest', zorder=1)
    fig.colorbar(cf, ax=ax)

    # CMET
    print("CMET")
    pres_cmet = adj_obs_data['P[Pa]'][:num_p].values / 100.
    rh_cmet = adj_obs_data['RH[%]'][:num_p].values
    cmetf = ax.scatter(obs_datetime, pres_cmet, c=rh_cmet, marker="8", s=250, norm=norm, cmap=colormap, edgecolor="k",
                       zorder=3)
    raw_obs_data = raw_obs_data[raw_obs_data.index <= adj_obs_data.index[:num_p].values[-1]]
    pres_cmet = raw_obs_data['P[Pa]'].values / 100  # [:num_p].values / 100.
    rh_cmet = raw_obs_data['RH[%]'].values  # [:num_p].values
    obs_datetime = raw_obs_data.index  # [:num_p]
    cmetf = ax.scatter(obs_datetime, pres_cmet, c=rh_cmet, s=100, norm=norm, cmap=colormap, zorder=2)  # , edgecolor="k")
    cmetf = ax.scatter(obs_datetime, pres_cmet, color="white", zorder=3, marker=".", s= 0.1)  # , edgecolor="k")

    # plt.gca()
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    plt.gcf().autofmt_xdate()
    plt.show()


def VC_getcmet(filename,julianday_adjustment, possible_lonlatname):
    #lon_name = lonname
    #lat_name = latname
    #####################################
    # READ AND ADJUST CMET BALOON DATA:
    #####################################
    raw_obs_data = pd.read_csv(filename, sep=r'\s*[,]\s*',  header=0, index_col=0) #sep="\s*[,]\s*"
    raw_obs_data.columns = raw_obs_data.columns.str.replace(' ', '')  # removes any aditional spaces

    #raw_obs_data["lon"] = raw_obs_data[lon_name]
    #raw_obs_data["lat"] = raw_obs_data[lat_name]
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

    # cmet = cmet[cmet['LonMsg[d]'].notna()].reset_index() #removes all missing lonlat positions
    raw_obs_data = raw_obs_data[raw_obs_data["lon"].notna()].reset_index()  # removes all missing lonlat positions
    print(raw_obs_data)
    raw_obs_data["datetime"] = pd.to_datetime("2022", format='%Y') + \
                       pd.to_timedelta(raw_obs_data.JulianDay + julianday_adjustment,
                                       unit='d')  # convert Julianday to datetime- 7 days offset in dataset

    raw_obs_data.index = raw_obs_data["datetime"]  # makes new index a datetime index
    # We average the results - nb does not work
    # obs_data = cmet[['RH[%]', 'P[Pa]','LonMsg[d]','LatMsg[d]','Lon[d]','Lat[d]']].resample("2min").mean().ffill()
    # obs_data = obs_data[ obs_data.index.minute== 0]
    # We pick the time closest to an hour
    idx = (raw_obs_data["datetime"].dt.round("H").sub(raw_obs_data["datetime"]).abs().groupby(raw_obs_data["datetime"].dt.round("H"),
                                                                              sort=False).idxmin())
    adj_obs_data = raw_obs_data.loc[idx]
    adj_obs_data = adj_obs_data[adj_obs_data["lon"].notna()]

    return adj_obs_data, raw_obs_data


def VC_get_model_data(obs_data, datetime, model):

    #####################################
    # READ AND ADJUST AROME ARCTIC MODEL DATA
    #####################################
    if datetime==None:
        firsthour = int(obs_data["datetime"].dt.strftime("%H").values[0])
        if firsthour >= 18:
            hour = "18"
        elif firsthour>= 12:
            hour = "12"
        elif firsthour >= 6:
            hour = "06"
        else:
            hour = "00"
        dd = obs_data["datetime"].dt.strftime("%Y%m%d").values[0]
        datetime = dd + hour
        print(datetime)
    param = ["air_pressure_at_sea_level", "surface_geopotential", "relative_humidity_pl"] #parameters we want
    p_level =  [500, 700, 800, 850, 925, 1000] # pressure levels we want
    # Possible pressure levels in model [50, 100 , 150, 200, 250, 300, 400, 500, 700, 800, 850, 925, 1000]
    modeldattime = pd.to_datetime(datetime, format="%Y%m%d%H")
    dmet_old= False
    num_p = len(adj_obs_data.index)#-len(adj_obs_data.index) + 3
    steps=[]
    for index, row in obs_data.iloc[:num_p].iterrows():
        point_lonlat = [row["lon"], row["lat"]]
        delta_t = index-modeldattime
        step = delta_t.components.hours
        steps = np.append(steps, step)
        dmet, data_domain, bad_param = checkget_data_handler(all_param=param, model=model,  date=datetime,
                                                         step=step,p_level=p_level,point_lonlat=point_lonlat,
                                                         domain_name=None )
        if dmet_old:
            for prm in dmet.param:
                if prm != "pressure":
                    setattr(dmet, prm,  np.array(np.append( np.array([getattr(dmet, prm)]), np.array([getattr(dmet_old, prm)])))   )
        dmet_old = deepcopy(dmet)
    setattr(dmet, "relative_humidity_pl", getattr(dmet, "relative_humidity_pl").reshape(num_p, len(dmet.pressure)))
    dmet.steps=steps
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
    parser.add_argument("--model", default="AromeArctic", help="MEPS or AromeArctic")
    parser.add_argument("--domain_name", default=["Svalbard"], nargs="+", type= none_or_str)
    parser.add_argument("--datetime", help="YYYYMMDDHH for modelrun", default=None,  type=str)
    parser.add_argument("--cmet", help="cmet1, cmet2.. etc", default="cmet1",  type=str)

    args = parser.parse_args()


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