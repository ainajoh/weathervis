# %%
# python Z500_VEL.py --datetime 2020091000 --steps 0 1 --model MEPS --domain_name West_Norway
from weathervis.checkget_data_handler import *

from weathervis.config import *
from weathervis.utils import *
from weathervis.check_data import *
from weathervis.domain import *
from weathervis.get_data import *
from weathervis.calculation import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
import pandas as pd

def plot_OLR(datetime, data_domain, dmet, steps=[0,2], coast_details="auto", model= None, domain_name = None, domain_lonlat = None, legend=False, info = False, save = True,grid=True, url = None):
  eval(f"data_domain.{domain_name}()")  # get domain info
  ## CALCULATE AND INITIALISE ####################
  scale = data_domain.scale  #scale is larger for smaller domains in order to scale it up.
  dmet.air_pressure_at_sea_level /= 100
  plev = 0 #velocity presuure level at first request
  MSLP = filter_values_over_mountain(dmet.surface_geopotential[:], dmet.air_pressure_at_sea_level[:])

  ########################################
  # PLOTTING ROUTNE ######################
  crs = default_map_projection(dmet) #change if u want another projection
  fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9), subplot_kw={'projection': crs})

  itim = 0
  for leadtime in np.array(steps):
    print('Plotting {0} + {1:02d} UTC'.format(datetime, leadtime))
    ax1 = default_mslp_contour(dmet.x, dmet.y, MSLP[itim, 0, :, :], ax1, scale=scale)

    #ttt = tim
    #tidx = tim - np.min(steps)
    #ZS = dmap_meps.surface_geopotential[tidx, 0, :, :]
    #MSLP = np.where(ZS < 3000, dmap_meps.air_pressure_at_sea_level[tidx, 0, :, :], np.NaN).squeeze()
    #ax = plt.subplot(projection=crs)

    #It is a bug in pcolormesh. supposedly newest is correct, but not older versions. Invalid corner values set to nan
    #https://github.com/matplotlib/basemap/issues/470
    x,y = np.meshgrid(dmet.x, dmet.y)
    #dlon,dlat=  np.meshgrid(dmap_meps.longitude, dmap_meps.latitude)

    nx, ny = x.shape
    mask = (
            (x[:-1, :-1] > 1e20) |
            (x[1:, :-1] > 1e20) |
            (x[:-1, 1:] > 1e20) |
            (x[1:, 1:] > 1e20) |
            (x[:-1, :-1] > 1e20) |
            (x[1:, :-1] > 1e20) |
            (x[:-1, 1:] > 1e20) |
            (x[1:, 1:] > 1e20))
    data =  dmet.toa_outgoing_longwave_flux[itim, 0,:nx - 1, :ny - 1].copy()
    data[mask] = np.nan
    #ax.pcolormesh(x, y, data[ :, :])#, cmap=plt.cm.Greys_r)

    ax1.pcolormesh(x, y, data[ :, :], vmin=-230,vmax=-110, cmap=plt.cm.Greys_r)


    ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'),edgecolor="brown", linewidth=0.5)  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).


    make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(dt))
    ax1.text(0, 1, "{0}_OLR_{1}+{2:02d}".format(model, datetime, leadtime), ha='left', va='bottom', transform=ax1.transAxes,
             color='dimgrey')
    legend=False
    if legend:
      pressure_dim = list(
      filter(re.compile(f'press*').match, dmet.__dict__.keys()))  # need to find the correcvt pressure name
      llg = {'W_over': {'color': 'red', 'linestyle': None,
                    'legend': f"W [m s-1]>0.07 m/s at {dmet.__dict__[pressure_dim[0]][plev]:.0f} hPa"},
            'W_under': {'color': 'blue', 'linestyle': 'dashed',
                   'legend': f"W [m s-1]<0.07 m/s at {dmet.__dict__[pressure_dim[0]][plev]:.0f} hPa"},
            'MSLP': {'color': 'gray', 'linestyle': None, 'legend': "MSLP [hPa]"}}
      nice_legend(llg, ax1)


    if grid:
      nicegrid(ax=ax1,color="orange")

    fig1.savefig(make_modelrun_folder + "/{0}_{1}_OLR_sat_{2}+{3:02d}.png".format(model, domain_name, dt, ttt), bbox_inches="tight", dpi=200)

    ax1.cla()
    fig1.clf()
    plt.close(fig1)

  ax1.cla()
  plt.clf()
plt.close("all")


def OLR(datetime, steps, model, domain_name, domain_lonlat, legend, info, grid, url, point_lonlat, use_latest,
        delta_index, coast_details):
  param = ["toa_outgoing_longwave_flux", "air_pressure_at_sea_level", "surface_geopotential"]
  p_level = None
  # Todo: In the future make this part of the entire checkget_datahandler or someother hidden solution
  domains_with_subdomains = find_subdomains(domain_name=domain_name, datetime=datetime, model=model,
                                            domain_lonlat=domain_lonlat,
                                            point_lonlat=point_lonlat, use_latest=use_latest, delta_index=delta_index,
                                            url=url)
  print(domains_with_subdomains)
  print(domains_with_subdomains.index.values)
  for domain_name in domains_with_subdomains.index.values:
    dmet, data_domain, bad_param = checkget_data_handler(p_level=p_level, model=model, step=steps, date=datetime,
                                                         domain_name=domain_name, all_param=param)
    #dmap_meps, data_domain, bad_param = checkget_data_handler( model=model, all_param=param,step=steps, date=dt, domain_name=domain_name)

    subdom = domains_with_subdomains.loc[domain_name]
    ii = subdom[subdom == True]
    subdom_list = list(ii.index.values)
    if subdom_list:
      for sub in subdom_list:
        plot_OLR(datetime=datetime, steps=steps, model=model, domain_name=sub, data_domain=data_domain,
                 domain_lonlat=domain_lonlat, legend=legend, info=info, grid=grid, url=url,
                 dmet=dmet, coast_details=coast_details)

if __name__ == "__main__":
    args = default_arguments()
    OLR(datetime=args.datetime, steps=args.steps, model=args.model, domain_name=args.domain_name,
        domain_lonlat=args.domain_lonlat, legend=args.legend, info=args.info, grid=args.grid, url=args.url,
        point_lonlat =args.point_lonlat,use_latest=args.use_latest,delta_index=args.delta_index, coast_details=args.coast_details)
    gc.collect()