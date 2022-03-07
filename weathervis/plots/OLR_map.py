# %%
#python OLR_sat.py --datetime 2022030400 --steps 0 3 --model AromeArctic --domain_name Svalbard North_Norway --use_latest 0
#
from weathervis.config import *
from weathervis.utils import *
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

warnings.filterwarnings("ignore", category=UserWarning) # suppress matplotlib warning

def plot_OLR(datetime, data_domain, dmet, steps=[0,2], coast_details="auto", model= None, domain_name = None, domain_lonlat = None, legend=False, info = False, save = True,grid=True, url = None):
  eval(f"data_domain.{domain_name}()")  # get domain info
  ## CALCULATE AND INITIALISE ####################
  scale = data_domain.scale  #scale is larger for smaller domains in order to scale it up.
  dmet.air_pressure_at_sea_level /= 100
  MSLP = filter_values_over_mountain(dmet.surface_geopotential[:], dmet.air_pressure_at_sea_level[:])
  ########################################
  # PLOTTING ROUTNE ######################
  crs = default_map_projection(dmet) #change if u want another projection
  fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9), subplot_kw={'projection': crs})
  itim = 0
  for leadtime in np.array(steps):
      print('Plotting {0} + {1:02d} UTC'.format(datetime, leadtime))
      ax1 = default_mslp_contour(dmet.x, dmet.y, MSLP[itim, 0, :, :], ax1, scale=scale)
      x,y = np.meshgrid(dmet.x, dmet.y)
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
      ax1.pcolormesh(x, y, data[ :, :], vmin=-230,vmax=-110, cmap=plt.cm.Greys_r)
      ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'),edgecolor="brown", linewidth=0.5)  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
      ax1.text(0, 1, "{0}_OLR_{1}+{2:02d}".format(model, datetime, leadtime), ha='left', va='bottom', transform=ax1.transAxes,color='dimgrey')
      if grid:
          nicegrid(ax=ax1,color="orange")

      print(data_domain.lonlat)  # [15.8, 16.4, 69.2, 69.4]
      if domain_name != model and data_domain != None:
          ax1.set_extent(data_domain.lonlat)

      make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(datetime))
      file_path = "{0}/{1}_{2}_{3}_{4}+{5:02d}.png".format(make_modelrun_folder, model, domain_name, "OLR", datetime,leadtime)
      print(f"filename: {file_path}")
      fig1.savefig(file_path, bbox_inches="tight", dpi=200)
      ax1.cla()
      itim += 1
  plt.close(fig1)
  plt.close("all")
  del dmet
  del data_domain
  del crs
  gc.collect()

def OLR(datetime, steps, model, domain_name, domain_lonlat, legend, info, grid, url, point_lonlat, use_latest,
        delta_index, coast_details):
  param = ["air_pressure_at_sea_level","surface_geopotential", "toa_outgoing_longwave_flux"]

  p_level = None
  # Todo: In the future make this part of the entire checkget_datahandler or someother hidden solution
  print("bef subdom")
  domains_with_subdomains = find_subdomains(domain_name=domain_name, datetime=datetime, model=model,
                                            domain_lonlat=domain_lonlat,
                                            point_lonlat=point_lonlat, use_latest=use_latest, delta_index=delta_index,
                                            url=url)
  print(domains_with_subdomains)
  print(domains_with_subdomains.index.values)
  print("subdom")
  for domain_name in domains_with_subdomains.index.values:
    dmet, data_domain, bad_param = checkget_data_handler(model=model, step=steps, date=datetime,
                                                         domain_name=domain_name, all_param=param)
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