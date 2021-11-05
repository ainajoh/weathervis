# %%
#python BLH_map.py --datetime 2020091000 --steps 0 1 --model MEPS --domain_name West_Norway
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

def plot_BLH(datetime, data_domain, dmet, steps=[0,2], model= None, domain_name = None, domain_lonlat = None, legend=False, info = False, save = True,grid=True, url = None):
  #RETRIEVE
  eval(f"data_domain.{domain_name}()")

  print(model)
  #CALCULATE AND INITIALISE
  scale = data_domain.scale
  dmet.air_pressure_at_sea_level /= 100
  plev = 0 #velocity presuure level at first request
  coast_details='auto' # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
  if scale < 8:
    print(np.shape(dmet.surface_geopotential))
    print(np.shape(dmet.upward_air_velocity_pl))

    W =  filter_values_over_mountain(dmet.surface_geopotential[:], dmet.upward_air_velocity_pl[:])
  else:
    W = dmet.upward_air_velocity_pl[:]#.squeeze()

  MSLP = filter_values_over_mountain(dmet.surface_geopotential[:], dmet.air_pressure_at_sea_level[:])
  BLH = (dmet.atmosphere_boundary_layer_thickness[:, :, :])#.squeeze()

  #PLOT
  crs = default_map_projection(dmet)
  fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9), subplot_kw={'projection': crs})

  itim = 0
  for leadtime in np.array(steps):
      print('Plotting {0} + {1:02d} UTC'.format(datetime,leadtime))
      ax1 = default_mslp_contour(dmet.x, dmet.y, MSLP[itim,0,:,:], ax1, scale=scale)

      # vertical velocity
      C_W = ax1.contour(dmet.x, dmet.y, W[itim,0,:,:], zorder=3, alpha=1.0,
                         levels=np.linspace(0.07,2.0,4*scale), colors="red", linewidths=0.7)
      C_W = ax1.contour(dmet.x, dmet.y, W[itim,0,:,:], zorder=3, alpha=1.0,
                        levels=np.linspace(-2.0,-0.07,4*scale ), colors="blue", linewidths=0.7)
      # boundary layer thickness
      CF_BLH = ax1.contourf(dmet.x, dmet.y, BLH[itim,0,:,:], zorder=1, alpha=0.5,
                        levels=np.arange(50, 5000, 200), linewidths=0.7,label = "BLH", cmap = "summer",extend="both")
      #coastline
      ax1.add_feature(cfeature.GSHHSFeature(scale=coast_details))


      #Done plotting, now adjusting
      ax_cb = adjustable_colorbar_cax(fig1, ax1 )
      fig1.colorbar(CF_BLH, fraction=0.046, pad=0.01,aspect=25,cax = ax_cb, label="Boundary layer thickness [m]", extend="both") ##__N
      print("HHH##############################################################################")
      print(model)
      ax1.text(0, 1, "{0}_BLH_{1}+{2:02d}".format(model, datetime, leadtime), ha='left', va='bottom', transform=ax1.transAxes, color='dimgrey')
      if legend:
        pressure_dim = list(filter(re.compile(f'press*').match, dmet.__dict__.keys())) #need to find the correcvt pressure name
        llg = {'W_over':  {'color': 'red', 'linestyle': None, 'legend': f"W [m s-1]>0.07 m/s at {dmet.__dict__[pressure_dim[0]][plev]:.0f} hPa"},
               'W_under': {'color': 'blue', 'linestyle': 'dashed', 'legend': f"W [m s-1]<0.07 m/s at {dmet.__dict__[pressure_dim[0]][plev]:.0f} hPa"},
               'MSLP': {'color': 'gray', 'linestyle': None, 'legend':  "MSLP [hPa]"}  }
        nice_legend(llg, ax1)
      if grid:
        nicegrid(ax=ax1)

      if domain_name != model and data_domain != None:  #
          ax1.set_extent(data_domain.lonlat)

      make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(datetime))
      file_path="{0}/{1}_{2}_{3}_{4}+{5:02d}.png".format(make_modelrun_folder, model, domain_name, "BLH", datetime, leadtime)

      print( f"filename: {file_path}")
      fig1.savefig(file_path, bbox_inches="tight",dpi=200)
      ax1.cla()
      itim +=1
  plt.close(fig1)
  plt.close("all")
  del dmet
  del data_domain
  del crs
  gc.collect()

def BLH(datetime, steps, model, domain_name, domain_lonlat, legend, info, grid, url,point_lonlat,use_latest,delta_index):

    param = ["air_pressure_at_sea_level", "surface_geopotential", "atmosphere_boundary_layer_thickness",
             "upward_air_velocity_pl"]
    #param = ["upward_air_velocity_pl","surface_geopotential"]

    #domain_name = ["Andenes"]  # , "KingsBay_Z0", "Svalbard_z2", "Svalbard_z1"] #args.domain_name
    #one way of making the process of subdomain faster is handling domain if called once,
    # url dont need to load for everytime we change domain_name
    domains_with_subdomains = find_subdomains(domain_name=domain_name, datetime=datetime, model=model, domain_lonlat=domain_lonlat,
                                point_lonlat=point_lonlat, use_latest=use_latest, delta_index=delta_index, url=url)

    for domain_name in domains_with_subdomains.index.values:
        dmet, data_domain, bad_param = checkget_data_handler(p_level=[850], date=datetime, domain_name=domain_name,
                                                             model=model, step=steps, all_param=param, url=url)

        #print(np.shape(dmet.upward_air_velocity_pl))
        #print(dmet.upward_air_velocity_pl[3,:,:,:])

        #print(np.shape(dmet.air_pressure_at_sea_level))
        #print(np.shape(dmet.surface_geopotential))

        #exit(1)
        subdom = domains_with_subdomains.loc[domain_name]
        ii = subdom[subdom == True]
        subdom_list = list(ii.index.values)
        subdom_list.remove(domain_name)
        if subdom_list:
            for sub in subdom_list:
                print(sub)
                plot_BLH(datetime=datetime, steps=steps, model=model, domain_name=sub, data_domain=data_domain,
                    domain_lonlat=domain_lonlat, legend=legend, info=info, grid=grid, url=url,
                    dmet=dmet)
        else:
            plot_BLH(datetime=datetime, steps=steps, model=model, domain_name=domain_name,
                data_domain=data_domain, dmet=dmet,
                domain_lonlat=domain_lonlat, legend=legend, info=info, grid=grid, url=url)


if __name__ == "__main__":
    args = default_arguments()
    BLH(datetime=args.datetime, steps=args.steps, model=args.model, domain_name=args.domain_name,
        domain_lonlat=args.domain_lonlat, legend=args.legend, info=args.info, grid=args.grid, url=args.url,
        point_lonlat =args.point_lonlat,use_latest=args.use_latest,delta_index=args.delta_index)

    #del bad_param
    gc.collect()


def is_box_in_box():
    pass
