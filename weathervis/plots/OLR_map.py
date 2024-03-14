# %%
#python OLR_sat.py --datetime 2022030400 --steps 0 3 --model AromeArctic --domain_name Svalbard North_Norway --use_latest 0
#The most updatet map plot setup for now. 
from weathervis.config import *
from weathervis.utils import (
filter_values_over_mountain, 
default_map_projection, 
default_mslp_contour, 
plot_by_subdomains
)
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
from weathervis.plots.add_overlays import add_overlay

#Settings that makes it easy to import the default parameter for this plot from another script
global OLR_map
MyObject = type('MyObject', (object,), {})
OLR_map = MyObject()
_param = ["air_pressure_at_sea_level", "surface_geopotential", "toa_outgoing_longwave_flux", "SIC"]
setattr(OLR_map, "param", _param)
warnings.filterwarnings("ignore", category=UserWarning) # suppress matplotlib warning

def plot_OLR(datetime,dmet,figax=None, scale=1, lonlat=None, steps=[0,2], coast_details="auto", model=None, domain_name=None,
             domain_lonlat=None, legend=True, info=False, grid=True,runid=None, outpath=None, url = None, save= True, overlays=None,  data_domain=None, **kwargs):

  if domain_name != None: 
     eval(f"data_domain.{domain_name}()")  # get domain info
     scale = data_domain.scale  #scale is larger for smaller domains in order to scale it up.

  ## CALCULATE AND INITIALISE ####################
  dmet.air_pressure_at_sea_level /= 100
  MSLP = filter_values_over_mountain(dmet.surface_geopotential[:], dmet.air_pressure_at_sea_level[:])
  # PLOTTING ROUTNE ######################
  crs = default_map_projection(dmet) #change if u want another projection
  fig1, ax1 = plt.subplots(1, 1, figsize=(7*3, 9*3), subplot_kw={'projection': crs}) if figax is None else figax
  itim = 0
  for leadtime in np.array(steps): #
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
      ax1.contour(dmet.x, dmet.y,
                             dmet.SIC[itim, :, :] if len(np.shape(dmet.SIC)) == 3 else dmet.SIC[itim,0, :, :],
                             zorder=2, linewidths=2.0, colors="black", levels=[0.1, 0.5])
      #ax1.plot(7,80,transform=ccrs.PlateCarree())
      #AINA remove when donen ssc---v---
      lon_blue = [7,7,7,7,7]
      lat_blue = [84,83,82,81,80]
      blueline = ax1.plot([7,7,7,7,7,7], [84,83,82,81,80,79], transform=ccrs.PlateCarree(),
      color='#70B6F4', zorder=6, linewidth=10)

      bluepoint = ax1.scatter(lon_blue, lat_blue, s=900, transform=ccrs.PlateCarree(),
      color='#70B6F4', zorder=7, linestyle='-', edgecolors='k', linewidths=3, marker='s')
      
      lon_red = [7,7,7,7,7,7,7,7,7,7,7,7,7]
      lat_red = [79,78,77,76,75,74,73,72,71,70,69,68,67]
      redline = ax1.plot([7,7,7,7,7,7,7,7,7,7,7,7,7], [79,78,77,76,75,74,73,72,71,70,69,68,67], transform=ccrs.PlateCarree(),
      color='#FD7173', zorder=6, linewidth=10)
      
      redpoint = ax1.scatter(lon_red, lat_red, s=900, transform=ccrs.PlateCarree(),
      color='#FD7173', zorder=7, linestyle='-', edgecolors='k', linewidths=3, marker='s')
       #AINA remove when donen ssc -----

      
      if grid:
        nicegrid(ax=ax1,color="orange")
      if overlays:
        add_overlay(overlays, ax=ax1, **kwargs)
      if domain_name != model and data_domain != None:
        ax1.set_extent(data_domain.lonlat)

      make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(datetime))
      file_path = "{0}/{1}_{2}_{3}_{4}+{5:02d}.png".format(make_modelrun_folder, model, domain_name, "OLR", datetime,leadtime)
      print(f"filename: {file_path}")
      if save: 
        fig1.savefig(file_path, bbox_inches="tight", dpi=300) 
      else:
        pass
        #plt.show()
      ax1.cla()
      itim += 1
  del MSLP, scale, itim, legend, grid, overlays, domain_name, data, mask, x, y,nx,ny
  del make_modelrun_folder, file_path
  if figax is None:
    plt.close(fig1)
    plt.close("all")
    del dmet, data_domain
    del fig1, ax1, crs
  gc.collect()

def OLR(datetime,use_latest, delta_index, coast_details, steps=0, model="MEPS", domain_name=None, domain_lonlat=None, legend=False, info=False, grid=True,
        runid=None, outpath=None, url=None, point_lonlat =None,overlays=None, point_name=None):
    param = OLR_map.param #["air_pressure_at_sea_level", "surface_geopotential", "toa_outgoing_longwave_flux", "SIC"]
    p_level = None
    plot_by_subdomains(plot_OLR, checkget_data_handler, datetime, steps, model, domain_name, domain_lonlat, legend,
                       info, grid, url, point_lonlat, use_latest,
                       delta_index, coast_details, param, p_level,overlays, runid, point_name)

if __name__ == "__main__":
    args = default_arguments()
    chunck_func_call(func = OLR, chunktype= args.chunktype, chunk=args.chunks, datetime=args.datetime, steps=args.steps, model=args.model,
            domain_name=args.domain_name, domain_lonlat=args.domain_lonlat, legend=args.legend, info=args.info, grid=args.grid, runid=args.id,
            outpath=args.outpath, use_latest=args.use_latest,delta_index=args.delta_index, coast_details=args.coast_details, url=args.url,
            point_lonlat =args.point_lonlat, overlays= args.overlays, point_name=args.point_name)
    gc.collect()


    #python OLR_map.py --datetime 2020031000 --steps 10 --model AromeArctic --domain_name AromeArctic --overlays point_name point_name --point_name P84 P83 P82 P81 P80 P79 P78 P77 P76 P75 P74 P73 P72 P71 P70 P69 P68 P67 