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
def domain_input_handler(dt, model, domain_name, domain_lonlat, file):
  if domain_name or domain_lonlat:
    if domain_lonlat:
      print(f"\n####### Setting up domain for coordinates: {domain_lonlat} ##########")
      data_domain = domain(dt, model, file=file, lonlat=domain_lonlat)
    else:
      data_domain = domain(dt, model, file=file)

    if domain_name != None and domain_name in dir(data_domain):
      print(f"\n####### Setting up domain: {domain_name} ##########")
      domain_name = domain_name.strip()
      if re.search("\(\)$", domain_name):
        func = f"data_domain.{domain_name}"
      else:
        func = f"data_domain.{domain_name}()"
      eval(func)
    else:
      print(f"No domain found with that name; {domain_name}")
  else:
    data_domain=None
  return data_domain

def OLR_sat(datetime, steps=0, model= "MEPS", domain_name = None, domain_lonlat = None, legend=False, info = False,grid=True):

  for dt in datetime: #modelrun at time..
    print(dt)
    param = ["toa_outgoing_longwave_flux","air_pressure_at_sea_level","surface_geopotential"]
    dmap_meps, data_domain, bad_param = checkget_data_handler( model=model, all_param=param,step=steps, date=dt, domain_name=domain_name)

    dmap_meps.air_pressure_at_sea_level/=100


    lon0 = dmap_meps.longitude_of_central_meridian_projection_lambert
    lat0 = dmap_meps.latitude_of_projection_origin_projection_lambert
    parallels = dmap_meps.standard_parallel_projection_lambert

    #fig = plt.figure(figsize=(7, 9))
    # setting up projection
    globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6371000., semiminor_axis=6371000.)
    crs = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0, standard_parallels=parallels,
                                globe=globe)

    for tim in np.arange(np.min(steps), np.max(steps)+1, 1):
      fig, ax = plt.subplots(1, 1, figsize=(7, 9),
                               subplot_kw={'projection': crs})

      ttt = tim
      tidx = tim - np.min(steps)
      ZS = dmap_meps.surface_geopotential[tidx, 0, :, :]
      MSLP = np.where(ZS < 3000, dmap_meps.air_pressure_at_sea_level[tidx, 0, :, :], np.NaN).squeeze()

      #ax = plt.subplot(projection=crs)

      print('Plotting {0} + {1:02d} UTC'.format(dt, ttt))

      C_P = ax.contour(dmap_meps.x, dmap_meps.y, MSLP, zorder=10, alpha=0.6,
                        levels=np.arange(round(np.nanmin(MSLP), -1) - 10, round(np.nanmax(MSLP), -1) + 10, 1),
                        colors='cyan', linewidths=0.5)
      C_P = ax.contour(dmap_meps.x, dmap_meps.y, MSLP, zorder=10, alpha=0.6,
                        levels=np.arange(round(np.nanmin(MSLP), -1) - 10, round(np.nanmax(MSLP), -1) + 10, 5),
                        colors='cyan', linewidths=1.0, label="MSLP [hPa]")
      ax.clabel(C_P, C_P.levels, inline=True, fmt="%3.0f", fontsize=10)

      #It is a bug in pcolormesh. supposedly newest is correct, but not older versions. Invalid corner values set to nan
      #https://github.com/matplotlib/basemap/issues/470
      x,y = np.meshgrid(dmap_meps.x, dmap_meps.y)
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
              (x[1:, 1:] > 1e20)
      )
      data =  dmap_meps.toa_outgoing_longwave_flux[tidx, 0,:nx - 1, :ny - 1].copy()
      data[mask] = np.nan
      #ax.pcolormesh(x, y, data[ :, :])#, cmap=plt.cm.Greys_r)

      ax.pcolormesh(x, y, data[ :, :], vmin=-230,vmax=-110, cmap=plt.cm.Greys_r)


      ax.add_feature(cfeature.GSHHSFeature(scale='intermediate'),edgecolor="brown", linewidth=0.5)  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).


      make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(dt))
      ax.text(0, 1, "{0}_{1}+{2:02d}".format(model, dt, ttt), ha='left', va='bottom', \
               transform=ax.transAxes, color='dimgrey')

      if grid:
        nicegrid(ax=ax,color="orange")
      fig.savefig(make_modelrun_folder + "/{0}_{1}_OLR_sat_{2}+{3:02d}.png".format(model, domain_name, dt, ttt), bbox_inches="tight", dpi=200)

      ax.cla()
      fig.clf()
      plt.close(fig)

    ax.cla()
    plt.clf()
  plt.close("all")


# fin

if __name__ == "__main__":
  import argparse
  def none_or_str(value):
    if value == 'None':
      return None
    return value
  parser = argparse.ArgumentParser()
  parser.add_argument("--datetime", help="YYYYMMDDHH for modelrun", required=True, nargs="+")
  parser.add_argument("--steps", default=0, nargs="+", type=int,help="forecast times example --steps 0 3 gives time 0 to 3")
  parser.add_argument("--model",default="MEPS", help="MEPS or AromeArctic")
  parser.add_argument("--domain_name", default=None, help="see domain.py", type = none_or_str)
  parser.add_argument("--domain_lonlat", default=None, help="[ lonmin, lonmax, latmin, latmax]")
  parser.add_argument("--legend", default=False, help="Display legend")
  parser.add_argument("--grid", default=True, help="Display legend")

  parser.add_argument("--info", default=False, help="Display info")
  args = parser.parse_args()
  OLR_sat(datetime=args.datetime, steps = args.steps, model = args.model, domain_name = args.domain_name,
          domain_lonlat=args.domain_lonlat, legend = args.legend, info = args.info,grid=args.grid)
  #datetime, step=4, model= "MEPS", domain = None
