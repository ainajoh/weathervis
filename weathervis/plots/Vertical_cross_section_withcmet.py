# %%
#python Vertical_cross_section_withcmet.py - -datetime 2022032500 - -model AromeArctic#
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

warnings.filterwarnings("ignore", category=UserWarning) # suppress matplotlib warning

def plot_VC_cmet(datetime, data_domain, dmet, steps=[0,2], coast_details="auto", model= None, domain_name = None,
             domain_lonlat = None, legend=False, info = False, save = True,grid=True, url = None, overlays=None, runid=None):
    print("start plot here, modeldata is in cmet")
    print(np.shape(dmet.relative_humidity_pl))
    print(dmet.pressure)






def VC_cmet(datetime, steps, model, domain_name, domain_lonlat, legend, info, grid, url,point_lonlat,
        use_latest,delta_index, coast_details, overlays, runid, outpath):
    search_param = check_data(model = model,search="humidity")#, p_level=[800,850,1000, 900])
    search_param = check_data(model = model,date=datetime)#, p_level=[800,850,1000, 900])
    #print(search_param.file.p_levels.loc[2])
    param = ["air_pressure_at_sea_level", "surface_geopotential", "relative_humidity_pl"]
    p_level =  [50, 100, 150, 200, 250, 300, 400, 500, 700, 800, 850, 925, 1000]
    plot_by_subdomains(plot_VC_cmet, checkget_data_handler, datetime, steps, model, domain_name, domain_lonlat, legend,
                       info, grid, url, point_lonlat, use_latest,
                       delta_index, coast_details, param, p_level, overlays, runid)

if __name__ == "__main__":
    args = default_arguments()

    chunck_func_call(func=VC_cmet, chunktype=args.chunktype, chunk=args.chunks, datetime=args.datetime, steps=args.steps,
                     model=args.model,
                     domain_name=args.domain_name, domain_lonlat=args.domain_lonlat, legend=args.legend, info=args.info,
                     grid=args.grid, runid=args.id,
                     outpath=args.outpath, use_latest=args.use_latest, delta_index=args.delta_index,
                     coast_details=args.coast_details, url=args.url,
                     point_lonlat=args.point_lonlat, overlays=args.overlays)