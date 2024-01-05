# %%

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
from weathervis.plots.add_overlays_new import add_overlay

import warnings
import gc

warnings.filterwarnings("ignore", category=UserWarning)  # suppress matplotlib warning

def none_or_str(value):
    if value == 'None':
        return None
    return value

def custom_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datetime", help="YYYYMMDDHH for modelrun", default=None,  type=str, nargs="+")
    parser.add_argument("--steps", default=["0"], nargs="+", type=str,
                        help="forecast times example --steps 0 3 gives time 0 and 3 \n --steps 0:1:3 gives timestep 0, 1, 2")
    parser.add_argument("--model", default=None, help="MEPS or AromeArctic")
    parser.add_argument("--domain_name", default=None, nargs="+", type= none_or_str)
    parser.add_argument("--domain_lonlat", default=None, help="[ lonmin, lonmax, latmin, latmax]")
    parser.add_argument("--point_name", default=None, nargs="+")
    parser.add_argument("--point_lonlat", default=None, help="[lon, lat]", type=str, nargs="+")
    parser.add_argument("--num_point", default=1, type=int)
    parser.add_argument("--legend", default=True, help="Display legend")
    parser.add_argument("--grid", default=True, help="Display legend")
    parser.add_argument("--info", default=False, help="Display info")
    parser.add_argument("--url", default=None, help="use url", type=str)
    parser.add_argument("--use_latest", default=False, type=bool)
    parser.add_argument("--delta_index", default=None, type=str)
    parser.add_argument("--coast_details", default="auto", type=str, help="auto, coarse, low, intermediate, high, or full")
    parser.add_argument("--id", default=None, help="Display legend", type=str)
    parser.add_argument("--outpath", default=None, help="Display legend", type=str)
    parser.add_argument("--overlays", default=None, nargs="+", help="Display legend", type=str)
    parser.add_argument("--chunktype", default=None,  help="eg steps", type=str)
    parser.add_argument("--chunks", default=6,  help="Display legend", type=int)
    parser.add_argument("--p_level", default=1000, nargs="+", type=int,help="p_level example --p_level 1000 925")


    args = parser.parse_args()

    global OUTPUTPATH
    if args.outpath != None:
        OUTPUTPATH = args.outpath
    if args.point_lonlat != None: #aina
        splitted_lonlat= re.split("]",  "".join(args.point_lonlat))
        s = re.split(",",  "".join(splitted_lonlat).replace("[", ""))        
        lonlat = np.reshape(s, (int(len(s)/2), 2)).astype(float).tolist()
        args.point_lonlat = lonlat

    step_together = "".join(args.steps[0:])
    if ":" in step_together:
        splitted_steps= re.split(":",  "".join(args.steps[0:]))
        sep = int(splitted_steps[2]) if len(splitted_steps) == 3 else 1
        args.steps = list(np.arange( int(splitted_steps[0]), int(splitted_steps[1]), int(sep) ))
    else:
        step = [args.steps] if type(args.steps) == int else args.steps
        args.steps = [int(i) for i in step]

    if args.domain_name == None:
        args.domain_name = [args.model]
    return args


def plot_Q(datetime, data_domain, dmet, steps=[0,2], coast_details="auto", model=None, domain_name=None,
             domain_lonlat=None, legend=True, info=False, grid=True,runid=None, outpath=None, url = None, save= True, overlays=None, **kwargs):

    eval(f"data_domain.{domain_name}()")  # get domain info
    ## CALCULATE AND INITIALISE ####################

    scale = (
        data_domain.scale
    )  # scale is larger for smaller domains in order to scale it up.
    dmet.air_pressure_at_sea_level /= 100
    plev = 0  # velocity presuure level at first request
    MSLP = filter_values_over_mountain(
        dmet.surface_geopotential[:], dmet.air_pressure_at_sea_level[:]
    )

    # prepare plot
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
        (x[1:, 1:] > 1e20)
    )

    lonlat = [dmet.longitude[0,0], dmet.longitude[-1,-1], dmet.latitude[0,0], dmet.latitude[-1,-1]]
    print(lonlat)

    # PLOTTING ROUTNE ######################
    crs = default_map_projection(dmet)
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9), subplot_kw={"projection": crs})
    itim = 0
    for dt in datetime:
        # loop over pressure levels, do not forget the indent the whole routine
        for ip,p in enumerate(dmet.pressure):
            for leadtime in np.array(steps):
                print("Plotting {0} + {1:02d} UTC".format(dt, leadtime))
                print(np.shape(MSLP))
                print(np.shape(default_mslp_contour))
                print(itim)
                ax1 = default_mslp_contour(
                    dmet.x, dmet.y, MSLP[itim,0, :, :], ax1, scale=scale
                )
                
                # plot spec humidity
                data = dmet.specific_humidity_pl[itim,ip,:,:]*1000 # get g/kg
                #CC=ax1.contourf(x,y,data,levels=[0.5,1.0, 1.5, 2.0, 2.5, 3.5, 5,10],
                #                colors=('#FFFFFF','#ffffd9','#e0f3b2','#97d6b9','#41b6c4',
                #                        '#1f80b8','#24429b','#081d58','#800080'),vmin=0,
                #                vmax=5,zorder=2,extend='both') 
                # summer-time range
                CC=ax1.contourf(x,
                                y,
                                data,
                                levels=[3, 5, 6, 7, 8, 9, 10,12, 14, 16],
                                colors=('#FFFFFF','#ffffd9','#e0f3b2','#97d6b9','#41b6c4',
                                        '#1f80b8','#24429b','#081d58','#800080'),
                                vmin=3,
                                vmax=16,
                                zorder=0,
                                extend='both'
                                ) 
                
                # coastline
                ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'),edgecolor="black", linewidth=1)  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
                #ax1.add_feature(cfeature.GSHHSFeature(scale=coast_details))

                # Done plotting, now adjusting
                ax_cb = adjustable_colorbar_cax(fig1, ax1)
                
                ax1.text(0,
                         1,
                         "{0}_Q_{1}_{2}+{3:02d}".format(model,int(p), dt, leadtime), 
                         ha='left', 
                         va='bottom',
                         transform=ax1.transAxes,
                         color='dimgrey'
                )
                if legend:
                    plt.colorbar(CC,cax = ax_cb, fraction=0.046, pad=0.01, aspect=25,
                                    label=r"specific humidity (g/kg)",extend='max')

                    proxy = [plt.axhline(y=0, xmin=0, xmax=0, color="gray",zorder=7)]
                    # proxy.extend(proxy1)
                    # legend's location fixed, otherwise it takes very long to find optimal spot
                    lg = ax1.legend(proxy, ["MSLP [hPa]"],loc='upper left')
                    frame = lg.get_frame()
                    frame.set_facecolor('white')
                    frame.set_alpha(0.8)
                if grid:
                    nicegrid(ax=ax1)
                if overlays:
                    add_overlay(overlays,ax=ax1,col="black", **kwargs)

                print(data_domain.lonlat)  # [15.8, 16.4, 69.2, 69.4]
                if domain_name != model and data_domain != None:  #
                    ax1.set_extent(data_domain.lonlat)

                # runid == "" if runid == None else runid
                # make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}-{1}".format(dt, runid))

                make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(dt))
                file_path = make_modelrun_folder +"/{0}_{1}_{2}_{3}_{4}+{5:02d}.png".format(
                    model, domain_name, "Q",int(p), dt, leadtime
                    )

                print(f"filename: {file_path}")
                if save:
                    fig1.savefig(file_path, bbox_inches="tight", dpi=200)

                else:
                    plt.show()
                ax1.cla()
                itim += 1
    plt.close(fig1)
    plt.close("all")
    del MSLP, scale, itim, legend, grid, overlays, domain_name, ax_cb
    del dmet, data_domain
    del fig1, ax1, crs
    del make_modelrun_folder, file_path
    gc.collect()


def Q(datetime,use_latest, delta_index, coast_details, steps=0, model="MEPS", domain_name=None, domain_lonlat=None, legend=False, info=False, grid=True,
        runid=None, outpath=None, url=None, point_lonlat =None,overlays=None, point_name=None,p_level=None):
    param = ['specific_humidity_pl',
             'air_pressure_at_sea_level',
             'surface_geopotential'
             ]
    
    #p_level = None
    print(datetime)
    
    plot_by_subdomains(plot_Q,checkget_data_handler, datetime, steps, model, domain_name, domain_lonlat, legend,
                       info, grid, url, point_lonlat, use_latest,
                       delta_index, coast_details, param, p_level,overlays, runid, point_name)


if __name__ == "__main__":
    args = custom_arguments()
    print(args.datetime)
    
    chunck_func_call(
            func=Q,chunktype= args.chunktype, chunk=args.chunks, datetime=args.datetime, steps=args.steps, model=args.model,
            domain_name=args.domain_name, domain_lonlat=args.domain_lonlat, legend=args.legend, info=args.info, grid=args.grid, runid=args.id,
            outpath=args.outpath, use_latest=args.use_latest,delta_index=args.delta_index, coast_details=args.coast_details, url=args.url,
            point_lonlat =args.point_lonlat, overlays= args.overlays, point_name=args.point_name,p_level=args.p_level)
    gc.collect()