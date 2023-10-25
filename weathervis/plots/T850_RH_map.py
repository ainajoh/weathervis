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


def plot_TRH(datetime, data_domain, dmet, steps=[0,2], coast_details="auto", model=None, domain_name=None,
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

    # convert fields
    dmet.air_pressure_at_sea_level /= 100
    dmet.air_temperature_2m -= 273.15
    dmet.air_temperature_pl -= 273.15
    dmet.relative_humidity_pl *= 100.0

    lonlat = [dmet.longitude[0,0], dmet.longitude[-1,-1], dmet.latitude[0,0], dmet.latitude[-1,-1]]
    print(lonlat)

    # PLOTTING ROUTNE ######################
    crs = default_map_projection(dmet)
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9), subplot_kw={"projection": crs})
    itim = 0
    for dt in datetime:
        for leadtime in np.array(steps):
            print("Plotting {0} + {1:02d} UTC".format(dt, leadtime))
            print(np.shape(MSLP))
            print(np.shape(default_mslp_contour))
            print(itim)
            ax1 = default_mslp_contour(
                dmet.x, dmet.y, MSLP[itim,0, :, :], ax1, scale=scale
            )
            
            Z = dmet.surface_geopotential[itim, 0, :, :]
            TA = np.where(Z < 3000, dmet.air_temperature_pl[itim, plev, :, :], np.NaN).squeeze()
            RH = (dmet.relative_humidity_pl[itim, plev, :, :]).squeeze()
            
            CF_T = ax1.contourf(
                dmet.x,
                dmet.y,
                TA,
                zorder=1,
                alpha=1,
                levels=np.arange(-40, 20, 1.0),
                label="TA",
                cmap="PRGn",
                extend="both",
            )
            C_T = ax1.contour(
                dmet.x,
                dmet.y,
                TA,
                zorder=4,
                alpha=1,
                levels=np.arange(-40, 20, 1.0),
                label="TA",
                linewidths=0.7,
                colors="red",
            )
            CF_RH = ax1.contour(
                dmet.x, 
                dmet.y, 
                RH, 
                zorder=4, 
                alpha=0.5,
                levels=np.linspace(70, 90, 3), 
                colors="blue", 
                linewidths=0.7,
                label = "RH 70-90% "
            )

            # coastline
            ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'),edgecolor="black", linewidth=1)  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
            #ax1.add_feature(cfeature.GSHHSFeature(scale=coast_details))

            # Done plotting, now adjusting
            ax_cb = adjustable_colorbar_cax(fig1, ax1)
            
            ax1.text(
                0,
                1,
                "{0}_T850_RH_{1}+{2:02d}".format(model, dt, leadtime),
                ha="left",
                va="bottom",
                transform=ax1.transAxes,
                color="dimgrey",
            )
            if legend:
                pressure_dim = list(
                    filter(re.compile(f"press*").match, dmet.__dict__.keys())
                )  # need to find the correcvt pressure name
                llg = {
                    "W_over": {
                        "color": "red",
                        "linestyle": "dashed",
                        "legend": f"W [m s-1]>0.07 m/s at {dmet.__dict__[pressure_dim[0]][plev]:.0f} hPa",
                    },
                    "W_under": {
                        "color": "blue",
                        "linestyle": "None",
                        "legend": f"W [m s-1]<0.07 m/s at {dmet.__dict__[pressure_dim[0]][plev]:.0f} hPa",
                    },
                    "MSLP": {"color": "gray", "linestyle": None, "legend": "MSLP [hPa]"},
                }
                nice_legend(llg, ax1)
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
            file_path = "{0}/{1}_{2}_{3}_{4}+{5:02d}.png".format(
                make_modelrun_folder, model, domain_name, "T850_RH", dt, leadtime
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


def TRH(datetime,use_latest, delta_index, coast_details, steps=0, model="MEPS", domain_name=None, domain_lonlat=None, legend=False, info=False, grid=True,
        runid=None, outpath=None, url=None, point_lonlat =None,overlays=None, point_name=None):
    param = [
        "air_pressure_at_sea_level",
        "surface_geopotential",
        "air_temperature_2m",
        "air_temperature_pl",
        "relative_humidity_pl"
    ]
    
    p_level = [850]
    print(datetime)
    
    plot_by_subdomains(plot_TRH,checkget_data_handler, datetime, steps, model, domain_name, domain_lonlat, legend,
                       info, grid, url, point_lonlat, use_latest,
                       delta_index, coast_details, param, p_level,overlays, runid, point_name)


if __name__ == "__main__":
    args = default_arguments()
    print(args.datetime)
    
    chunck_func_call(
            func=TRH,chunktype= args.chunktype, chunk=args.chunks, datetime=args.datetime, steps=args.steps, model=args.model,
            domain_name=args.domain_name, domain_lonlat=args.domain_lonlat, legend=args.legend, info=args.info, grid=args.grid, runid=args.id,
            outpath=args.outpath, use_latest=args.use_latest,delta_index=args.delta_index, coast_details=args.coast_details, url=args.url,
            point_lonlat =args.point_lonlat, overlays= args.overlays, point_name=args.point_name)
    gc.collect()
