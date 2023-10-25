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


def plot_T2M(datetime, data_domain, dmet, steps=[0,2], coast_details="auto", model=None, domain_name=None,
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
    #dmet.air_temperature_pl -= 273.15
    #dmet.relative_humidity_pl *= 100.0

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
                        
            print('Plotting T2M {0} + {1:02d} UTC'.format(dt,leadtime))
            # gather, filter and squeeze variables for plotting
            plev = 0
            #reduces noise over mountains by removing values over a certain height.

            Z = dmet.surface_geopotential[itim, 0, :, :]
            TA = np.where(Z < 50000, dmet.air_temperature_2m[itim, :, :], np.NaN).squeeze()
            # air temperature (C)
            CF_T= ax1.contourf(dmet.x,
                               dmet.y,
                               TA, 
                               zorder=1,
                               alpha=1.0,
                               levels=np.arange(-20, 20, 1.0), 
                               cmap="PRGn")
            TA = np.where(Z < 2000, dmet.air_temperature_2m[itim, :, :], np.NaN).squeeze()
            C_T = ax1.contour(dmet.x,
                              dmet.y,
                              TA,
                              zorder=4,
                              alpha=1.0,
                              levels=np.arange(-20, 20, 1.0), 
                              colors="red",
                              linewidths=0.7)
            ax1.clabel(C_T, C_T.levels[::2], inline=True, fmt="%3.0f", fontsize=10)

            ax1 = default_mslp_contour(
                dmet.x, dmet.y, MSLP[itim,0, :, :], ax1, scale=scale
            )
            # relative humidity above 80%
            #CF_RH = ax1.contour(dmet.x, dmet.y, RH, zorder=4, alpha=0.5,
            #                  levels=np.linspace(70, 100, 4), colors="blue", linewidths=0.7,label = "RH >70% [%]")

            #lat_p = 60.2
            #lon_p = 5.4167
            #mainpoint = ax1.scatter(lon_p, lat_p, s=9.0 ** 2, transform=ccrs.PlateCarree(),
            #                        color='lime', zorder=6, linestyle='None', edgecolors="k", linewidths=3)
            
            # coastline
            ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'),edgecolor="black", linewidth=1)  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
            #ax1.add_feature(cfeature.GSHHSFeature(scale=coast_details))

            # Done plotting, now adjusting
            ax_cb = adjustable_colorbar_cax(fig1, ax1)
            cb = plt.colorbar(CF_T, cax= ax_cb, fraction=0.046, pad=0.01, ax=ax1, aspect=25, label =f"2m Temperature [°C]", extend = "both")

            ax1.text(
                0,
                1,
                "{0}_T2M_{1}+{2:02d}".format(model, dt, leadtime),
                ha="left",
                va="bottom",
                transform=ax1.transAxes,
                color="dimgrey",
            )
            legend = False
            if legend:
                proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0], )
                    for pc in CF_T.collections]
                proxy1 = [plt.axhline(y=0, xmin=1, xmax=1, color="red"),
                        plt.axhline(y=0, xmin=1, xmax=1, color="red", linestyle="dashed"),
                        plt.axhline(y=0, xmin=1, xmax=1, color="gray")]
                proxy.extend(proxy1)
                lg = ax1.legend(proxy, [f"RH > 80% [%] at {dmet.pressure[plev]:.0f} hPa",
                                    f"T>0 [C] at {dmet.pressure[plev]:.0f} hPa",
                                    f"T<0 [C] at {dmet.pressure[plev]:.0f} hPa", "MSLP [hPa]", ""])
                frame = lg.get_frame()
                frame.set_facecolor('white')
                frame.set_alpha(1)
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
                make_modelrun_folder, model, domain_name, "T2M", dt, leadtime
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


def T2M(datetime,use_latest, delta_index, coast_details, steps=0, model="MEPS", domain_name=None, domain_lonlat=None, legend=False, info=False, grid=True,
        runid=None, outpath=None, url=None, point_lonlat =None,overlays=None, point_name=None):
    param = [
        "air_pressure_at_sea_level",
        "air_temperature_2m",           
        "surface_geopotential",
        "relative_humidity_pl"
    ]
    
    p_level = None
    print(datetime)
    
    plot_by_subdomains(plot_T2M,checkget_data_handler, datetime, steps, model, domain_name, domain_lonlat, legend,
                       info, grid, url, point_lonlat, use_latest,
                       delta_index, coast_details, param, p_level,overlays, runid, point_name)


if __name__ == "__main__":
    args = default_arguments()
    print(args.datetime)
    
    chunck_func_call(
            func=T2M,chunktype= args.chunktype, chunk=args.chunks, datetime=args.datetime, steps=args.steps, model=args.model,
            domain_name=args.domain_name, domain_lonlat=args.domain_lonlat, legend=args.legend, info=args.info, grid=args.grid, runid=args.id,
            outpath=args.outpath, use_latest=args.use_latest,delta_index=args.delta_index, coast_details=args.coast_details, url=args.url,
            point_lonlat =args.point_lonlat, overlays= args.overlays, point_name=args.point_name)
    gc.collect()
