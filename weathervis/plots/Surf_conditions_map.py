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


def plot_surf(datetime, data_domain, dmet, steps=[0,2], coast_details="auto", model=None, domain_name=None,
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


    lonlat = [dmet.longitude[0,0], dmet.longitude[-1,-1], dmet.latitude[0,0], dmet.latitude[-1,-1]]
    print(lonlat)

    lon0 = dmet.longitude_of_central_meridian_projection_lambert
    lat0 = dmet.latitude_of_projection_origin_projection_lambert
    parallels = dmet.standard_parallel_projection_lambert

    
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
                                
                
            ZS = dmet.surface_geopotential[itim, 0, :, :]
            MSLP = np.where(ZS < 50000, dmet.air_pressure_at_sea_level[itim, 0, :, :], np.NaN).squeeze()
            #TP = precip_acc(dmet.precipitation_amount_acc, acc=1)[itim, 0, :,:].squeeze()
            L = dmet.LE[itim,:,:].squeeze()
            L = np.where(ZS < 50000, L, np.NaN).squeeze()
            SH = dmet.H[itim,:,:].squeeze()
            SH = np.where(ZS < 50000, SH, np.NaN).squeeze()
            SST = dmet.SST[itim,:,:].squeeze()
            Ux = dmet.x_wind_10m[itim, 0, :, :].squeeze()
            Vx = dmet.y_wind_10m[itim, 0, :, :].squeeze()
            xm,ym = np.meshgrid(dmet.x, dmet.y)
            uxx = dmet.x_wind_10m[itim, 0, :, :].squeeze()
            vxx = dmet.y_wind_10m[itim, 0, :, :].squeeze()

            #SST_new
            #levels=np.arange(270,294,2)
            SST = SST - 273.15
            #levels = [np.min(SST), np.max(SST), 3]
            levels = [-2,0,2,4,6,8,10]
            C_SS = ax1.contour(dmet.x, dmet.y, SST, colors="black", linewidths=2, levels =levels, zorder=2)
            ax1.clabel(C_SS, C_SS.levels, inline=True, fmt="%3.0f", fontsize=10 )

            #wind#
            #skip = (slice(50, -50, 50), slice(50, -50, 50))
            #skip = (slice(10, -10, 30), slice(10, -10, 30)) #70
            skip = (slice(20, -20, 45), slice(20, -20, 45)) #70
            scale = 1.94384
            CVV = ax1.barbs(xm[skip], ym[skip], uxx[skip]*scale, vxx[skip]*scale, length=5.5, zorder=1,alpha=0.6)

            #LATENT_new
            #levels=np.arange(270,294,2)
            cmap = plt.get_cmap("coolwarm")
            levels = np.linspace(-150,200,8)

            CLH = ax1.contourf(dmet.x, 
                               dmet.y, 
                               L, 
                               zorder=1, 
                               levels=levels, 
                               alpha=0.7, 
                               cmap = cmap, 
                               extend = "both", 
                               #transform=data
                               )
            
            ax1.text(0, 1, "{0}_surf_{1}+{2:02d}".format(model, dt, leadtime), ha='left', va='bottom', transform=ax1.transAxes, color='dimgrey')
            ##########################################################
            #handles, labels = ax1.get_legend_handles_labels()

            #SENSIBLE
            CSH = plt.contour(dmet.x, dmet.y, SH, alpha=1.0, colors="blue", linewidths=0.7, zorder=1)
            

            # coastline
            ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'),edgecolor="black", linewidth=1)  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
            #ax1.add_feature(cfeature.GSHHSFeature(scale=coast_details))

            # Done plotting, now adjusting
            ax_cb = adjustable_colorbar_cax(fig1, ax1)
            
            if legend:
                
                llg = {
                    "SH": {
                        "color": "blue",
                        "linestyle": None,
                        "legend": f"Sensible heat [W/m2]",
                    },
                    "SST": {
                        "color": "black",
                        "linestyle": None,
                        "legend": f"SST [C]",
                    },
                }
                nice_legend(llg, ax1,loc=1)
                cb = plt.colorbar(CLH, cax= ax_cb, fraction=0.046, pad=0.01, ax=ax1, aspect=25, label =f"Latent heat (W/m2)", extend = "both")

            if grid:
                nicegrid(ax=ax1)
            if overlays:
                add_overlay(overlays,ax=ax1,col="green", **kwargs)

            print(data_domain.lonlat)  # [15.8, 16.4, 69.2, 69.4]
            if domain_name != model and data_domain != None:  #
                ax1.set_extent(data_domain.lonlat)

            # runid == "" if runid == None else runid
            # make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}-{1}".format(dt, runid))

            make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(dt))
            file_path = "{0}/{1}_{2}_{3}_{4}+{5:02d}.png".format(
                make_modelrun_folder, model, domain_name, "surf", dt, leadtime
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


def surf(datetime,use_latest, delta_index, coast_details, steps=0, model="MEPS", domain_name=None, domain_lonlat=None, legend=False, info=False, grid=True,
        runid=None, outpath=None, url=None, point_lonlat =None,overlays=None, point_name=None):
    param =["surface_geopotential",
            "air_pressure_at_sea_level",
            "x_wind_10m",
            "y_wind_10m",
            "precipitation_amount_acc",
            "wind_speed",
            "LE",
            "H",
            "SST"]

    p_level = None
    print(datetime)
    
    plot_by_subdomains(plot_surf,checkget_data_handler, datetime, steps, model, domain_name, domain_lonlat, legend,
                       info, grid, url, point_lonlat, use_latest,
                       delta_index, coast_details, param, p_level,overlays, runid, point_name)


if __name__ == "__main__":
    args = default_arguments()
    print(args.datetime)
    
    chunck_func_call(
            func=surf,chunktype= args.chunktype, chunk=args.chunks, datetime=args.datetime, steps=args.steps, model=args.model,
            domain_name=args.domain_name, domain_lonlat=args.domain_lonlat, legend=args.legend, info=args.info, grid=args.grid, runid=args.id,
            outpath=args.outpath, use_latest=args.use_latest,delta_index=args.delta_index, coast_details=args.coast_details, url=args.url,
            point_lonlat =args.point_lonlat, overlays= args.overlays, point_name=args.point_name)
    gc.collect()
