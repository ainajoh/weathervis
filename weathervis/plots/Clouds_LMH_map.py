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
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable  ##__N
from weathervis.checkget_data_handler import checkget_data_handler
from weathervis.plots.add_overlays_new import add_overlay

import warnings
import gc

warnings.filterwarnings("ignore", category=UserWarning)  # suppress matplotlib warning


def plot_LMH(datetime, data_domain, dmet, steps=[0,2], coast_details="auto", model=None, domain_name=None,
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
        for leadtime in np.array(steps):
            print("Plotting {0} + {1:02d} UTC".format(dt, leadtime))
            print(np.shape(MSLP))
            print(np.shape(default_mslp_contour))
            print(itim)
            
            LC = dmet.low_type_cloud_area_fraction[itim,0,:,:].squeeze()
            MC = dmet.medium_type_cloud_area_fraction[itim,0,:,:].squeeze()
            HC = dmet.high_type_cloud_area_fraction[itim,0,:,:].squeeze()
            # reduce the detail of the plot -> everythin larger 0.5 =1, rest is nan
            # I chose the colormaps here due to their nice "end" colors
            data =  LC[:, :].copy()
            data[np.where(data>=0.5)] = 1
            data[np.where(data<0.5)] = np.nan
            CCl=ax1.contourf(x,
                             y,
                             data,
                             levels=np.linspace(0.0, 1, 10),
                             cmap=plt.cm.coolwarm
                             ,alpha=0.9,
                             zorder=1
                             )
        
            data =  MC[:, :].copy()
            data[np.where(data>=0.5)] = 1
            data[np.where(data<0.5)] = np.nan # take 0 and very small cloud covers away
            CCl=ax1.contourf(x,
                             y,
                             data,
                             levels=np.linspace(0.0, 1, 10),
                             cmap=plt.cm.coolwarm_r,
                             alpha=0.8,
                             zorder=1
                             )
            
            data =  HC[:, :].copy()
            data[np.where(data>=0.5)] = 1
            data[np.where(data<0.5)] = np.nan # take 0 and very small cloud covers away
            CCl=ax1.contourf(x,
                             y,
                             data,
                             levels=np.linspace(0.0, 1, 10),
                             cmap=plt.cm.gnuplot,
                             alpha=0.4,
                             zorder=1
                             ) 
            ax1 = default_mslp_contour(
                dmet.x, dmet.y, MSLP[itim,0, :, :], ax1, scale=scale
            )

            # coastline
            ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'),edgecolor="black", linewidth=1)  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
            #ax1.add_feature(cfeature.GSHHSFeature(scale=coast_details))

            # Done plotting, now adjusting            
            ax1.text(
                0,
                1,
                "{0}_CCLMH_{1}+{2:02d}".format(model, dt, leadtime),
                ha="left",
                va="bottom",
                transform=ax1.transAxes,
                color="dimgrey",
            )
            if legend:
                custom_lines = [Line2D([0], [0], color='grey', lw=2),
                                Line2D([0], [0], color='C3', lw=8),
                                Line2D([0], [0], color='#0099FF', lw=8),
                                Line2D([0], [0], color='#FFCC00', lw=8,alpha=0.5)]
                lg = ax1.legend(custom_lines,
                                ['MSLP', 'low clouds', 'medium clouds','high clouds'],
                                loc='upper left')
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
            file_path = "{0}/{1}_{2}_{3}_{4}+{5:02d}.png".format(
                make_modelrun_folder, model, domain_name, "CCLMH", dt, leadtime
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
    del MSLP, scale, itim, legend, grid, overlays, domain_name
    del dmet, data_domain
    del fig1, ax1, crs
    del make_modelrun_folder, file_path
    gc.collect()


def LMH(datetime,use_latest, delta_index, coast_details, steps=0, model="MEPS", domain_name=None, domain_lonlat=None, legend=False, info=False, grid=True,
        runid=None, outpath=None, url=None, point_lonlat =None,overlays=None, point_name=None):
    param = [
            'low_type_cloud_area_fraction',
            'medium_type_cloud_area_fraction', 
            'high_type_cloud_area_fraction',
            'air_pressure_at_sea_level',
            'surface_geopotential'
            ]
    
    p_level = None
    print(datetime)
    
    plot_by_subdomains(plot_LMH,checkget_data_handler, datetime, steps, model, domain_name, domain_lonlat, legend,
                       info, grid, url, point_lonlat, use_latest,
                       delta_index, coast_details, param, p_level,overlays, runid, point_name)


if __name__ == "__main__":
    args = default_arguments()
    print(args.datetime)
    
    chunck_func_call(
            func=LMH,chunktype= args.chunktype, chunk=args.chunks, datetime=args.datetime, steps=args.steps, model=args.model,
            domain_name=args.domain_name, domain_lonlat=args.domain_lonlat, legend=args.legend, info=args.info, grid=args.grid, runid=args.id,
            outpath=args.outpath, use_latest=args.use_latest,delta_index=args.delta_index, coast_details=args.coast_details, url=args.url,
            point_lonlat =args.point_lonlat, overlays= args.overlays, point_name=args.point_name)
    gc.collect()