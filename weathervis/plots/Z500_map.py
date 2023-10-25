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
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable  ##__N
from weathervis.checkget_data_handler import checkget_data_handler
from weathervis.plots.add_overlays_new import add_overlay

import warnings
import gc

warnings.filterwarnings("ignore", category=UserWarning)  # suppress matplotlib warning


def plot_Z500(datetime, data_domain, dmet, steps=[0,2], coast_details="auto", model=None, domain_name=None,
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
    dmet.air_pressure_at_sea_level/=100
    #dmet.precipitation_amount_acc*=1000.0
    print(dmet.units.precipitation_amount_acc)
    dmet.geopotential_pl/=10.0
    dmet.units.geopotential_pl ="m"
    #u,v = xwind2uwind(dmet.x_wind_pl,dmet.y_wind_pl, dmet.alpha)
    vel = wind_speed(dmet.x_wind_pl,dmet.y_wind_pl)


    lonlat = [dmet.longitude[0,0], dmet.longitude[-1,-1], dmet.latitude[0,0], dmet.latitude[-1,-1]]
    print(lonlat)

    lon0 = dmet.longitude_of_central_meridian_projection_lambert
    lat0 = dmet.latitude_of_projection_origin_projection_lambert
    parallels = dmet.standard_parallel_projection_lambert

    
    # PLOTTING ROUTNE ######################
    crs = default_map_projection(dmet)
    itim = 0
    for dt in datetime:
        for leadtime in np.array(steps):
            fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9), subplot_kw={"projection": crs})

            print("Plotting {0} + {1:02d} UTC".format(dt, leadtime))
            print(np.shape(MSLP))
            print(np.shape(default_mslp_contour))
            print(itim)
            ax1 = default_mslp_contour(
                dmet.x, dmet.y, MSLP[itim,0, :, :], ax1, scale=scale
            )
            
            print('Plotting Z500 {0} + {1:02d} UTC'.format(dt, leadtime))
            plev2 = 0
            embr = 0
            ZS = dmet.surface_geopotential[itim, 0, :, :]
            acc = 1
            TP = precip_acc(dmet.precipitation_amount_acc,acc=acc)[itim, 0, :,:].squeeze()
            VEL = (vel[itim, plev2, :, :]).squeeze()
            Z = (dmet.geopotential_pl[itim, plev2, :, :]).squeeze()
            #Ux = u[itim, 0,:, :].squeeze()
            #Vx = v[itim, 0,:, :].squeeze()
            uxx = dmet.x_wind_pl[itim, 0,:, :].squeeze()
            vxx = dmet.y_wind_pl[itim, 0,:, :].squeeze()
            cmap = plt.get_cmap("tab20c")
            lvl = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 1.5, 2, 3, 5, 10, 15]
            norm = mcolors.BoundaryNorm(lvl, cmap.N)
            
            #try: #workaround for a stupid matplotlib error not handling when all values are outside of range in lvl or all just nans..
                #https://github.com/SciTools/cartopy/issues/1290
                #cmap =  mcolors.ListedColormap('hsv', 'hsv') #plt.get_cmap("hsv")PuBu
                #TP.filled(np.nan) #fill mask with nan to avoid:  UserWarning: Warning: converting a masked element to nan.
            CF_prec = plt.contourf(dmet.x,
                                    dmet.y,
                                    TP,
                                    zorder=1,
                                    cmap=cmap, 
                                    norm = norm,
                                    alpha=0.4,
                                    #antialiased=True,
                                    levels=lvl,
                                    extend = "max"
                                    )#
            #except:
            #    pass

            skip = (slice(20, -20, 50), slice(20, -20, 50)) #70
            xm,ym=np.meshgrid(dmet.x, dmet.y)
            scale=1.94384
            CVV = ax1.barbs(xm[skip], 
                            ym[skip], 
                            uxx[skip]*scale, 
                            vxx[skip]*scale,
                            length=5.5, 
                            zorder=5)
            #CS = ax1.contour(dmet.x, dmet.y, VEL, zorder=3, alpha=1.0,
            #                   levels=np.arange(-80, 80, 5), colors="green", linewidths=0.7)
            # geopotential
            CS = ax1.contour(dmet.x,
                             dmet.y, 
                             Z, 
                             zorder=3, 
                             alpha=1.0,
                             levels=np.arange(4600, 5800, 20), 
                             colors="blue",
                             linewidths=0.7)
            ax1.clabel(CS, CS.levels, inline=True, fmt="%4.0f", fontsize=10)

            
            # coastline
            ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'),edgecolor="black", linewidth=1)  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
            #ax1.add_feature(cfeature.GSHHSFeature(scale=coast_details))

            # Done plotting, now adjusting
                        
            ax1.text(
                0,
                1,
                "{0}_Z500_{1}+{2:02d}".format(model, dt, leadtime),
                ha="left",
                va="bottom",
                transform=ax1.transAxes,
                color="dimgrey",
            )
            if legend:
                try:
                    ax_cb = adjustable_colorbar_cax(fig1, ax1)

                    cb = plt.colorbar(CF_prec, cax=ax_cb,fraction=0.046, pad=0.01, aspect=25, label =f"{acc}h acc. prec. [mm/{acc}h]", extend="both")

                except:
                    pass
                
                pressure_dim = list(
                    filter(re.compile(f"press*").match, dmet.__dict__.keys())
                )  # need to find the correcvt pressure name
                llg = {
                    "GP": {
                        "color": "blue",
                        "linestyle": "-",
                        "legend": f"Geopotential height [m] at {dmet.__dict__[pressure_dim[0]][plev]:.0f} hPa",
                    },
                    "MSLP": {"color": "gray", "linestyle": None, "legend": "MSLP [hPa]"},
                }
                nice_legend(llg, ax1)
            if grid:
                nicegrid(ax=ax1)
            if overlays:
                add_overlay(overlays,ax=ax1,col="red", **kwargs)

            print(data_domain.lonlat)  # [15.8, 16.4, 69.2, 69.4]
            if domain_name != model and data_domain != None:  #
                ax1.set_extent(data_domain.lonlat)

            # runid == "" if runid == None else runid
            # make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}-{1}".format(dt, runid))

            make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(dt))
            file_path = "{0}/{1}_{2}_{3}_{4}+{5:02d}.png".format(
                make_modelrun_folder, model, domain_name, "Z500", dt, leadtime
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


def Z500(datetime,use_latest, delta_index, coast_details, steps=0, model="MEPS", domain_name=None, domain_lonlat=None, legend=False, info=False, grid=True,
        runid=None, outpath=None, url=None, point_lonlat =None,overlays=None, point_name=None):
    param = [
        "air_pressure_at_sea_level",
        "precipitation_amount_acc",
        "surface_geopotential",
        "x_wind_pl",
        "y_wind_pl",
        "geopotential_pl"
        ]

    p_level = [500]
    print(datetime)
    
    plot_by_subdomains(plot_Z500,checkget_data_handler, datetime, steps, model, domain_name, domain_lonlat, legend,
                       info, grid, url, point_lonlat, use_latest,
                       delta_index, coast_details, param, p_level,overlays, runid, point_name)


if __name__ == "__main__":
    args = default_arguments()
    print(args.datetime)
    
    chunck_func_call(
            func=Z500,chunktype= args.chunktype, chunk=args.chunks, datetime=args.datetime, steps=args.steps, model=args.model,
            domain_name=args.domain_name, domain_lonlat=args.domain_lonlat, legend=args.legend, info=args.info, grid=args.grid, runid=args.id,
            outpath=args.outpath, use_latest=args.use_latest,delta_index=args.delta_index, coast_details=args.coast_details, url=args.url,
            point_lonlat =args.point_lonlat, overlays= args.overlays, point_name=args.point_name)
    gc.collect()
