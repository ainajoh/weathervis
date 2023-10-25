#%%
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


def plot_DXS(datetime, data_domain, dmet, steps=[0,2], coast_details="auto", model=None, domain_name=None,
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
            
        
            print('Plotting d-excess {0} + {1:02d} UTC'.format(dt, itim))
            SI = dmet.SIC[itim, :, :].squeeze()
            SImask = np.where(SI >= 0.1, dmet.SIC[itim,:,:], np.NaN).squeeze()

            SST = dmet.SST[itim, :, :].squeeze() - 273.15
            es = 6.1094 * np.exp(17.625 * SST / (SST + 243.04))
            mslp = dmet.air_pressure_at_sea_level[itim, :, :].squeeze() / 100
            
            Q = dmet.specific_humidity_2m[itim, :, :].squeeze()
            Qs = 0.622 * es / (mslp - 0.37 * es)
            RH_2m = dmet.relative_humidity_2m[itim,:,:].squeeze()*100
            RH = Q / Qs * 100
            # print("RH: {}".format(RH_2m[0][0]))
            d = 48.2 - 0.54 * RH_2m

            
            # SST
            levels = np.arange(-10, 45, 5)
            cmap = plt.get_cmap("cividis_r")
            Cd = ax1.contourf(dmet.x, dmet.y, d, zorder=1, alpha=0.7, cmap=cmap, levels=levels, extend="both")
            Cd10 = ax1.contour(dmet.x, dmet.y, d, zorder=1, alpha=0.7, colors='tab:blue', levels=[10])
            SI = ax1.contourf(dmet.x, dmet.y, SImask, zorder=2, alpha=1, colors='azure')
            ax1.contour(dmet.x, dmet.y, SImask, zorder=2, alpha=1, colors='black', levels=[0.15], linestyles='--')
            
            # coastline
            ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'),edgecolor="black", linewidth=1)  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
            #ax1.add_feature(cfeature.GSHHSFeature(scale=coast_details))

            # Done plotting, now adjusting
            ax_cb = adjustable_colorbar_cax(fig1, ax1)
            cb = plt.colorbar(Cd, fraction=0.046, pad=0.01, ax=ax1, aspect=25, cax= ax_cb, label="d-excess ($\perthousand$)",
                                    extend="both")

            ax1.text(
                0,
                1,
                "{0}_DXS_{1}+{2:02d}".format(model, dt, leadtime),
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
                    "dxs": {"color": "tab:blue", 
                            "linestyle": None, 
                            "legend": "d-xs = 10"
                            },
                    "MSLP": {"color": "gray", 
                             "linestyle": None, 
                             "legend": "MSLP [hPa]"
                             },
                    "SIC": {"color": "black",
                            "linestyle": "dashed",
                            "legend": "Sea Ice conc. > 10%"
                            }
                }
                nice_legend(llg, ax1)
            if grid:
                nicegrid(ax=ax1)
            if overlays:
                add_overlay(overlays,ax=ax1, **kwargs)

            print(data_domain.lonlat)  # [15.8, 16.4, 69.2, 69.4]
            if domain_name != model and data_domain != None:  #
                ax1.set_extent(data_domain.lonlat)

            # runid == "" if runid == None else runid
            # make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}-{1}".format(dt, runid))

            make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(dt))
            file_path = "{0}/{1}_{2}_{3}_{4}+{5:02d}.png".format(
                make_modelrun_folder, model, domain_name, "DXS", dt, leadtime
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
    del MSLP, scale, itim, legend, grid, overlays, domain_name, ax_cb,
    del dmet, data_domain
    del fig1, ax1, crs,
    del make_modelrun_folder, file_path
    gc.collect()


def DXS(datetime,use_latest, delta_index, coast_details, steps=0, model="MEPS", domain_name=None, domain_lonlat=None, legend=False, info=False, grid=True,
        runid=None, outpath=None, url=None, point_lonlat =None,overlays=None, point_name=None):
    param = ["air_temperature_2m", 
             "specific_humidity_2m", 
             "air_pressure_at_sea_level", 
             "surface_geopotential",
             "SST",
             "SIC",
             "relative_humidity_2m"
             ]

    p_level = None
    print(datetime)
    
    plot_by_subdomains(plot_DXS,checkget_data_handler, datetime, steps, model, domain_name, domain_lonlat, legend,
                       info, grid, url, point_lonlat, use_latest,
                       delta_index, coast_details, param, p_level,overlays, runid, point_name)


if __name__ == "__main__":
    args = default_arguments()
    print(args.datetime)
    print("step:"+str(args.steps))

    
    chunck_func_call(
            func=DXS,chunktype= args.chunktype, chunk=args.chunks, datetime=args.datetime, steps=args.steps, model=args.model,
            domain_name=args.domain_name, domain_lonlat=args.domain_lonlat, legend=args.legend, info=args.info, grid=args.grid, runid=args.id,
            outpath=args.outpath, use_latest=args.use_latest,delta_index=args.delta_index, coast_details=args.coast_details, url=args.url,
            point_lonlat =args.point_lonlat, overlays= args.overlays, point_name=args.point_name)
    gc.collect()