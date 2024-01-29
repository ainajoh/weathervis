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


def plot_IVT_IWV(datetime, data_domain, dmet, steps=[0,2], coast_details="auto", model=None, domain_name=None,
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
    itim = 0
    for dt in datetime:
        for leadtime in np.array(steps):
            print("Plotting {0} + {1:02d} UTC".format(dt, leadtime))
            print(np.shape(MSLP))
            print(np.shape(default_mslp_contour))
            print(itim)
            
            fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9), subplot_kw={"projection": crs})
            
#--------------------IVT------------------------------------------
            # calculate IVT
            IVT = wind_speed(dmet.x_wind_ml[itim,:,:,:],dmet.y_wind_ml[itim,:,:,:])
            
            IVT = IVT * dmet.specific_humidity_ml[itim,:,:,:].squeeze()
            # use an average dp
            dp=np.insert(np.diff(np.mean(dmet.surface_air_pressure[itim,0,:,:].squeeze()) * dmet.b + dmet.ap),0,0)
            # calculate multiplication on axis without copying data
            shape = np.swapaxes(IVT, IVT.ndim-1, 0).shape
            B_brc = np.broadcast_to(dp, shape)
            # Swap back the axes. As before, this only changes our "point of view".
            B_brc = np.swapaxes(B_brc, IVT.ndim-1, 0)
            IVT = np.sum(IVT * B_brc,axis=0) / 9.81

            #CC=ax1.contourf(x,y,IVT.squeeze(),levels=[5,10,15,20,25,30,40,50,60,70,80,90,100,120,150,200],
            #                cmap='tab20c',vmin=1,vmax=200,zorder=2,extend='both')  # arctic, wintertime
            CC=ax1.contourf(x,
                            y,
                            IVT.squeeze(),
                            levels=[10,20,30,40,50,60,70,80,90,100,120,150,200,250,300,350],
                            cmap='tab20c',
                            vmin=10,
                            vmax=350,
                            zorder=1,
                            extend='both')  # mid-latitudes, summertime
            ax1 = default_mslp_contour(
                dmet.x, dmet.y, MSLP[itim,0, :, :], ax1, scale=scale
            )
            
            # coastline
            ax1.add_feature(cfeature.GSHHSFeature(scale="high"),edgecolor="black", linewidth=1)  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
            #ax1.add_feature(cfeature.GSHHSFeature(scale=coast_details))

            # Done plotting, now adjusting
            ax_cb = adjustable_colorbar_cax(fig1, ax1)
            
            ax1.text(
                0,
                1,
                "{0}_IVT_{1}+{2:02d}".format(model, dt, leadtime),
                ha="left",
                va="bottom",
                transform=ax1.transAxes,
                color="dimgrey",
            )
            if legend:
                plt.colorbar(CC,cax = ax_cb, fraction=0.046, pad=0.01, aspect=25,
                            label=r"IVT (kg m-1 s-1)")

                proxy = [plt.axhline(y=0, xmin=0, xmax=0, color="gray",zorder=7)]
                # proxy.extend(proxy1)
                # legend's location fixed, otherwise it takes very long to find optimal spot
                lg = ax1.legend(proxy, [f"MSLP (hPa)", f"Sea ice at 50%"])

                frame = lg.get_frame()
                frame.set_facecolor('white')
                frame.set_alpha(0.8)
            if grid:
                nicegrid(ax=ax1)
            if overlays:
                kwargs["col"] = ["red","red","red","red"]
                kwargs["size"] =80
                add_overlay(overlays,ax=ax1, **kwargs)

            print(data_domain.lonlat)  # [15.8, 16.4, 69.2, 69.4]
            if domain_name != model and data_domain != None:  #
                ax1.set_extent(data_domain.lonlat)

            # runid == "" if runid == None else runid
            # make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}-{1}".format(dt, runid))

            make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(dt))
            
            file_path = "{0}/{1}_{2}_{3}_{4}+{5:02d}.png".format(
                make_modelrun_folder, model, domain_name, "IVT", dt, leadtime
            )

            print(f"filename: {file_path}")
            if save:
                fig1.savefig(file_path, bbox_inches="tight", dpi=200)
            else:
                plt.show()
            ax1.cla()
            plt.close(fig1)
            plt.close("all")
            del ax_cb
    
#-----------------------IWV------------------------------------------
            fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9), subplot_kw={"projection": crs})

            # calculate IWV
            #SImask = np.where(dmet.SIC >= 0.5, dmet.SIC, np.NaN)
            ax1.contour(dmet.x, dmet.y,
                             dmet.SIC[itim, :, :] if len(np.shape(dmet.SIC)) == 3 else dmet.SIC[itim,0, :, :],
                             zorder=2, linewidths=2.0, colors="black", levels=[0.5])  #

            IWV=dmet.specific_humidity_ml[itim,:,:,:] #kg/kg
            #print(IWV[itim,100,100])
            #p(n,k,j,i) = ap(k) + b(k)*ps(n,j,i)
            shape = np.swapaxes(IWV, IWV.ndim-1, 0).shape
            B_brc = np.broadcast_to(dp, shape)
            # Swap back the axes. As before, this only changes our "point of view".
            B_brc = np.swapaxes(B_brc, IWV.ndim-1, 0)
            #print(B_brc)
            IWV = np.sum(IWV * B_brc,axis=0) / 9.81
            
            #CC=ax1.contourf(x,y,IWV.squeeze(),levels=[1.0,2.0,3.0,4.0,5,6,7,8,9,10,12.5,15],
                #                colors=('#FFFFFF','#ffffd9','#e0f3b2','#97d6b9','#41b6c4',
                #                        '#1f80b8','#24429b','#081d58','#da30da','#a520a5','#600060'),vmin=1,
                #                vmax=15,zorder=2,extend='both') 
            #levels=[5,6,7,8,9,10,12,14,17,20,23,26,30,40],

            CC=ax1.contourf(x,
                            y,
                            IWV.squeeze(),
                            levels=[2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                            colors=('#FFFFFF','#ffffd9','#e0f3b2','#97d6b9','#41b6c4',
                                    '#1f80b8','#24429b','#081d58','#fa80fa','#e550e5','#ca30ca','#a520a5','#600060'),
                            vmin=5,
                            vmax=45,
                            zorder=1,
                            extend='both'
                            ) 
            ax1 = default_mslp_contour(
                dmet.x, dmet.y, MSLP[itim,0, :, :], ax1, scale=scale
            )
            # coastline
            ax1.add_feature(cfeature.GSHHSFeature(scale="high"),edgecolor="black", linewidth=1)  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
            #ax1.add_feature(cfeature.GSHHSFeature(scale=coast_details))
            
            # Done plotting, now adjusting
            ax_cb = adjustable_colorbar_cax(fig1, ax1)
            
            ax1.text(
                0,
                1,
                "{0}_IWV_{1}+{2:02d}".format(model, dt, leadtime),
                ha="left",
                va="bottom",
                transform=ax1.transAxes,
                color="dimgrey",
            )
            if legend:
                plt.colorbar(CC,cax = ax_cb, fraction=0.046, pad=0.01, aspect=25,
                            label=r'IWV ($kg/m^2$)')

                #proxy = [plt.axhline(y=0, xmin=0, xmax=0, color="gray",zorder=7)]
                proxy = [plt.axhline(y=0, xmin=1, xmax=1, color="grey"),
                         plt.axhline(y=0, xmin=1, xmax=1, color="black", linewidth=2)]
                # proxy.extend(proxy1)
                # legend's location fixed, otherwise it takes very long to find optimal spot
                lg = ax1.legend(proxy, [f"MSLP (hPa)",f"Sea ice at 50%"])
                #lg = ax1.legend(proxy, [f"MSLP (hPa)", f"Sea ice at 50%"])
                frame = lg.get_frame()
                #frame.set_zorder(10)
                frame.set_facecolor('white')
                frame.set_alpha(1.0)
            if grid:
                nicegrid(ax=ax1)
            if overlays:
                kwargs["col"] = ["red","red","red","red"]
                kwargs["size"] =80
                add_overlay(overlays,ax=ax1,**kwargs)

            print(data_domain.lonlat)  # [15.8, 16.4, 69.2, 69.4]
            if domain_name != model and data_domain != None:  #
                ax1.set_extent(data_domain.lonlat)

            # runid == "" if runid == None else runid
            # make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}-{1}".format(dt, runid))

            make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(dt))
            
            file_path = "{0}/{1}_{2}_{3}_{4}+{5:02d}.png".format(
                make_modelrun_folder, model, domain_name, "IWV", dt, leadtime
            )

            print(f"filename: {file_path}")
            if save:
                fig1.savefig(file_path, bbox_inches="tight", dpi=200)
            else:
                plt.show()
            ax1.cla()
            plt.close(fig1)
            plt.close("all")
            del ax_cb
            itim += 1
    del MSLP, scale, itim, legend, grid, overlays, domain_name
    del dmet, data_domain
    del fig1, ax1, crs
    del make_modelrun_folder, file_path
    gc.collect()


def IVT_IWV(datetime,use_latest, delta_index, coast_details="auto", steps=0, model="MEPS", domain_name=None, domain_lonlat=None, legend=False, info=False, grid=True,
        runid=None, outpath=None, url=None, point_lonlat =None,overlays=None, point_name=None):
    param = [
        'specific_humidity_ml',
        'air_pressure_at_sea_level',
        'surface_geopotential',
        'surface_air_pressure',
        'x_wind_ml',
        'y_wind_ml',
        "SIC"
    ]

    
    p_level = None
    print(datetime)
    
    plot_by_subdomains(plot_IVT_IWV,checkget_data_handler, datetime, steps, model, domain_name, domain_lonlat, legend,
                       info, grid, url, point_lonlat, use_latest,
                       delta_index, coast_details, param, p_level,overlays, runid, point_name)


if __name__ == "__main__":
    args = default_arguments()
    print(args.datetime)
    
    chunck_func_call(
            func=IVT_IWV,chunktype= args.chunktype, chunk=args.chunks, datetime=args.datetime, steps=args.steps, model=args.model,
            domain_name=args.domain_name, domain_lonlat=args.domain_lonlat, legend=args.legend, info=args.info, grid=args.grid, runid=args.id,
            outpath=args.outpath, use_latest=args.use_latest,delta_index=args.delta_index, coast_details=args.coast_details, url=args.url,
            point_lonlat =args.point_lonlat, overlays= args.overlays, point_name=args.point_name)
    gc.collect()
