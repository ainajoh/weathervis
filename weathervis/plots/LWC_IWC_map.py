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
    setup_directory,
    nice_vprof_colorbar
)

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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


def plot_LWC_IWC(datetime, data_domain, dmet, steps=[0,2], coast_details="auto", model=None, domain_name=None,
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

    x,y = np.meshgrid(dmet.x, dmet.y)
        #dlon,dlat=  np.meshgrid(dmet.longitude, dmet.latitude)

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
            
            
            # calculate LWC
            LWC=dmet.mass_fraction_of_cloud_condensed_water_in_air_ml[itim,:,:,:]
            #p(n,k,j,i) = ap(k) + b(k)*ps(n,j,i)
            dp=np.insert(np.diff(np.mean(dmet.surface_air_pressure[itim,0,:,:].squeeze()) * dmet.b + dmet.ap),0,0)
            shape = np.swapaxes(LWC, LWC.ndim-1, 0).shape
            B_brc = np.broadcast_to(dp, shape)
            # Swap back the axes. As before, this only changes our "point of view".
            B_brc = np.swapaxes(B_brc, LWC.ndim-1, 0)
            LWC = np.sum(LWC * B_brc,axis=0) / 9.81

            IWC=dmet.mass_fraction_of_cloud_ice_in_air_ml[itim,:,:,:]
            shape = np.swapaxes(IWC, IWC.ndim-1, 0).shape
            B_brc = np.broadcast_to(dp, shape)
            # Swap back the axes. As before, this only changes our "point of view".
            B_brc = np.swapaxes(B_brc, IWC.ndim-1, 0)
            IWC = np.sum(IWC * B_brc,axis=0) / 9.81

            #dmet.LWC = np.sum(dmet.mass_fraction_of_cloud_condensed_water_in_air_ml[:,:,:,:],axis=1)
            #dmet.LWC =dmet.LWC*1000
            dmet.units.LWC = "mm"
            #dmet.IWC = np.sum(dmet.mass_fraction_of_cloud_ice_in_air_ml[:,:,:,:],axis=1)
            #dmet.IWC = dmet.IWC*1000
            dmet.units.IWC = "mm"

            # for more normal unit of g/m^2 read
            # #https://www.nwpsaf.eu/site/download/documentation/rtm/docs_rttov12/rttov_gas_cloud_aerosol_units.pdf
            # https://www.researchgate.net/post/How-to-convert-the-units-of-specific-cloud-liquid-water-from-ERA5-kg-kg-to-kg-m2
            #print('LWC')
            #print(np.max(LWC))
            #print('IWC')
            #print(np.max(IWC))
            #LWC[np.where(LWC < 0.002)] = np.nan # arctic, wintertime
            #IWC[np.where(IWC < 0.0005)] = np.nan
            LWC[np.where(LWC < 0.005)] = np.nan # mid-latitudes, summertime
            IWC[np.where(IWC < 0.001)] = np.nan

            ZS = dmet.surface_geopotential[itim, 0, :, :]

            #pcolor as pcolormesh and  this projection is not happy together. If u want faster, try imshow
            #dmet.LWC[np.where( dmet.LWC <= 0.09)] = np.nan
            ldata =  LWC[:nx - 1, :ny - 1].copy()
            ldata[mask] = np.nan
            idata =  IWC[:nx - 1, :ny - 1].copy()
            idata[mask] = np.nan


            #CI= ax1.pcolormesh(x, y, idata[:, :], cmap=plt.cm.Blues, alpha=1, vmin=0.0005, vmax=0.04,zorder=1)
            #CC=ax1.pcolormesh(x, y,  ldata[:, :], cmap=plt.cm.Reds, vmin=0.002, vmax=0.1,zorder=2) # for arctic, wintertime
            CC=ax1.pcolormesh(x, 
                            y, 
                            ldata[:, :], 
                            cmap=plt.cm.Reds,
                            alpha=1.0,
                            vmin=0.005,
                            vmax=2.5,
                            zorder=1,
                            )
            #IWC[np.where(IWC > 0.005)] = np.nan # only for Arctic
            #CI= ax1.pcolormesh(x, y, idata[:, :], cmap=plt.cm.Blues,alpha=0.3, vmin=0.0005, vmax=0.04,zorder=3) # for arctic, wintertime
            CI= ax1.pcolormesh(x, 
                             y, 
                             idata[:, :], 
                             cmap=plt.cm.Blues,
                             alpha=0.4,
                             vmin=0.001,
                             vmax=0.01,
                             zorder=1,
                            )
            
            ax1 = default_mslp_contour(
                dmet.x, dmet.y, MSLP[itim,0, :, :], ax1, scale=scale
            )

            # coastline
            ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'),edgecolor="black", linewidth=1)  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
            #ax1.add_feature(cfeature.GSHHSFeature(scale=coast_details))

            # Done plotting, now adjusting
            #ax_cb = adjustable_colorbar_cax(fig1, ax1)
            
            ax1.text(0, 
                     1, 
                     "{0}_LWP_IWP__{1}+{2:02d}".format(model, dt, leadtime), 
                     ha='left', 
                     va='bottom', 
                     transform=ax1.transAxes, 
                     color='dimgrey')
            if legend:
                llg = {
                    
                    "MSLP": {"color": "gray", "linestyle": None, "legend": "MSLP [hPa]"},
                }
                #nice_legend(llg,ax1)

                
                cbar = nice_vprof_colorbar(CI, ax=ax1, extend="max",label='IWP (mm)',x0=0.75,y0=0.95,width=0.26,height=0.05,
                                            format='%.3f', ticks=[0.001, np.nanmax(IWC[:, :])*0.8])
                
                
                cbar = nice_vprof_colorbar(CF=CC, ax=ax1, extend="max", label='LWP (mm)',x0=0.50,y0=0.95,width=0.26,height=0.05,
                                            format='%.1f',ticks=[0.09, np.nanmax(LWC[:, :])*0.8])
                '''
                dat = np.random.rand(100,50)
                r=ax1.pcolormesh(dat)
                cbar  = nice_vprof_colorbar(r,ax1,extend="max",label="IWP (mm)",x0=0.75,y0=0.95,width=0.26,height=0.05,
                                        format='%.3f', ticks=[0.001, np.nanmax(dat)*0.8])
                '''                
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
                add_overlay(overlays,ax=ax1,col="red", **kwargs)

            print(data_domain.lonlat)  # [15.8, 16.4, 69.2, 69.4]
            if domain_name != model and data_domain != None:  #
                ax1.set_extent(data_domain.lonlat)

            # runid == "" if runid == None else runid
            # make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}-{1}".format(dt, runid))

            make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(dt))
            file_path = "{0}/{1}_{2}_{3}_{4}+{5:02d}.png".format(
                make_modelrun_folder, model, domain_name, "LWC_IWC", dt, leadtime
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
    del MSLP, scale, itim, legend, grid, overlays, domain_name#, ax_cb
    del dmet, data_domain
    del fig1, ax1, crs
    del make_modelrun_folder, file_path
    gc.collect()


def LWC_IWC(datetime,use_latest, delta_index, coast_details, steps=0, model="MEPS", domain_name=None, domain_lonlat=None, legend=False, info=False, grid=True,
        runid=None, outpath=None, url=None, point_lonlat =None,overlays=None, point_name=None):
    param = ['mass_fraction_of_cloud_condensed_water_in_air_ml',
             'surface_air_pressure',             
             'mass_fraction_of_cloud_ice_in_air_ml',
             "air_pressure_at_sea_level",
             "surface_geopotential"
            ]
        
    
    p_level = [850]
    print(datetime)
    
    plot_by_subdomains(plot_LWC_IWC,checkget_data_handler, datetime, steps, model, domain_name, domain_lonlat, legend,
                       info, grid, url, point_lonlat, use_latest,
                       delta_index, coast_details, param, p_level,overlays, runid, point_name,save=False,read_from_saved=False)


if __name__ == "__main__":
    args = default_arguments()
    print(args.datetime)
    
    chunck_func_call(
            func=LWC_IWC,chunktype= args.chunktype, chunk=args.chunks, datetime=args.datetime, steps=args.steps, model=args.model,
            domain_name=args.domain_name, domain_lonlat=args.domain_lonlat, legend=args.legend, info=args.info, grid=args.grid, runid=args.id,
            outpath=args.outpath, use_latest=args.use_latest,delta_index=args.delta_index, coast_details=args.coast_details, url=args.url,
            point_lonlat =args.point_lonlat, overlays= args.overlays, point_name=args.point_name)
    gc.collect()
