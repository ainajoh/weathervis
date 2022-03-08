# %%
# python CAO.py --datetime 2020091000 --steps 0 1 --model MEPS --domain_name West_Norway

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
#from mpl_toolkits.axes_grid1 import make_axes_locatable ##__N
from weathervis.checkget_data_handler import *
import gc
from weathervis.plots.add_overlays import add_overlay

# suppress matplotlib warning
warnings.filterwarnings("ignore", category=UserWarning)


def plot_CAO(datetime, data_domain, dmet, steps=[0,2], coast_details="auto", model=None, domain_name=None,
             domain_lonlat=None, legend=True, info=False, grid=True,runid=None, outpath=None, url = None, save= True, overlays=None):
    eval(f"data_domain.{domain_name}()")  # get domain info
    ## CALCULATE AND INITIALISE ####################
    scale = data_domain.scale  # scale is larger for smaller domains in order to scale it up.
    MSLP = filter_values_over_mountain(dmet.surface_geopotential[:], dmet.air_pressure_at_sea_level[:]/100) #in hpa
    pt = potential_temperatur(dmet.air_temperature_pl, dmet.pressure*100)
    pt_sst = potential_temperatur(dmet.SST, dmet.air_pressure_at_sea_level)
    dpt_sst = pt_sst[:, :, :] - pt[:, np.where(dmet.pressure == 850)[0], :, :].squeeze()
    DELTAPT = np.where(dmet.SIC <= 0.99, dpt_sst, 0)
    SImask = np.where(dmet.SIC.squeeze() >= 0.1, dmet.SIC.squeeze(), np.NaN).squeeze()
    lvl = range(-1, 13)
    C = [[255, 255, 255],
         [204, 191, 189],
         [155, 132, 127],
         [118, 86, 80],
         [138, 109, 81],
         [181, 165, 102],
         [229, 226, 124],
         [213, 250, 128],
         [125, 231, 111],
         [55, 212, 95],
         [25, 184, 111],
         [17, 138, 234],
         [21, 82, 198],
         [37, 34, 137]]
    C = np.array(C)
    C = np.divide(C, 255.)  # RGB has to be between 0 and 1 in python
    # PLOTTING ROUTNE ######################
    crs = default_map_projection(dmet)
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9), subplot_kw={'projection': crs})
    itim = 0
    for leadtime in np.array(steps):
        print('Plotting {0} + {1:02d} UTC'.format(datetime, leadtime))
        ax1 = default_mslp_contour(dmet.x, dmet.y, MSLP[itim, 0, :, :], ax1, scale=scale)
        CF_prec = ax1.contourf(dmet.x, dmet.y, DELTAPT[itim,0,:,:], zorder=0,
                              antialiased=True, extend="max", levels=lvl, colors=C, vmin=0, vmax=12)  #
        ax1.contourf(dmet.x, dmet.y, SImask[itim, :, :], zorder=1, alpha=0.5, colors='azure')
        ax1.contour(dmet.x, dmet.y,
                             dmet.SIC[itim, :, :] if len(np.shape(dmet.SIC)) == 3 else dmet.SIC[itim,0, :, :],
                             zorder=2, linewidths=2.0, colors="black", levels=[0.1, 0.5])  #
        ax1.add_feature(cfeature.GSHHSFeature(scale=coast_details))

        if legend:
            proxy = [plt.axhline(y=0, xmin=1, xmax=1, color="grey"),
                    plt.axhline(y=0, xmin=1, xmax=1, color="black", linewidth=4)]
            try:
                ax_cb = adjustable_colorbar_cax(fig1, ax1)
                plt.colorbar(CF_prec, cax=ax_cb, fraction=0.046, pad=0.01, aspect=25,
                            label=r"$\theta_{SST}-\theta_{850}$", extend="both")
            except:
                pass
            lg = ax1.legend(proxy, [f"MSLP (hPa)", f"Sea ice at 10%, 80%, 99%"])
            frame = lg.get_frame()
            frame.set_facecolor('white')
            frame.set_alpha(1)
        if grid:
            nicegrid(ax=ax1)
        if overlays:
            add_overlay(overlays, ax=ax1)
        if domain_name != model and data_domain != None:
            ax1.set_extent(data_domain.lonlat)
        # save and clean ###############################################################
        make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}{1}".format(datetime, "-"+runid if runid!=None else ""))
        file_path = "{0}/{1}_{2}_{3}_{4}+{5:02d}.png".format(make_modelrun_folder, model, domain_name,"CAOi", datetime,leadtime)
        print(f"filename: {file_path}")
        fig1.savefig(file_path, bbox_inches="tight", dpi=200)
        ax1.cla()
        itim += 1
    plt.close(fig1)
    plt.close("all")

    del MSLP, scale, pt, pt_sst, dpt_sst, SImask, DELTAPT, lvl, C, itim, legend, grid, overlays, domain_name
    del dmet,data_domain
    del fig1, ax1, CF_prec, crs
    del make_modelrun_folder, file_path
    gc.collect()

def CAO(datetime,use_latest, delta_index, coast_details, steps=0, model="MEPS", domain_name=None, domain_lonlat=None, legend=False, info=False, grid=True,
        runid=None, outpath=None, url=None, point_lonlat =None,overlays=None):
    param= ["air_pressure_at_sea_level", "surface_geopotential", "air_temperature_pl", "SST", "SIC"]  # add later
    p_level = [850, 1000]
    plot_by_subdomains(plot_CAO, checkget_data_handler, datetime, steps, model, domain_name, domain_lonlat, legend,
                       info, grid, url, point_lonlat, use_latest,
                       delta_index, coast_details, param, p_level,overlays, runid)

if __name__ == "__main__":
    args = default_arguments()
    chunck_func_call(func = CAO, chunktype= args.chunktype, chunk=args.chunks, datetime=args.datetime, steps=args.steps, model=args.model,
            domain_name=args.domain_name, domain_lonlat=args.domain_lonlat, legend=args.legend, info=args.info, grid=args.grid, runid=args.id,
            outpath=args.outpath, use_latest=args.use_latest,delta_index=args.delta_index, coast_details=args.coast_details, url=args.url,
            point_lonlat =args.point_lonlat, overlays= args.overlays)
