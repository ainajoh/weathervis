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
from matplotlib.legend import Legend
from mpl_toolkits.axes_grid1 import make_axes_locatable  ##__N
from weathervis.checkget_data_handler import checkget_data_handler
from weathervis.plots.add_overlays_new import add_overlay

import warnings
import gc

warnings.filterwarnings("ignore", category=UserWarning)  # suppress matplotlib warning


def plot_BT(datetime, data_domain, dmet, steps=[0,2], coast_details="auto", model=None, domain_name=None,
             domain_lonlat=None, legend=True, info=False, grid=True,runid=None, outpath=None, url = None, save= True, overlays=None, **kwargs):

    eval(f"data_domain.{domain_name}()")  # get domain info
    ## CALCULATE AND INITIALISE ####################

    scale = (
        data_domain.scale
    )  # scale is larger for smaller domains in order to scale it up.
    #dmet.air_pressure_at_sea_level /= 100
    
    #MSLP = filter_values_over_mountain(
    #    dmet.surface_geopotential[:], dmet.air_pressure_at_sea_level[:]
    #)
    
    # can be replaced at some time with proper height at model levels, but works for now        
    H = [24122.6894480669, 20139.2203688489,17982.7817599549, 16441.7123200128,
    15221.9607620438, 14201.9513633491, 13318.7065659522, 12535.0423836784,
    11827.0150898454, 11178.2217936245, 10575.9136768674, 10010.4629764989,
    9476.39726730647, 8970.49319005479, 8490.10422494626, 8033.03285976169,
    7597.43079283063, 7181.72764002209, 6784.57860867911, 6404.82538606181,
    6041.46303718354, 5693.61312218488, 5360.50697368367, 5041.46826162131,
    4735.90067455394, 4443.27792224573, 4163.13322354697, 3895.05391218293,
    3638.67526925036, 3393.67546498291, 3159.77069480894, 2936.71247430545,
    2724.28467132991, 2522.30099074027, 2330.60301601882, 2149.05819142430,
    1977.55945557602, 1816.02297530686, 1664.38790901915, 1522.61641562609,
    1390.69217292080, 1268.36594816526, 1154.95528687548, 1049.75817760629,
    952.260196563843, 861.980320753114, 778.466725603312, 701.292884739207,
    630.053985133223, 564.363722589458, 503.851644277509, 448.161118360263,
    396.946085973573, 349.869544871297, 306.601457634038, 266.817025119099,
    230.194566908004, 196.413229972062, 165.151934080260, 136.086183243070,
    108.885366240509, 83.2097562375566,58.7032686584901, 34.9801888163106,
    11.6284723290378]
    
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
            #print(np.shape(MSLP))
            #print(np.shape(default_mslp_contour))
            print(itim)
            #ax1 = default_mslp_contour(
            #    dmet.x, dmet.y, MSLP[itim,0, :, :], ax1, scale=scale
            #)
        
            ZS = dmet.surface_geopotential[itim, 0, :, :]
            CT   = dmet.cloud_top_altitude[itim,0,:,:].copy()
            CB   = dmet.cloud_base_altitude[itim,0,:,:].copy()
            CB[np.where(CB > 20000)] = np.nan # get rid of lage fill values
            # take three cloud level intervalls, to idicate by markers
            # level are 0-1000m, 1000-2000m, 2000-3000m 
            CB1 = CB.copy()
            CB2 = CB.copy()
            CB3 = CB.copy() 
            CB1[np.where(CB>1000)]        = np.nan
            CB2[np.where(~np.isnan(CB1))] = np.nan # no double counting
            CB2[np.where(CB>2000)]        = np.nan
            CB3[np.where(~np.isnan(CB1))] = np.nan # no double counting
            CB3[np.where(~np.isnan(CB2))] = np.nan # no double counting
            CB3[np.where(CB>3000)]        = np.nan
            xxx = CT.shape[0]
            yyy = CT.shape[1]
            HH = np.zeros([65,xxx,yyy])
            for i,z in enumerate(H):
                HH[i,:,:] = z + dmet.surface_geopotential[itim,0,:,:]/9.81
            
            caf3 = dmet.cloud_area_fraction_ml[itim,:,:,:].copy()
            caf3 = caf3[14:65,:,:] # only the levels wie are interested in
            buf = np.zeros(caf3.shape)
            thresh = 0.43 # threshold for cloud ..kind of aligns with MET values
            buf[np.where(caf3>=thresh)] = 1  
            HH3 = HH[14:65,:,:]
            ### this can for sure be improved!
            CT_new = np.zeros([xxx,yyy])
            for z in range(HH3.shape[0]-1,0,-1):# go from lowest to topmost and overwrite values
                xx,yy = np.where(buf[z,:,:]==1)
                CT_new[xx,yy] = HH3[z,xx,yy]
            CT_new[np.where(CT_new==0)] = np.nan # filter out cloud free areas
            # plot 
            #CT2 = CT.copy()
            # set all cloud tops above 14000m to nan, choose 14000 to align more with LMH plot
            #CT2[np.where(CT2>14000)] = np.nan
            #data =  CT2[:nx - 1, :ny - 1].copy()
            
            # indicate possible cells by latent heat, and plot below color patches
            cells = np.zeros(np.shape(caf3[0,:,:]))
            cells[np.where(dmet.SFX_LE[itim,:,:]>100)] = 1
            cells[np.where(cells==0)] = np.nan 


            data =  CT_new[:nx - 1, :ny - 1].copy()
            data[mask] = np.nan
            # making a contour for cirrus clouds, make a rough estimate of 0.5
            highC = dmet.high_type_cloud_area_fraction[itim,0,:,:]
            highC[np.where(highC>0.5)] = 1
            highC[np.where(highC<=0.5)] = 0
            cmap = plt.cm.get_cmap('rainbow_r', 9)
            cmap.set_over('lightgrey')
            CCl   = ax1.pcolormesh(x, y,  data[:, :], cmap=cmap,
                                    vmin=0, vmax=9000,zorder=1)
            # indicate cloud base height by markers
            co = '#393939'
            skip = (slice(10, None, 20), slice(10, None, 20))
            xx = x.copy()
            yy = y.copy()
            xx[np.where(np.isnan(CB1))] = np.nan
            yy[np.where(np.isnan(CB1))] = np.nan
            sc1 = ax1.scatter(xx[skip], yy[skip], s=30, zorder=2, marker='o', linewidths=0.9,
                                c=co, alpha=0.75)
            xx = x.copy()
            yy = y.copy()
            xx[np.where(np.isnan(CB2))] = np.nan
            yy[np.where(np.isnan(CB2))] = np.nan
            sc2 = ax1.scatter(xx[skip], yy[skip], s=30, zorder=2, marker='o', linewidths=0.9,
                                facecolors='none',edgecolors=co, alpha=0.75)
            xx = x.copy()
            yy = y.copy()
            xx[np.where(np.isnan(CB3))] = np.nan
            yy[np.where(np.isnan(CB3))] = np.nan
            sc3 = ax1.scatter(xx[skip], yy[skip], s=30, zorder=2, marker='x', linewidths=0.9,
                    c=co, alpha=0.75)

            # indicate possible subgrid cellular convection with hatched contour
            cel = ax1.contourf(dmet.x,dmet.y,cells,colors='none',hatches=['////',None])

            # instead plot a contour of high cloud cover for clarity of plot
            ax1.contour(x, y, highC, colors='grey',zorder=2)
            
            
            # coastline
            ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'),edgecolor="black", linewidth=1,zorder=3)  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
            #ax1.add_feature(cfeature.GSHHSFeature(scale=coast_details))

            # Done plotting, now adjusting
            ax_cb = adjustable_colorbar_cax(fig1, ax1)
            
            ax1.text(
                0,
                1,
                "{0}_CB_CT_{1}+{2:02d}".format(model, dt, leadtime),
                ha="left",
                va="bottom",
                transform=ax1.transAxes,
                color="dimgrey",
            )
            if legend:
                '''
                llg = {
                    
                    "MSLP": {"color": "gray", "linestyle": None, "legend": "MSLP [hPa]"},
                }
                nice_legend(llg, ax1)
                '''
                artists,labels = cel.legend_elements()
                leg = Legend(ax1,[sc1,sc2,sc3,artists[0]],['[0m, 1000m]',
                                                           '[1000m, 2000m]',
                                                           '[2000m, 3000m]',
                                                           'subgrid cells'],
                            loc='upper left', facecolor='white',fontsize=10)
                leg.set_alpha(1)
                ax1.add_artist(leg)
                plt.colorbar(CCl,cax = ax_cb, fraction=0.046, pad=0.01, aspect=25,
                                 label=r"cloud top height (m)",extend='max')
                l1 = ax1.legend(loc='upper left')
                frame = l1.get_frame()
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
                make_modelrun_folder, model, domain_name, "CB_CT", dt, leadtime
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
    del scale, itim, legend, grid, overlays, domain_name, ax_cb#, MSLP
    del dmet, data_domain
    del fig1, ax1, crs
    del make_modelrun_folder, file_path
    gc.collect()


def BT(datetime,use_latest, delta_index, coast_details, steps=0, model="MEPS", domain_name=None, domain_lonlat=None, legend=False, info=False, grid=True,
        runid=None, outpath=None, url=None, point_lonlat =None,overlays=None, point_name=None):
    param = ['cloud_base_altitude',
             'cloud_top_altitude',
             #'air_pressure_at_sea_level',
             'surface_geopotential',
             'cloud_area_fraction_ml',
             'high_type_cloud_area_fraction',
             'SFX_LE'
             ]

    
    p_level = None
    print(datetime)
    
    plot_by_subdomains(plot_BT,checkget_data_handler, datetime, steps, model, domain_name, domain_lonlat, legend,
                       info, grid, url, point_lonlat, use_latest,
                       delta_index, coast_details, param, p_level,overlays, runid, point_name)


if __name__ == "__main__":
    args = default_arguments()
    print(args.datetime)
    
    chunck_func_call(
            func=BT,chunktype= args.chunktype, chunk=args.chunks, datetime=args.datetime, steps=args.steps, model=args.model,
            domain_name=args.domain_name, domain_lonlat=args.domain_lonlat, legend=args.legend, info=args.info, grid=args.grid, runid=args.id,
            outpath=args.outpath, use_latest=args.use_latest,delta_index=args.delta_index, coast_details=args.coast_details, url=args.url,
            point_lonlat =args.point_lonlat, overlays= args.overlays, point_name=args.point_name)
    gc.collect()
