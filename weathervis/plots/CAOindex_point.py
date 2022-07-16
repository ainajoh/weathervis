

from weathervis.config import *
from weathervis.plots.add_overlays import add_overlay

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
from weathervis.plots.CAOindex_map import plot_CAO

# suppress matplotlib warning
warnings.filterwarnings("ignore", category=UserWarning)


def plot_CAOindex_point(point, dmet, barb=True,fillbg=False, **args):
    rad = (len(dmet.x)/2)*2.5
    textinfo = f"{point.index.values[0]}; cord: ({point['lat'].values[0]}, {point['lon'].values[0]}), radius:{rad} km"
    print(textinfo)
    CAOi = CAO_index(dmet.air_temperature_pl,dmet.pressure,dmet.SST,dmet.air_pressure_at_sea_level, p_level=850)
    CAOialleval= ( CAOi != np.nan).sum(axis=(1,2))
    CAOiabove4 = ( CAOi >= 4).sum(axis=(1,2))
    CAOiabove8 = ( CAOi >= 8).sum(axis=(1,2))
    CAOiabove12= ( CAOi >= 12).sum(axis=(1,2))
    CAOiabove16= (CAOi >= 16).sum(axis=(1,2))

    pros_CAOabove4 = 100*(CAOiabove4/CAOialleval)
    pros_CAOabove8 = 100*(CAOiabove8/CAOialleval)
    pros_CAOabove12 = 100*(CAOiabove12/CAOialleval)
    pros_CAOabove16 = 100*(CAOiabove16/CAOialleval)
    
    mean_CAOi = CAOi.mean(axis= (1,2))   #-694.14347092 
    std_CAOi = CAOi.std(axis= (1,2))   #-694.14347092 
    time= pd.to_datetime(dmet.time, unit='s') #origin=pd.Timestamp('1960-01-01')
    
    #PLOTTING
    fig, ax = plt.subplots(figsize=(10,6)) #    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9), subplot_kw={'projection': crs})
    ax2=ax.twinx(); ax2.set_ylim(0,100)
    ax.set_ylabel("CAO index")
    ax2.set_ylabel('percentage of position above limits')
    ax.set_xlabel("Time [M-d HH")
    ax.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2
    ax.patch.set_visible(False) 
    ax.errorbar(time, mean_CAOi,std_CAOi,linestyle='None', marker='^', fillstyle="full",color="k", label=f"CAO index for: \n{textinfo}")
    ax.legend(loc='upper left')
    colors = ["#00AEAD", "#019875","#72CC50","#BFD834"]
    if barb == True: 
        width=0.03
        print(pros_CAOabove8)
        if (pros_CAOabove8!=pros_CAOabove4).any(): a4 = ax2.bar(time, pros_CAOabove4, width, color=colors[0], alpha= 0.9, align='center',label='>4')
        if (pros_CAOabove12!=pros_CAOabove8).any(): a3 = ax2.bar(time, pros_CAOabove8, width, color=colors[1], alpha= 0.9, align='center',label='>8')
        if (pros_CAOabove16!=pros_CAOabove12).any():a2 = ax2.bar(time, pros_CAOabove12,width, color=colors[2], alpha= 0.9, align='center',label='>12')
        a1 = ax2.bar(time, pros_CAOabove16,width, color=colors[3], alpha= 0.9, align='center',label='>16')
        ax2.legend(loc='upper right')
    elif fillbg==True:
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib as mpl
        a4 = ax2.fill_between(time, pros_CAOabove4,0, color=colors[0], alpha= 0.9)
        a3 = ax2.fill_between(time, pros_CAOabove8,0, color=colors[1], alpha= 0.9)
        a2 = ax2.fill_between(time, pros_CAOabove12,0, color=colors[2], alpha= 0.9)
        a1 = ax2.fill_between(time, pros_CAOabove16,0, color=colors[3], alpha= 0.9)
        cycles=np.linspace(0,0.75,4)
        cm = LinearSegmentedColormap.from_list('defcol', ["#00AEAD", "#019875", "#72CC50","#BFD834" ], N=4)
        ax3 = fig.add_axes([0.15, 0.9, 0.7, 0.05]) #left, bottom, width, height] 
        ax3.set_zorder(ax.get_zorder()+1) # put ax in front of ax2
        cb = mpl.colorbar.ColorbarBase(ax3, cmap=cm, ticks=cycles,orientation="horizontal")
        cb.ax.set_xticklabels(['>4', '>8', '>12','>16'], zorder=100)

    #figname = f"simplehist_{id}_{config.figname_base}.png"
    #plt.savefig(f"{config.path_fig}{figname}")
    make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(args["datetime"]))
    file_path = "{0}/{1}_{2}_{3}_{4}.png".format(
                make_modelrun_folder, args["model"], args["point_name"][0], "CAOi_point", args["datetime"])
    print(f"filename: {file_path}")
    fig.savefig(file_path, bbox_inches="tight", dpi=200)
    plt.cla()


def plot_CAOindex_map(**kwargs ):
    kwargs["domain_name"]="Svalbard"
    kwargs["overlays"]=["point_name"]
    kwargs["point_name"]=["CAO1"]
    plot_CAO(**kwargs)


def CAO(datetime,use_latest, delta_index, coast_details, steps=0, model="MEPS", domain_name=[None], domain_lonlat=None, legend=False, info=False, grid=True,
        runid=None, outpath=None, url=None, point_lonlat =None,overlays=None, point_name=None, num_point=1):
    param= ["air_pressure_at_sea_level", "surface_geopotential", "air_temperature_pl", "SST", "SIC"]  # add later
    p_level = [850, 1000]
    domain_name= None
    sites = pd.read_csv("../data/sites.csv", sep=";", header=0, index_col=0)
    point = sites.loc[point_name]
    #lonlat = [sites.loc[point_name].lon, sites.loc[point_name].lat]
    dmet, data_domain, bad_param = checkget_data_handler(p_level=p_level, model=model, step=steps, date=datetime,
                                                             domain_name=domain_name, point_name= point_name,
                                                              all_param=param, num_point=1000)
    print(np.shape(dmet.air_pressure_at_sea_level))
    #plot_CAOindex_map(datetime=datetime, dmet=dmet, data_domain=data_domain, steps=steps, model=model, domain_name=domain_name,
    #                     domain_lonlat=domain_lonlat, legend=legend, info=info, grid=grid, url=url,
    #                     overlays=overlays, point_name=point_name)

    plot_CAOindex_point(point=point, datetime=datetime, dmet=dmet, data_domain=data_domain, steps=steps, model=model, domain_name=domain_name,
                         domain_lonlat=domain_lonlat, legend=legend, info=info, grid=grid, url=url,
                         overlays=overlays, point_name=point_name)

if __name__ == "__main__":
    # python CAOindex_point.py --datetime 2020031300 --point_name CAO1 --model AromeArctic
    args = default_arguments()
    chunck_func_call(func = CAO, chunktype= args.chunktype, chunk=args.chunks, datetime=args.datetime, steps=args.steps, model=args.model,
            domain_name=args.domain_name, domain_lonlat=args.domain_lonlat, legend=args.legend, info=args.info, grid=args.grid, runid=args.id,
            outpath=args.outpath, use_latest=args.use_latest,delta_index=args.delta_index, coast_details=args.coast_details, url=args.url,
            point_lonlat =args.point_lonlat, overlays= args.overlays, point_name=args.point_name, num_point=args.num_point)