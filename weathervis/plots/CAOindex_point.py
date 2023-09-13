

from weathervis.config import *
from weathervis.plots.add_overlays import add_overlay
from matplotlib.axis import Axis
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
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
from weathervis.plots.Domain_map import plot_domain
import matplotlib.dates as mdates
plt.rcParams.update({'font.size': 17})
plt.rcParams["lines.antialiased"]=False
# suppress matplotlib warning
warnings.filterwarnings("ignore", category=UserWarning)


def plot_CAOindex_point(point, dmet, barb=False,fillbg=True, **args):
    #######CALCULATE VARIABLES TO PLOT###########################################
    rad = (len(dmet.x)/2)*2.5
    textinfo = f"{point.name}; cord: ({point['lat']}, {point['lon']}), radius:{rad} km"
    CAOi_all = CAO_index(dmet.air_temperature_pl,dmet.pressure,dmet.SST,dmet.air_pressure_at_sea_level, p_level=850).squeeze()
    CAOi = np.where(dmet.SIC <= 0.1, CAOi_all, np.NaN)
    CAOi = np.where(dmet.land_area_fraction.squeeze() == 0, CAOi, np.NaN)
    CAOi = np.where(CAOi<-100,np.nan,  CAOi)
    SImask = np.where(dmet.SIC >= 0.1, dmet.SIC, np.NaN)
    try: 
         CAOi= CAOi.squeeze(axis=1)
         SImask= SImask.squeeze(axis=1)
    except: 
        SImask = SImask
        SImask = SImask

    CAOialleval= np.nansum(CAOi >-9999, axis=(1,2))
    CAOiabove4 = np.nansum(CAOi >= 4, axis=(1,2))
    CAOiabove8 = np.nansum(CAOi >= 8,axis=(1,2))
    CAOiabove12= np.nansum(CAOi >= 12,axis=(1,2))
    CAOiabove16= np.nansum(CAOi >= 16,axis=(1,2))

    pros_CAOabove4 = 100*(CAOiabove4/CAOialleval)
    pros_CAOabove8 = 100*(CAOiabove8/CAOialleval)
    pros_CAOabove12 = 100*(CAOiabove12/CAOialleval)
    pros_CAOabove16 = 100*(CAOiabove16/CAOialleval)
    
    mean_CAOi = np.nanmean(CAOi, axis= (1,2))
    std_CAOi = np.nanstd(CAOi, axis= (1,2))  
    min_CAOi = np.nanmin(CAOi, axis=(1,2))
    max_CAOi = np.nanmax(CAOi, axis=(1,2))
    time= pd.to_datetime(dmet.time, unit='s') #origin=pd.Timestamp('1960-01-01')
    diff = mean_CAOi-10
    cao_auto= time[np.where(diff<0.5)]
    print(cao_auto)
    print(mean_CAOi[np.where(diff<0.5)])
    a1 = pd.Series(mean_CAOi)
    a2= pd.Series(time)
    a3=pd.concat([a1,a2], axis=1)
    pd.DataFrame(a3).to_csv("mean_CAOi.csv")

    #exit(1)
    #CAO periods
    #CAO1_time= time[np.logical_and(time>='2020-02-25 00:00:00', time<='2020-02-29 00:00:00')]
    #CAO2_time= time[np.logical_and(time>='2020-03-01 12:00:00', time<='2020-03-03 12:00:00')]
    #WAI_time = time[np.logical_and(time>='2020-03-03 12:00:00', time<='2020-03-08 00:00:00')]
    #CAO3_time= time[np.logical_and(time>='2020-03-09 12:00:00', time<='2020-03-12 00:00:00')]
    #CAO4_time= time[np.logical_and(time>='2020-03-12 00:00:00', time<='2020-03-15 00:00:00')]
    #CAO periods #2
    CAO1_time= time[np.logical_and(time>='2020-02-24 20:00:00', time<='2020-02-29 12:00:00')]
    CAO2_time= time[np.logical_and(time>='2020-02-29 12:00:00', time<='2020-03-03 14:00:00')]
    WAI_time = time[np.logical_and(time>='2020-03-03 14:00:00', time<='2020-03-08 18:00:00')]
    #CAO3_time=[]
    CAO3_time= time[np.logical_and(time>='2020-03-08 18:00:00', time<='2020-03-12 00:00:00')]
    CAO4_time= time[np.logical_and(time>='2020-03-12 00:00:00', time<='2020-03-16 00:00:00')]
    #CAO4_time= time[np.logical_and(time>='2020-03-08 18:00:00', time<='2020-03-16 00:00:00')]
    ######################################################################################

    ################ PLOTTING ####################################
    fig, ax = plt.subplots(figsize=(15,8)) #    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9), subplot_kw={'projection': crs})

    ax.set_ylabel(r"CAO index / K", color="limegreen")
    ax.yaxis.label.set_color('limegreen')
    ax.tick_params(axis='y', colors='limegreen')
    ax.patch.set_visible(False) 
    #ax.errorbar(time, mean_CAOi,[mean_CAOi - min_CAOi, max_CAOi - mean_CAOi],linestyle='None', ecolor='k', marker='^', fillstyle="full",color="red", label=f"CAO index for: \n{textinfo}")
    
    ax.plot(time, mean_CAOi, color="limegreen", label=f"CAO index for: \n{textinfo}", linewidth=6, zorder=100)
    ax.hlines(y=10, xmin=time[0], xmax=time[-1], linewidth=3, color="limegreen", linestyle="--")#,zorder=100)

    ax.set_xticks(time)
    ax.set_xticklabels(time,rotation=90, fontsize=16)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))

    ax2=ax.twinx()
    ax2.set_ylim(0,100)
    ax2.set_ylabel('Sea area %')

    ax4=ax.twinx(); #for the CAO periods
    ax4.set_ylim(0,100) #will not show just for adjusting.
    #horsizontal colors
    if len(CAO1_time) != 0: ax4.hlines(y=100, xmin=CAO1_time[0], xmax=CAO1_time[-1], linewidth=8, color='blue', zorder=ax.get_zorder()+1) 
    if len(CAO2_time) != 0: ax4.hlines(y=100, xmin=CAO2_time[0], xmax=CAO2_time[-1], linewidth=8, color='k', zorder=ax.get_zorder()+1)
    if len(WAI_time) != 0: ax4.hlines(y=100, xmin=WAI_time[0], xmax=WAI_time[-1], linewidth=8, color='red', zorder=ax.get_zorder()+1)
    if len(CAO3_time) != 0: ax4.hlines(y=100, xmin=CAO3_time[0], xmax=CAO3_time[-1], linewidth=8, color='k', zorder=ax.get_zorder()+1)
    if len(CAO4_time) != 0: ax4.hlines(y=100, xmin=CAO4_time[0], xmax=CAO4_time[-1], linewidth=8, color='blue', zorder=ax.get_zorder()+1)
    #alternatives
    #if len(CAO1_time) != 0: ax4.plot(CAO1_time,np.full(np.shape(CAO1_time), 100), linewidth=8, color='blue', zorder=100)
    #if len(CAO2_time) != 0: ax4.plot(CAO2_time,np.full(np.shape(CAO2_time), 100), linewidth=8, color='blue', zorder=100)
    #if len(WAI_time) != 0: ax4.plot(WAI_time,np.full(np.shape(WAI_time), 100), linewidth=8, color='red', zorder=100)
    #if len(CAO3_time) != 0: ax4.plot(CAO3_time,np.full(np.shape(CAO3_time), 100), linewidth=8, color='k', zorder=100)
    #if len(CAO4_time) != 0: ax4.plot(CAO4_time,np.full(np.shape(CAO4_time), 100), linewidth=8, color='blue', zorder=100)

    #vertical dashed lines
    if len(CAO1_time) != 0: ax2.plot(np.full(5,CAO1_time[0]),np.linspace(0, 100, 5), linestyle="--", linewidth=2, color='k')#, zorder=1)
    if len(CAO1_time) != 0: ax2.plot(np.full(5,CAO1_time[-1]),np.linspace(0, 100, 5), linestyle="--", linewidth=2, color='k')#, zorder=1)
    #if len(CAO2_time) != 0 and CAO1_time[-1] !=CAO2_time[0]: ax2.plot(np.full(5,CAO2_time[0]),np.linspace(0, 100, 5), linestyle="--", linewidth=2, color='k')#, zorder=1)
    if len(CAO2_time) != 0: ax2.plot(np.full(5,CAO2_time[-1]),np.linspace(0, 100, 5), linestyle="--", linewidth=2, color='k')#,# zorder=1)
    #if len(WAI_time) != 0 and CAO2_time[-1] != WAI_time[0] : ax2.plot(np.full(5,WAI_time[0]),np.linspace(0, 100, 5), linestyle="--", linewidth=2, color='k')#, zorder=1)
    if len(WAI_time) != 0: ax2.plot(np.full(5,WAI_time[-1]),np.linspace(0, 100, 5), linestyle="--", linewidth=2, color='k')#, zorder=1)
    #if len(CAO3_time) != 0 and WAI_time[-1] != CAO3_time[0]: ax2.plot(np.full(5,CAO3_time[0]),np.linspace(0, 100, 5), linestyle="--", linewidth=2, color='k')#, zorder=1)
    if len(CAO3_time) != 0: ax2.plot(np.full(5,CAO3_time[-1]),np.linspace(0, 100, 5), linestyle="--", linewidth=2, color='k')#, zorder=1)
    #if len(CAO4_time) != 0 and WAI_time[-1] != CAO4_time[0]: ax2.plot(np.full(5,CAO4_time[0]),np.linspace(0, 100, 5), linestyle="--", linewidth=2, color='k')
    if len(CAO4_time) != 0: ax2.plot(np.full(5,CAO4_time[-1]),np.linspace(0, 100, 5), linestyle="--", linewidth=2, color='k')
    #if len(CAO4_time) != 0: ax2.plot(np.full(5,'2020-03-15 15:00:00'),np.linspace(0, 100, 5), linestyle="--", linewidth=2, color='k')

    #alternatives
    #if len(CAO1_time) != 0: ax4.vlines(x=CAO1_time[0], ymin=0, ymax=100, linewidth=2, color='k', zorder=100) 
    #if len(CAO2_time) != 0: ax4.vlines(y=100, xmin=CAO2_time[0], xmax=CAO2_time[-1], linewidth=5, color='k', zorder=100)
    #if len(WAI_time) != 0: ax4.vlines(y=100, xmin=WAI_time[0], xmax=WAI_time[-1], linewidth=5, color='red', zorder=100)
    #if len(CAO3_time) != 0: ax4.vlines(x=CAO3_time[0], ymin=0, ymax=100, linewidth=2, color='k', zorder=100) 
    #if CAO4_time[0] == '2020-03-12 00:00:00': ax4.vlines(x=CAO4_time[0], ymin=0, ymax=100, linewidth=2, color='k', zorder=100)
    #if CAO4_time[-1] == '2020-03-15 00:00:00': ax2.vlines(x=CAO4_time[-1], ymin=0, ymax=100, linewidth=2, color='k')
    #if CAO4_time[-1] == '2020-03-15 15:00:00': ax2.vlines(x=CAO4_time[-1], ymin=0, ymax=100, linewidth=2, color='k')


    
    colors = ["#00AEAD", "#019875","#72CC50","#BFD834"]
    colors = ["#B6B6B6", "#828282", "#505050", "#323232"] #rgb ["182	182	182, 130	130	130	, 80	80	80	, 29	29	29	"]
    if barb == True: 
        width=0.03
        a4 = ax2.bar(time, pros_CAOabove4, width, color=colors[0], alpha= 0.9, align='center',label='>4')
        a3 = ax2.bar(time, pros_CAOabove8, width, color=colors[1], alpha= 0.9, align='center',label='>8')
        a2 = ax2.bar(time, pros_CAOabove12,width, color=colors[2], alpha= 0.9, align='center',label='>12')
        a1 = ax2.bar(time, pros_CAOabove16,width, color=colors[3], alpha= 0.9, align='center',label='>16')
    elif fillbg==True:
        all_pros=np.concatenate((pros_CAOabove4,pros_CAOabove8,pros_CAOabove12,pros_CAOabove16 ), axis=None).reshape(4,len(time))
        CF_Q = ax2.contourf(time, np.linspace(0,100,4), all_pros,  colors=colors, levels=[40,50,60,80,100],alpha=0)#,extend="both", zorder=1)
        ax2.fill_between(time, pros_CAOabove4,0, color=colors[0], alpha= 1)
        ax2.fill_between(time, pros_CAOabove8,0, color=colors[1], alpha= 1)
        ax2.fill_between(time, pros_CAOabove12,0, color=colors[2], alpha= 1)
        ax2.fill_between(time, pros_CAOabove16,0, color=colors[3], alpha= 1)
        cycles=np.linspace(0,0.75,4)
        cm = LinearSegmentedColormap.from_list('defcol', colors, N=4)
        cbar=nice_vprof_colorbar(CF=CF_Q, ax=ax2, x0=0.73,y0=0.02,width=0.26,height=0.22, label='CAO index / K')#, lvl = [40,50,60,80], ticks=[40,50,60,80])#, label='CAO index / K', highlight_val=None, highlight_linestyle="k--",format='%.1f', extend="both",x0=0.75,y0=0.86,width=0.26,height=0.13)
        cbar.ax.set_xticklabels(['>4', '>8', '>12','>16', ""], zorder=1000)
        cbar.solids.set(alpha=1)

        #alternative cbar without using the fake contourf
        #ax3 = fig.add_axes([0.91, 0.38, 0.01, 0.35]) #left, bottom, width, height] 
        #ax3.set_ylabel("CAO")
        #ax3.set_zorder(ax.get_zorder()+1) # put ax in front of ax2
        #cb = mpl.colorbar.ColorbarBase(ax3, cmap=cm, ticks=cycles,orientation="horizontal")
        #cb = mpl.colorbar.ColorbarBase(ax3, cmap=cm, ticks=cycles, orientation="vertical")
        #cb.ax.set_yticklabels(['>4', '>8', '>12','>16'], zorder=100)
        #cb.set_label('CAO index / K')

    #zorder handle: 
    ax.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2
    #ax4.set_zorder(ax.get_zorder()+1) x
    #ax2.set_zorder(ax4.get_zorder()+1) # put ax in front of ax2


    #save it. 
    make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(args["datetime"][0]+"_"+args["datetime"][-1]))
    file_path = "{0}/{1}_{2}_{3}_{4}.png".format(
                make_modelrun_folder, args["model"], args["point_name"], "CAOi_point", args["datetime"][0]+"_"+args["datetime"][-1])
    print(f"filename: {file_path}")
    fig.tight_layout(pad=7.0)
    print("after tight layout")
    fig.savefig(file_path, bbox_inches="tight", dpi=300) #pad_inches
    #fig.savefig(file_path, dpi=300) #pad_inches

    plt.cla()


def plot_CAOindex_point_diff_areas(point, dmet, barb=True,fillbg=False, **args):
    rad = (len(dmet.x)/2)*2.5
    textinfo = f"{point.name}; cord: ({point['lat']}, {point['lon']}), radius:{rad} km"
    CAOi_all = CAO_index(dmet.air_temperature_pl,dmet.pressure,dmet.SST,dmet.air_pressure_at_sea_level, p_level=850).squeeze()
    CAOi = np.where(dmet.SIC <= 0.1, CAOi_all, np.NaN)
    CAOi = np.where(dmet.land_area_fraction.squeeze() == 0, CAOi, np.NaN)
    CAOi = np.where(CAOi<-100,np.nan,  CAOi)

    SImask = np.where(dmet.SIC >= 0.1, dmet.SIC, np.NaN)
    try: 
         CAOi= CAOi.squeeze(axis=1)
         SImask= SImask.squeeze(axis=1)
    except: 
        SImask = SImask
        SImask = SImask
    
    #CAOi_area_red = CAOi[:,area:-area, area:-area]
    CAOialleval= np.nansum(CAOi >-9999, axis=(1,2))
    CAOiabove4 = np.nansum(CAOi >= 4, axis=(1,2))
    CAOiabove8 = np.nansum(CAOi >= 8,axis=(1,2))
    CAOiabove12= np.nansum(CAOi >= 12,axis=(1,2))
    CAOiabove16= np.nansum(CAOi >= 16,axis=(1,2))

    pros_CAOabove4 = 100*(CAOiabove4/CAOialleval)
    pros_CAOabove8 = 100*(CAOiabove8/CAOialleval)
    pros_CAOabove12 = 100*(CAOiabove12/CAOialleval)
    pros_CAOabove16 = 100*(CAOiabove16/CAOialleval)
        
    mean_CAOi = np.nanmean(CAOi, axis= (1,2))
    std_CAOi = np.nanstd(CAOi, axis= (1,2))  
    min_CAOi = np.nanmin(CAOi, axis=(1,2))
    max_CAOi = np.nanmax(CAOi, axis=(1,2))

    if args["hold"]: 
        fig, ax = plt.subplots(figsize=(15,8)) #    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9), subplot_kw={'projection': crs})
    else:
        fig = args["fig"]
        ax = args["ax"]
    
    time= pd.to_datetime(dmet.time, unit='s') #origin=pd.Timestamp('1960-01-01')
    ax2=ax.twinx(); ax2.set_ylim(-1,101)
    ax.set_xticks(time)
    ax.set_xticklabels(time,rotation=90, fontsize=16)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.set_ylabel("CAO index")
    ax2.set_ylabel('Sea area %')
    ax.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2
    ax.patch.set_visible(False) 
    ax.errorbar(time, mean_CAOi,[mean_CAOi - min_CAOi, max_CAOi - mean_CAOi],linestyle='None', alpha=0.3, marker='o',capsize=2, capthick=2, fillstyle="full",color=args["col"], label=f"CAO index for: \n{textinfo}")
    ax.plot(time, mean_CAOi, marker='o', color=args["col"], linestyle = 'None' )
    ax.legend(loc='upper left') #[means - mins, maxes - means]
    ax2.plot(time,pros_CAOabove12, color=args["col"] )
    return fig, ax
    

def plot_CAOindex_map(**kwargs ):
    kwargs["domain_name"]="Svalbard"
    #kwargs["domain_name"]="AromeArctic"
    #kwargs["overlays"]=["point_name"]
    #kwargs["point_name"]= [kwargs.point_name]
    plot_CAO(**kwargs)

def plot_CAOindex_map_all(**kwargs ):
    #kwargs["domain_name"]="Svalbard"
    #kwargs["domain_name"]="AromeArctic"
    #kwargs["overlays"]=["point_name"]
    #kwargs["point_name"]= [kwargs.point_name]
    plot_domain(**kwargs)


def plot_domain( dmet,datetime,data_domain, **kwargs ):
    time= pd.to_datetime(dmet.time, unit='s') #origin=pd.Timestamp('1960-01-01')

    dt=datetime
    crs = default_map_projection(dmet)
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9), subplot_kw={'projection': crs})
    itim=0

    an_array = np.empty( (len(dmet.y),len(dmet.x)) )
    ax1.contourf(dmet.x, dmet.y,an_array, alpha=0.1, colors="gray",ls=None, lw=0, levels=np.arange(-1,100,0.001))
    overlays=["point_name"]
    kwargs["point_name"]= ["NYA_WMO"]
    add_overlay(overlays, ax=ax1,crs=crs,size=20,marker="o",col=["blue"],**kwargs)

    ax1.add_feature(cfeature.GSHHSFeature(scale="high"))
    #ax1.add_feature(cfeature.GSHHSFeature(scale="high"))
    domain_name ="Svalbard_z1" 
    eval(f"data_domain.{domain_name}()")
    ax1.set_extent(data_domain.lonlat)
    
    nicegrid(ax=ax1, xx = np.arange(-20, 80, 10),yy = np.arange(50, 90, 4), color='gray', alpha=0.5, linestyle='--')

    make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}{1}".format(dt,"-"))
    file_path = "{0}/{1}_{2}.png".format(make_modelrun_folder, "domain", dt)
    print(f"filename: {file_path}")
    fig1.savefig(file_path, bbox_inches="tight", dpi=200)


def plot_sea_ice( dmet,datetime,data_domain, **kwargs ):
    # = np.where(dmet.SIC <= 0.1, dmet.SIC, np.NaN)
    SIC = np.where(dmet.land_area_fraction.squeeze() <= 0, dmet.SIC,  np.nan)

    time= pd.to_datetime(dmet.time, unit='s') #origin=pd.Timestamp('1960-01-01')
    CAO1_time= time[np.logical_and(time>='2020-02-25 00:00:00', time<='2020-02-29 00:00:00')]
    CAO2_time= time[np.logical_and(time>='2020-03-01 12:00:00', time<='2020-03-03 12:00:00')]
    WAI_time = time[np.logical_and(time>='2020-03-03 12:00:00', time<='2020-03-08 00:00:00')]
    CAO3_time= time[np.logical_and(time>='2020-03-09 12:00:00', time<='2020-03-12 00:00:00')]
    CAO4_time= time[np.logical_and(time>='2020-03-12 00:00:00', time<='2020-03-15 00:00:00')]
    dt=datetime
    crs = default_map_projection(dmet)
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9), subplot_kw={'projection': crs})
    
    an_array = np.empty( (len(dmet.y),len(dmet.x)) )
    ax1.contourf(dmet.x, dmet.y,an_array, alpha=0.4, colors="gray",ls=None, lw=0, levels=np.arange(-1,100,0.001))
    
    overlays=["point_name"]
    kwargs["point_name"]= ["NYA_WMO"]
    add_overlay(overlays, ax=ax1,crs=crs,size=20,marker="o",col=["blue"],**kwargs)

    ax1.add_feature(cfeature.GSHHSFeature(scale="low"))
    #ax1.add_feature(cfeature.GSHHSFeature(scale="high"))
    domain_name ="Svalbard_z1"
    eval(f"data_domain.{domain_name}()")
    ax1.set_extent(data_domain.lonlat)
    nicegrid(ax=ax1, xx = np.arange(-20, 80, 10),yy = np.arange(50, 90, 4), color='gray', alpha=0.5, linestyle='--')
    #itim=0
    #for itim in range(0,len(time)):
    #    print(itim)
    #    ax1.contour(dmet.x, dmet.y, SIC[itim, :, :] if len(np.shape(SIC)) == 3 else SIC[itim,0, :, :],
    #                            zorder=2, linewidths=2.0, colors="green", levels=[0.1])  #
    # 
    SIC= np.where(SIC <= 0.1, SIC, np.nan)
    ax1.contourf(dmet.x, dmet.y, SIC[0,:,:], colors="gray", alpha=0.7,ls=None, lw=0, vmax=0.1, levels=[-1,0.1])
    #ax1.imshow(SIC[0,:,:], vmax=0.1)
    #ax1.plot(dmet.x[SIC[0,:,:] >= 0.1], dmet.y[SIC[0,:,:] >= 0.1] , colors="gray", alpha=0.7)
    make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}{1}".format(dt,"-"))
    file_path = "{0}/{1}_{2}.png".format(make_modelrun_folder, "CAOi", dt[0])
    print(f"filename: {file_path}")
    #plt.show()
    #fig1.savefig(file_path, bbox_inches="tight", dpi=300)
    fig1.savefig(file_path, dpi=300)

    ax1.cla()
    plt.close(fig1)
    plt.close("all")
    gc.collect()



def CAO(datetime,use_latest, delta_index, coast_details, steps=0, model="MEPS", domain_name=[None], domain_lonlat=None, legend=False, info=False, grid=True,
        runid=None, outpath=None, url=None, point_lonlat =None,overlays=None, point_name=["NYA_WMO"], num_point=30000):
    param= ["air_pressure_at_sea_level", "surface_geopotential", "air_temperature_pl", "SST", "SIC","land_area_fraction"]  # add later land_area_fraction
    p_level = [850, 1000]
    package_path = os.path.dirname(__file__)
    sites = pd.read_csv(f"{package_path}../data/sites.csv", sep=";", header=0, index_col=0)
    diff_areas=False
    if diff_areas:
        count=0
        my_col = ['r', 'g', 'b', 'y']
        num_point_all = [30000] #10000   
        hold=1
        fig=None
        ax=None
        for num_point in num_point_all: 
            for one_point_name in point_name:
                point = sites.loc[one_point_name]
                dmet, data_domain, bad_param = checkget_data_handler(all_param=param, model=model, p_level=p_level, date=datetime, step=steps, point_name= [one_point_name], domain_name=None, num_point=num_point)
                plot_CAOindex_map_all(datetime=datetime, dmet=dmet, data_domain=data_domain, steps=steps, model=model, domain_name=domain_name,
                         domain_lonlat=domain_lonlat, legend=legend, info=info, grid=grid, url=url,
                         overlays=overlays, point_name=[one_point_name],num_point=num_point)
            
                fig, ax= plot_CAOindex_point_diff_areas(point=point, datetime=datetime, dmet=dmet, data_domain=data_domain, steps=steps, model=model, domain_name=domain_name,
                         domain_lonlat=domain_lonlat, legend=legend, info=info, grid=grid, url=url,
                         overlays=overlays, point_name=one_point_name, num_point=num_point, hold =hold, fig=fig, ax=ax, col=my_col[count])
                hold=0
                count+=1
        make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(datetime[0]+"_"+datetime[-1]))
        file_path = "{0}/DIFFA{1}_{2}_{3}_{4}.png".format(
        make_modelrun_folder, model, point_name, "CAOi_point", datetime[0]+"_"+datetime[-1])
        print(f"filename: {file_path}")
        fig.savefig(file_path, bbox_inches="tight", dpi=200)
        plt.cla()
    
    
    #CAO_fram
    point = sites.loc[point_name[0]]
    dmet, data_domain, bad_param = checkget_data_handler(all_param=param, model=model, p_level=p_level, date=datetime, step=steps, point_name= point_name, domain_name=None, num_point=30000, read_from_saved="20200223-2020030315+24.nc") #read_from_saved
    #dmet, data_domain, bad_param = checkget_data_handler(all_param=param, model=model, p_level=p_level, date=datetime, step=steps, domain_name="CAO_fram")

    #plot_sea_ice( dmet=dmet, datetime=datetime,data_domain=data_domain, scale=0)


    #plot_domain( dmet=dmet,datetime=datetime, data_domain=data_domain, scale=scale)
    #plot_CAOindex_point_diff_areas(point=point, datetime=datetime, dmet=dmet, data_domain=data_domain, steps=steps, model=model, domain_name=domain_name,
    #                     domain_lonlat=domain_lonlat, legend=legend, info=info, grid=grid, url=url,
    #                     overlays=overlays, point_name=one_point_name, num_point=num_point, hold =hold, fig=fig, ax=ax, col=my_col[count])
    #exit(1)
    plot_CAOindex_point(point=point, datetime=datetime, dmet=dmet, data_domain=data_domain, steps=steps, model=model, domain_name=domain_name,
                         domain_lonlat=domain_lonlat, legend=legend, info=info, grid=grid, url=url,
                         overlays=overlays, point_name=point_name)
        
    #plot_CAOindex_map(datetime=datetime, dmet=dmet, data_domain=data_domain, steps=steps, model=model, domain_name=domain_name,
    #                     domain_lonlat=domain_lonlat, legend=legend, info=info, grid=grid, url=url,
    #                     overlays=overlays, point_name=point_name)


    

if __name__ == "__main__":
    # python CAOindex_point.py --datetime 2020031300 --point_name CAO1 --model AromeArctic
    args = default_arguments()
    chunck_func_call(func = CAO, chunktype= args.chunktype, chunk=args.chunks, datetime=args.datetime, steps=args.steps, model=args.model,
            domain_name=args.domain_name, domain_lonlat=args.domain_lonlat, legend=args.legend, info=args.info, grid=args.grid, runid=args.id,
            outpath=args.outpath, use_latest=args.use_latest,delta_index=args.delta_index, coast_details=args.coast_details, url=args.url,
            point_lonlat =args.point_lonlat, overlays= args.overlays, point_name=args.point_name, num_point=args.num_point)
    



    #python CAOindex_point.py --datetime 2020022300 2020022400 2020022500 2020022600 2020022700 2020022800 2020022900 2020030100 2020030200 2020030300 2020030400 2020030500 2020030600 2020030700 2020030800 2020030900 2020031000 2020031100 2020031200 2020031300 2020031400 2020031500 --point_name NYA_WMO --model AromeArctic --steps 0:24


    #python CAOindex_point.py --datetime 2020022300 2020022303 2020022306 2020022309 2020022312 2020022315 2020022318 2020022321 2020022400 2020022403 2020022406 2020022409 2020022412 2020022415 2020022418 2020022421 2020022500 2020022503 2020022506 2020022509 2020022512 2020022515 2020022518 2020022521 2020022600 2020022603 2020022606 2020022609 2020022612 2020022615 2020022618 2020022621 2020022700 2020022703 2020022706 2020022709 2020022712 2020022715 2020022718 2020022721 2020022800 2020022803 2020022806 2020022809 2020022812 2020022815 2020022818 2020022821 2020022900 2020022903 2020022906 2020022909 2020022912 2020022915 2020022918 2020022921 2020030100 2020030103 2020030106 2020030109 2020030112 2020030115 2020030118 2020030121 2020030200 2020030203 2020030206 2020030209 2020030212 2020030215 2020030218 2020030221 2020030300 2020030303 2020030306 2020030309 2020030312 2020030315 2020030318 2020030321 2020030400 2020030403 2020030406 2020030409 2020030412 2020030415 2020030418 2020030421 2020030500 2020030503 2020030506 2020030509 2020030512 2020030515 2020030518 2020030521 2020030600 2020030603 2020030606 2020030609 2020030612 2020030615 2020030618 2020030621 2020030700 2020030703 2020030706 2020030709 2020030712 2020030715 2020030718 2020030721 2020030800 2020030803 2020030806 2020030809 2020030812 2020030815 2020030818 2020030821 2020030900 2020030903 2020030906 2020030909 2020030912 2020030915 2020030918 2020030921 2020031000 2020031003 2020031006 2020031009 2020031012 2020031015 2020031018 2020031021 2020031100 2020031103 2020031106 2020031109 2020031112 2020031115 2020031118 2020031121 2020031200 2020031203 2020031206 2020031209 2020031212 2020031215 2020031218 2020031221 2020031300 2020031303 2020031306 2020031309 2020031312 2020031315 2020031318 2020031321 2020031400 2020031403 2020031406 2020031409 2020031412 2020031415 2020031418 2020031421 2020031500 2020031503 2020031506 2020031509 2020031512 2020031515 2020031518 2020031521 --point_name NYA_WMO --model AromeArctic --steps 0:2
    #                                 
    # 
    # 
    # python CAOindex_point.py --datetime 2020022400 2020022500 2020022600 2020022700 2020022800 2020022900 2020030100 2020030200 2020030300 2020030400 2020030500 2020030600 2020030700 2020030800 2020030900 2020031000 2020031100 2020031200 2020031300 2020031400 2020031500 --point_name NYA_WMO --model AromeArctic --steps 0:24



    #diff = np.abs(mean_CAOi-10)
    #cao_auto= time[np.where(diff<0.1)]
    #DatetimeIndex(['2020-02-23 04:00:00', '2020-02-24 20:00:00',
     #          '2020-02-29 12:00:00', '2020-02-29 13:00:00',
     #          '2020-02-29 14:00:00', '2020-03-08 17:00:00',
     #          '2020-03-08 18:00:00', '2020-03-08 19:00:00',
     #          '2020-03-08 20:00:00', '2020-03-08 21:00:00'],
     #         dtype='datetime64[ns]', freq=None)