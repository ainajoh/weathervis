# %%
#python OLR_sat.py --datetime 2022030400 --steps 0 3 --model AromeArctic --domain_name Svalbard North_Norway --use_latest 0
#
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
from mpl_toolkits.axes_grid1 import make_axes_locatable ##__N
from weathervis.checkget_data_handler import *
import gc
from weathervis.plots.add_overlays import add_overlay
from matplotlib.colors import ListedColormap, BoundaryNorm

plt.rcParams.update({'font.size': 15})

global obukhov_map
MyObject = type('MyObject', (object,), {})
obukhov_map = MyObject()
_param = ["H", "surface_geopotential", "air_temperature_ml", "air_temperature_2m", "air_temperature_0m", "FMU", "FMV", "ap", "b","air_pressure_at_sea_level","toa_outgoing_longwave_flux","surface_air_pressure"]
setattr(obukhov_map, "param", _param)
#obukhov(T1M = ds.T[64,:,:],ps= ds.SP,tsurf=ds.T2m, 
#        H=ds.SSHF,ustress= ds.USTRESS, vstress=ds.VSTRESS, 
#         ak1=ds.Ak[64], bk1=ds.Bk[64] )

warnings.filterwarnings("ignore", category=UserWarning) # suppress matplotlib warning

def plot_obukhov(datetime,dmet,figax=None, scale=1, lonlat=None, steps=[0,2], coast_details="auto", model=None, domain_name=None,
             domain_lonlat=None, legend=True, info=False, grid=True,runid=None, outpath=None, url = None, save= True, overlays=None,  data_domain=None, **kwargs):

  #if data_domain is not None: eval(f"data_domain.{domain_name}()")  # get domain info

  ## CALCULATE AND INITIALISE ####################
  #scale = data_domain.scale  #scale is larger for smaller domains in order to scale it up.
  dmet.air_pressure_at_sea_level /= 100
  MSLP = filter_values_over_mountain(dmet.surface_geopotential[:], dmet.air_pressure_at_sea_level[:])
  T1M=dmet.air_temperature_ml[-1].squeeze()
  ps = dmet.surface_air_pressure[:].squeeze()
  tsurf=dmet.air_temperature_2m[:].squeeze()
  #tsurf=dmet.air_temperature_0m[:].squeeze()
  H=dmet.H[:].squeeze()
  ustress=dmet.FMU[:].squeeze()
  vstress=dmet.FMV[:].squeeze()
  ak1=dmet.ap[-1].squeeze()
  bk1=dmet.b[-1].squeeze()
  #ol=
  ol =calc_obukhov(T1M,ps,tsurf,H, ustress,vstress,ak1,bk1 )
  ol=ol.squeeze()

  #print("aiaiaia")
  #print(ol)
  # PLOTTING ROUTNE ######################
  crs = default_map_projection(dmet) #change if u want another projection
  fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9), subplot_kw={'projection': crs}) if figax is None else figax
  itim = 0
  for leadtime in np.array(steps): #
      print('Plotting {0} + {1:02d} UTC'.format(datetime, leadtime))
      ax1 = default_mslp_contour(dmet.x, dmet.y, MSLP[itim, 0, :, :], ax1, scale=scale)
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
            (x[1:, 1:] > 1e20))
      data =  dmet.toa_outgoing_longwave_flux[itim, 0,:nx - 1, :ny - 1].copy()
      data[mask] = np.nan
      #ax1.pcolormesh(x, y, data[ :, :], vmin=-230,vmax=-110, cmap=plt.cm.Greys_r, zorder=2, alpha=1)
      ax1.add_feature(cfeature.GSHHSFeature(scale='high'),linewidth=0.5, zorder=6, facecolor='gray')  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).
      #ax1.text(0, 1, "{0}_obukhov_{1}+{2:02d}".format(model, datetime, leadtime), ha='left', va='bottom', transform=ax1.transAxes,color='dimgrey')
      
      #land_feature = cfeature.GSHHSFeature(scale='auto', levels=[1], facecolor='gray')
      #ax.add_feature(land_feature)


      #Obukhov length

      coolwarm = plt.get_cmap('tab20b_r') #tab10'
      colors = coolwarm(np.linspace(0, 1, 100))  # Get 256 colors from the coolwarm cmap
      new_cmap = coolwarm#ListedColormap(colors)
      new_cmap.set_over('black')
      new_cmap.set_under('black')
      levels=[-500,-200,-100,-50,10,50,100,200,500]
      norm = BoundaryNorm([-500,-200,-100,-50,10,50,100,200,500],ncolors=coolwarm.N)
      ol2 = deepcopy(ol)
      ol2[ol2<-50] = np.nan
      ol2[ol2>10] = np.nan
      norm2 = BoundaryNorm([10,0,-50],ncolors=10)

      ol3 = deepcopy(ol) #pnly stable
      ol3[ol3<0] = np.nan

      ol4 = deepcopy(ol) #only unstable
      ol4[ol4>0] = np.nan
      #tab20b
      #cs =ax1.pcolormesh(x, y, T1M, alpha=0.5, zorder=3)
      #cs =ax1.pcolormesh(x, y, ol, alpha=0.5, vmin=-500, vmax=500, zorder=3)
      #cs = ax1.pcolormesh(x, y, ol, cmap=new_cmap, norm=norm, vmax=500, vmin=-500,zorder=3, alpha=1)#, cbar_kwargs={'label': 'Air temperature (K)'}) #ransform=crs,
      cs = ax1.pcolormesh(x, y, ol, cmap="tab20b_r", vmax=500, vmin=-500,zorder=3, alpha=1)#, cbar_kwargs={'label': 'Air temperature (K)'}) #ransform=crs,

      #cs2 = ax1.pcolormesh(x, y, ol2, cmap="Reds_r", vmax=0, vmin=-50,zorder=4, alpha=1)
      #cs = ax1.pcolormesh(x, y, ol3, cmap="Blues_r", vmax=500, vmin=0,zorder=4, alpha=1)
      #cs4 = ax1.pcolormesh(x, y, ol4, cmap="hot", vmax=0, vmin=-500,zorder=4, alpha=1)

      #cs = ax1.contourf(x, y, ol, cmap=coolwarm, levels=levels, extend="both", vmax=500, vmin=-500,zorder=3, alpha=0.7)#, cbar_kwargs={'label': 'Air temperature (K)'}) #ransform=crs,
      #      ax1.pcolormesh(x, y, data[ :, :], vmin=-230,vmax=-110, cmap=plt.cm.Greys_r)
      #  ax1.add_feature(cfeature.GSHHSFeature(scale='intermediate'),edgecolor="brown", linewidth=0.5)  # ‘auto’, ‘coarse’, ‘low’, ‘intermediate’, ‘high, or ‘full’ (default is ‘auto’).

      #plt.colorbar(cs,extend="both")
      
      latsline= np.arange(69,85,1)
      lons7 = np.full_like(latsline, 7)
      ax1.plot(lons7,latsline,color='white', linestyle='-', linewidth=4,transform=ccrs.Geodetic(), marker="o", markersize=7, zorder=5)
      
      stablecolor="blue"
      neutralcolor="k"
      unstablecolor="red"
      markeredgecolor="white"
      markeredgewidth=1
      #markerfacecolor='red', markeredgewidth=2, markeredgecolor='black'
      ax1.plot(7, 84, markerfacecolor=neutralcolor, markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor, transform=ccrs.Geodetic(), marker="o", markersize=7, zorder=6)
      ax1.plot(7, 83, markerfacecolor=stablecolor, markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor,transform=ccrs.Geodetic(), marker="o", markersize=7, zorder=6)
      ax1.plot(7, 82, markerfacecolor=stablecolor, markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor,transform=ccrs.Geodetic(), marker="o", markersize=7, zorder=6)
      ax1.plot(7, 81, markerfacecolor=stablecolor, markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor,transform=ccrs.Geodetic(), marker="o", markersize=7, zorder=6)
      ax1.plot(7, 80, markerfacecolor=unstablecolor, markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor,transform=ccrs.Geodetic(), marker="o", markersize=7, zorder=6)
      ax1.plot(7, 79, markerfacecolor=unstablecolor, markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor,transform=ccrs.Geodetic(), marker="o", markersize=7, zorder=6)
      ax1.plot(7, 78, markerfacecolor=unstablecolor, markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor,transform=ccrs.Geodetic(), marker="o", markersize=7, zorder=6)
      ax1.plot(7, 77, markerfacecolor=unstablecolor,markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor, transform=ccrs.Geodetic(), marker="o", markersize=7, zorder=6)
      ax1.plot(7, 76, markerfacecolor=unstablecolor, markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor,transform=ccrs.Geodetic(), marker="o", markersize=7, zorder=6)
      ax1.plot(7, 75, markerfacecolor=unstablecolor, markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor,transform=ccrs.Geodetic(), marker="o", markersize=7, zorder=6)
      ax1.plot(7, 74, markerfacecolor=unstablecolor, markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor,transform=ccrs.Geodetic(), marker="o", markersize=7, zorder=6)
      ax1.plot(7, 73, markerfacecolor=unstablecolor, markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor,transform=ccrs.Geodetic(), marker="o", markersize=7, zorder=6)
      ax1.plot(7, 72, markerfacecolor=unstablecolor, markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor,transform=ccrs.Geodetic(), marker="o", markersize=7, zorder=6)
      ax1.plot(7, 71, markerfacecolor=unstablecolor, markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor,transform=ccrs.Geodetic(), marker="o", markersize=7, zorder=6)
      ax1.plot(7, 70, markerfacecolor=unstablecolor, markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor,transform=ccrs.Geodetic(), marker="o", markersize=7, zorder=6)
      ax1.plot(7, 69, markerfacecolor=unstablecolor, markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor,transform=ccrs.Geodetic(), marker="o", markersize=7, zorder=6)
  
      
      #ax1.plot(7,77,color='black', linestyle='-', lw=2, marker="o",transform=ccrs.Geodetic())
      legend=True;grid=True
      if legend:
        ax_cb = adjustable_colorbar_cax(fig1, ax1)
        plt.colorbar(cs,extend="both", cax = ax_cb, fraction=0.046, pad=0.01, aspect=25,
                label=r"Monin–Obukhov length / m", alpha=1)
        #plt.colorbar(cs,extend="both",fraction=0.046,
        #        label=r"Monin obukhob length")

        #proxy = [plt.axhline(y=0, xmin=0, xmax=0, color="gray",zorder=7)]
        # proxy.extend(proxy1)
        # legend's location fixed, otherwise it takes very long to find optimal spot
        #lg = ax1.legend(proxy, [f"MSLP (hPa)", f"Sea ice at 50%"])

        #frame = lg.get_frame()
        #frame.set_facecolor('white')
        #frame.set_alpha(0.8)
       
      if grid:
        nicegrid(ax=ax1,zorder=5,yy=[84,83,82,81,80,79,78,77,76,75,74,73,72,71,70,69])
      if overlays:
        add_overlay(overlays, ax=ax1, **kwargs)
      if domain_name != model and data_domain != None:
        ax1.set_extent(data_domain.lonlat)

      ax1.set_extent([10,30,67,85])
      #ax1.set_extent([9,30,67,86])

      make_modelrun_folder = setup_directory(OUTPUTPATH, "{0}".format(datetime))
      file_path = "{0}/{1}_{2}_{3}_{4}+{5:02d}.png".format(make_modelrun_folder, model, domain_name, "obukhov", datetime,leadtime)
      print(f"filename: {file_path}")
      if save: 
        fig1.savefig(file_path, bbox_inches="tight", dpi=300) 
      else:
        pass
        #plt.show()
      #ax1.cla()
      itim += 1
  del MSLP, scale, itim, legend, grid, overlays, domain_name, data, mask, x, y,nx,ny
  del make_modelrun_folder, file_path
  if figax is None:
    plt.close(fig1)
    plt.close("all")
    del dmet, data_domain
    del fig1, ax1, crs
  gc.collect()

def obukhov(datetime,use_latest, delta_index, coast_details, steps=0, model="AromeArctic", domain_name=None, domain_lonlat=None, legend=False, info=False, grid=True,
        runid=None, outpath=None, url=None, point_lonlat =None,overlays=None, point_name=None):
    param = obukhov_map.param #["air_pressure_at_sea_level", "surface_geopotential", "toa_outgoing_longwave_flux", "SIC"]
    p_level = None
    m_level=[64] #64 is lowest level
    plot_by_subdomains(plt_func= plot_obukhov, checkget_data_handler= checkget_data_handler, datetime=datetime, steps=steps,model= model, domain_name=domain_name,domain_lonlat=  domain_lonlat, legend=legend,
                       info=info, grid=grid, url=url, point_lonlat=point_lonlat, use_latest=use_latest,
                       delta_index=delta_index, coast_details=coast_details, param=param, p_level=p_level,m_level=m_level,overlays=overlays, runid=runid, point_name=point_name,
                       save2file=False, read_from_saved="obukhov.nc")



if __name__ == "__main__":
    args = default_arguments()
    #save2file=False, read_from_saved=False,
    chunck_func_call(func = obukhov, chunktype= args.chunktype, chunk=args.chunks, datetime=args.datetime, steps=args.steps, model=args.model,
            domain_name=args.domain_name, domain_lonlat=args.domain_lonlat, legend=args.legend, info=args.info, grid=args.grid, runid=args.id,
            outpath=args.outpath, use_latest=args.use_latest,delta_index=args.delta_index, coast_details=args.coast_details, url=args.url,
            point_lonlat =args.point_lonlat, overlays= args.overlays, point_name=args.point_name)
    gc.collect()