import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import shapely as sh
import numpy as np
import cartopy.crs as ccrs

def add_overlay(types=[None], crs=ccrs,  **kwargs): #for quick on/off in default map
  print("in add_overlays")
  print(kwargs)
  kwargs["crs"] = crs
  print(types)
  #exit()
  for ty in types:
    if ty=="ISLAS":
      add_ISLAS_overlays(**kwargs)
    if ty=="point_name":
      point_on_map(**kwargs)
    #if ty=="topinfotext":
    #  add_topinfotext(**kwargs)
    #  #add_topinfotext(ax, model, datetime, leadtime, plot_name="missin_name", **kwargs)
 ##red #FF8886   blue #81d4fa
def point_on_map(ax,col=["#81d4fa", "#FF8886"],size = 1200, **kwargs):
  print("in overlays")
  #import cartopy.crs as ccrs
  sites="../data/sites.csv"
  locs = pd.read_csv(sites,sep=';')
  with ax.hold_limits():
    i_p = 0
    for pn in kwargs["point_name"]:
      if len(col) == i_p:
        i_p = 0
      print(pn)
      pointplot = locs[locs["Name"]==pn] #sites.loc
      print(pointplot)
      print(float(pointplot["lon"].values[0]))
      ax.scatter(float(pointplot["lon"].values[0]),float(pointplot["lat"].values[0]),s=size,color=col[i_p],marker='s',zorder=12, transform=ccrs.PlateCarree())
      i_p+=1
def add_default_mslp_contour(x, y, MSLP, ax1, scale=1):
  # MSLP with contour labels every 10 hPa
  C_P = ax1.contour(x, y, MSLP, zorder=1, alpha=1.0,
                      levels=np.arange(round(np.nanmin(MSLP), -1) - 10, round(np.nanmax(MSLP), -1) + 10, 1 / scale),
                      colors='grey', linewidths=0.5)
  C_P = ax1.contour(x, y, MSLP, zorder=2, alpha=1.0,
                      levels=np.arange(round(np.nanmin(MSLP), -1) - 10, round(np.nanmax(MSLP), -1) + 10, 10 / scale),
                      colors='grey', linewidths=1.0, label="MSLP [hPa]")
  ax1.clabel(C_P, C_P.levels, inline=True, fmt="%3.0f", fontsize=10)

  return ax1

def add_nicegrid(ax, xx=np.arange(-20, 80, 20), yy=np.arange(50, 90, 4), color='gray', alpha=0.5, linestyle='--'):
  """A function for cartopy making nice gridlines
    input:
        ax: axis for a matplotlib cartopy figure  : Required
        xx: longitude array for drawing gridlines : optional
        yy: latitude array for drawing gridlines  : optional
        color: color of gridlines                 : optional
        alpha: slpha of gridlines                 : optional
        linestyle: linestyle of gridlines         : optional
    usage:

  """
  import matplotlib.ticker as mticker  # used in nicegrid()
  gl = ax.gridlines(draw_labels=True, linewidth=1, color=color, alpha=alpha, linestyle=linestyle, zorder=10)
  gl.xlabels_top = False

  gl.xlocator = mticker.FixedLocator(xx)
  gl.ylocator = mticker.FixedLocator(yy)
  gl.xlabel_style = {'color': color}

def add_topinfotext(ax, model, datetime, leadtime, plot_name="missin_name", **kwargs):
  ax.text(0, 1, "{0}_{1}_{2}+{3:02d}".format(model, plot_name, datetime, leadtime), ha='left', va='bottom',
           transform=ax.transAxes, color='dimgrey')


def add_ISLAS_overlays(ax,col='red', **kwargs):
  import cartopy.crs as ccrs
  import geog as gg  #todo: eliminate such small packages making weathervis too much dependent on these
  #pip install geog todo: not listed as requirements..

  print(col)

  lon=[10, 0, 0, 5,10.5,20.1,22,29,29.5,10]
  lat=[79,72,60,57,58.5,58.5,65,65,79  ,79]
  # Coordinates of boundary points of the flight area :
  #1  :  N72Â° - E000Â2  :  N60Â° - E000Â°3  :  N57Â° - E005Â°
  #4  :  N58Â°3 E010Â°3	5  :  N58Â°3 E020Â6  :  N65Â° - E022Â°
  #7  :  N65Â° - E029Â°	8  :  N79Â° - E0299  :  N79Â° - E010Â°

#------------------------Tim---------------------------------------
#GND-2500FT
  lat_f1 = [69.43, 69.45, 69.05, 69.27, 69.33, 68.75, 67.9, 67.9, 
        68.18, 68.28, 68.3, 68.44, 68.44, 69.23, 69.5, 69.5, 
        69.67, 69.5, 69.5, 69.48, 69.42, 69.4, 69.43]
  lon_f1 = [19.5, 20.0, 20.0, 18.75, 18.45, 17.02, 16.75, 16.75, 
        14.9, 14.95, 14.58, 14.07, 14.07, 14.55, 16.67, 16.67, 
        17.7, 18.17, 18.48, 18.4, 19.05, 19.03, 19.5]

  #GND-FL115 (11500ft)
  lat_f2 = [69.33, 69.27, 69.05, 69.033, 68.73, 68.7, 68.7, 68.3, 
        68.6, 68.55, 68.52, 68.1, 68.2, 67.9, 68.37, 68.75, 69.33]
  lon_f2 = [18.45, 18.75, 20.0, 20.05, 20.27, 20.22, 20.21, 20.0, 
        18.4, 18.1, 18.11, 18.11, 17.41, 16.75, 16.92, 17.06, 18.45]

  # WP files
  # RF B - 20230217
  #lat_WP = [69.29, 69.28, 71.38, 71.45, 69.29] 
  #lon_WP = [16.14, 16.01, 17.34, 14.0, 16.14]
  
  # RF E - 20230224
  #lat_WP = [69.29, 70.8, 71.28, 70.26, 69.71, 69.32, 69.3, 69.27, 69.29]
  #lon_WP = [16.14, 9.18, 10.54, 14.82, 13.67, 15.91, 16.03, 16.21, 16.14]
  
  # Box of Interest during Joint Viking
  lat_WP = [69.29, 70.5, 70.5, 72.48, 72.57, 71.17, 70.5]
  lon_WP = [16.14, 16.14, 6.15, 5.76, 19.51, 19.51, 16.14]

  # cross-section coordinates
  lon_s1 = [4.48069442,4.85184,5.22201596,5.59118371,5.95933048,6.32642438,6.69245751,7.05739099,7.42121725,7.783919, 8.14544846,8.50582583,8.86500184,9.22296394,9.57970969,9.93520038,10.28943229,10.64237789,10.99403634,11.34437057,11.69338545,12.04106361,12.38736733,12.73231844,13.07587542,13.41803582,13.75878882,14.09812172,14.43602451,14.77246887,15.10747908,15.44100987,15.77306521,16.10365217]
  lat_s1 = [70.52626109,70.49986398,70.47271368,70.44480856,70.41615916,70.38676096,70.35662299,70.32574715,70.29413492,70.26179401,70.2287225,70.19492924,70.16041371,70.12518338,70.08923995,70.05258614,70.01523112,69.97717034,69.93841536,69.8989671,69.85882909,69.81800774,69.77650416,69.7343262,69.69147313,69.64795461,69.60377178,69.55892826,69.51343263,69.46728344,69.42048941,69.37305263,69.32497993,69.27627291]
  
  lon_s2 = [16.22677534,16.62366009,17.01861902,17.4116386,17.80269112,18.19176644,18.57884933,18.96392351,19.34697666,19.72799304,20.10696515,20.48387968,20.85872157,21.23148948,21.60217028,21.97075065,22.33723292,22.70160524,23.06385704,23.42399191,23.78200033,24.13787543,24.49161978,24.84322714,25.19269518,25.54002337,25.88520962,26.2282569,26.56915963,26.9079218,27.24454894,27.57903206,27.91138579,28.24160383]
  lat_s2 = [69.2581639,69.20568413,69.15229855,69.09800969,69.0428261,68.98675776,68.92980791,68.87198406,68.81329685,68.75375008,68.69335046,68.63210913,68.57003024,68.50712,68.4433899,68.37884413,68.31348888,68.24733566,68.18038851,68.11265398,68.04414341,67.97486037,67.90481229,67.83401002,67.76245637,67.69016019,67.61713157,67.54337219,67.46889294,67.39370255,67.31780203,67.24120414,67.16391496,67.08593812]

  lon_s3 = [12.23260449,12.4479641,12.65279968,12.84786917,13.0338599,13.21142437,13.38110993,13.54342688,13.69884485,13.84779233,13.99071165,14.12794028,14.25981557,14.38664986,14.50873672,14.62635896,14.73973559,14.84909405,14.9546573,15.05664042,15.15521792,15.25055536,15.34281424,15.43214657,15.51869796,15.60259447,15.6839602,15.7629174,15.83958365,15.91403965,15.98638412,16.05672319,16.12514483,16.19171952]
  lat_s3 = [78.91670076,78.62834142,78.33983446,78.05119064,77.76241968,77.47352815,77.18452574,76.89542058,76.60621856,76.31692476,76.02754416,75.73808464,75.4485511,75.15894796,74.86927898,74.57954596,74.28975373,73.99990512,73.71000471,73.42005348,73.13005517,72.840013,72.54992933,72.25980407,71.96963852,71.67943759,71.38920304,71.09893582,70.8086364,70.51830666,70.22794829,69.93756269,69.64715082,69.35671533]

  # plot WPs with connecting line
  with ax.hold_limits():
      ax.plot(lon_WP,lat_WP,linewidth=2.0,color='w',linestyle='solid',zorder=12,transform=ccrs.PlateCarree())
      ax.plot(lon_WP,lat_WP,linewidth=1.0,color='c',linestyle='dashed',zorder=12,transform=ccrs.PlateCarree())
      # ax.plot(lon_WP2,lat_WP2,linewidth=2.0,color='w',linestyle='solid',zorder=12,transform=ccrs.PlateCarree())
      # ax.plot(lon_WP2,lat_WP2,linewidth=1.8,color='violet',linestyle=':',zorder=12,transform=ccrs.PlateCarree())
      # ax.plot(lon_FAAM,lat_FAAM,linewidth=2.0,color='w',linestyle='solid',zorder=12,transform=ccrs.PlateCarree())
      # ax.plot(lon_FAAM,lat_FAAM,linewidth=1.0,color='g',linestyle='dashed',zorder=12,transform=ccrs.PlateCarree())
      # ax.plot(lon_HALO,lat_HALO,linewidth=2.0,color='w',linestyle='solid',zorder=12,transform=ccrs.PlateCarree())
      # ax.plot(lon_HALO,lat_HALO,linewidth=1.0,color='r',linestyle='dashed',zorder=12,transform=ccrs.PlateCarree())
      ax.scatter(lon_WP,lat_WP,s=35,color='c',marker='x',zorder=13,transform=ccrs.PlateCarree())
      #ax.scatter(lon_WP2,lat_WP2,s=35,color='violet',marker='x',zorder=13,transform=ccrs.PlateCarree())

  # plot CR22 outline
  #with ax.hold_limits():
  #    ax.plot(lon_f1,lat_f1,linewidth=1.0,color=col,linestyle='dashed',zorder=12,transform=ccrs.PlateCarree())
  #with ax.hold_limits():
  #    ax.plot(lon_f2,lat_f2,linewidth=1.0,color=col,linestyle='solid',zorder=12,transform=ccrs.PlateCarree())

  # plot cross sections
  with ax.hold_limits():
      ax.plot(lon_s1,lat_s1,linewidth=1.5,color='pink',linestyle='dotted', zorder=12,transform=ccrs.Geodetic())
  with ax.hold_limits():
      ax.plot(lon_s2,lat_s2,linewidth=1.5,color='pink',linestyle='dotted', zorder=12,transform=ccrs.Geodetic())
  with ax.hold_limits():
      ax.plot(lon_s3,lat_s3,linewidth=1.5,color='pink',linestyle='dotted', zorder=12,transform=ccrs.Geodetic())
  
  # Joint Viking military restriction
  lon_jv_1 = np.linspace(0,30,200)
  lat_jv_1 = np.ones(200)
  lat_jv_1[:] = 70.5

  lon_jv_2 = np.ones(100)
  lon_jv_2[:] = 16.14
  lat_jv_2 = np.linspace(69.29, 70.5, 100)

  with ax.hold_limits():
      ax.plot(lon_jv_1, lat_jv_1, linewidth=2.0,color='white',linestyle='solid',zorder=12,transform=ccrs.PlateCarree())
      ax.plot(lon_jv_1, lat_jv_1, linewidth=1.5,color='pink',linestyle='solid',zorder=13,transform=ccrs.PlateCarree())
      ax.plot(lon_jv_2, lat_jv_2, linewidth=2.0,color='white',linestyle='solid',zorder=12,transform=ccrs.PlateCarree())
      ax.plot(lon_jv_2, lat_jv_2, linewidth=1.5,color='pink',linestyle='solid',zorder=13,transform=ccrs.PlateCarree())

  # add range circle for Andenes
  p = np.array([16.12,69.31]) # location
  n_points = 50
  angles = np.linspace(0, 360, n_points)


  #d = 300 * 1000 * 1.852  # nautical miles
  d = 1100 * 1000 # 2207km = 1000 km one way (transit speed 5 hours)
  polygon1 = gg.propagate(p, angles, d) # draws a 20 point circle around the location
  #d = 450 * 1000 * 1.852  # nautical miles
  #d = 430 * 1000 # hypothetical 1 h science time (1.5 h transit each way)

  #polygon2 = gg.propagate(p, angles, d) # draws a 20 point circle around the location
  with ax.hold_limits():
      ax.plot(polygon1[:,0],polygon1[:,1],linewidth=1.0,color=col,linestyle='dashdot',zorder=12,transform=ccrs.PlateCarree())
      #ax.plot(polygon2[:,0],polygon2[:,1],linewidth=1.0,color=col,linestyle='dotted',zorder=12,transform=ccrs.PlateCarree())
      ax.scatter([16.0], [69.2667], c='red', marker='^', s = 120, zorder=15, transform = ccrs.PlateCarree())


  # add range circles for coastal weather radars
  # maximum coverage for the reflectivity mode: 240 km (radius of the circle on the maps)
  n_points = 50
  
  angles_rad1a = np.linspace(87, 170, n_points)
  angles_rad1b = np.linspace(-73, -19, n_points)
  angles_rad2 = np.linspace(-99, 155, n_points)
  angles_rad3 = np.linspace(110, 353, n_points)

  #ANDOYA-TROLLTINDEN,69.2413888889,16.0030555556,436
  p1 = np.array([16.00306, 69.24139]) # location
  d = 240 * 1000
  polygon_rad1a = gg.propagate(p1, angles_rad1a, d)
  polygon_rad1b = gg.propagate(p1, angles_rad1b, d)

  #HASVIK-SLUSKFJELLET,70.6069444444,22.4427777778,438
  p2 = np.array([22.44278, 70.60694]) # location
  polygon_rad2 = gg.propagate(p2, angles_rad2, d)

  #RADAR Røst,67.5302777778,12.0988888889,3
  p3 = np.array([12.0989, 67.530278])
  polygon_rad3 = gg.propagate(p3, angles_rad3, d)

  with ax.hold_limits():
      ax.plot(polygon_rad1a[:,0],polygon_rad1a[:,1],linewidth=1.0,color=col,linestyle=(0, (1, 1)),zorder=12,transform=ccrs.PlateCarree())
      ax.plot(polygon_rad1b[:,0],polygon_rad1b[:,1],linewidth=1.0,color=col,linestyle=(0, (1, 1)),zorder=12,transform=ccrs.PlateCarree())
      ax.plot(polygon_rad2[:,0],polygon_rad2[:,1],linewidth=1.0,color=col,linestyle=(0, (1, 1)),zorder=12,transform=ccrs.PlateCarree())
      ax.plot(polygon_rad3[:,0],polygon_rad3[:,1],linewidth=1.0,color=col,linestyle=(0, (1, 1)),zorder=12,transform=ccrs.PlateCarree())

#-----------------------------------------------------------------------
  
  '''
  # add forecasting locations
  sites="/home/marc/Documents/weathervis/weathervis/data/sites.csv"
  locs = pd.read_csv(sites,sep=';')
  with ax.hold_limits():
    ax.scatter(locs["lon"],locs["lat"],s=30,color=col,marker='+',zorder=12,transform=ccrs.PlateCarree())
    #ax.text(locs["lon"],locs["lat"],s=20,color='red',marker='+',zorder=12,transform=ccrs.PlateCarree())
  
  # plot domain outline
  with ax.hold_limits():
    ax.plot(lon,lat,linewidth=1.5,color=col,linestyle='dashed',zorder=12,transform=ccrs.PlateCarree())
  '''
  # add forecasting locations
  sites="/home/marc/Documents/weathervis/weathervis/data/airports.csv"
  locs = pd.read_csv(sites,sep=';')
  with ax.hold_limits():
    ax.scatter(locs["lon"],locs["lat"],s=100,color=col,marker='+',zorder=12,transform=ccrs.PlateCarree())
    #ax.text(locs["lon"],locs["lat"],s=20,color='red',marker='+',zorder=12,transform=ccrs.PlateCarree())
  '''
  # add range circle for Kiruna
  p = np.array([20.31891,67.8222]) # location
  n_points = 50
  angles = np.linspace(0, 360, n_points)
  #d = 300 * 1000 * 1.852  # nautical miles
  #polygon1 = gg.propagate(p, angles, d) # draws a 20 point circle around the location
  d = 450 * 1000 * 1.852  # nautical miles
  polygon2 = gg.propagate(p, angles, d) # draws a 20 point circle around the location
  with ax.hold_limits():
  #  ax.plot(polygon1[:,0],polygon1[:,1],linewidth=1.0,color=col,linestyle='dashdot',zorder=12,transform=ccrs.PlateCarree())
    ax.plot(polygon2[:,0],polygon2[:,1],linewidth=1.0,color=col,linestyle='dotted',zorder=12,transform=ccrs.PlateCarree())
  '''