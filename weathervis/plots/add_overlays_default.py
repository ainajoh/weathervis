'''
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import pandas as pd
import shapely as sh
import numpy as np
import geog as gg

def add_ISLAS_overlays(ax,col='red'):

  lon=[10, 0, 0, 5,10.5,20.1,22,29,29.5,10]
  lat=[79,72,60,57,58.5,58.5,65,65,79  ,79]
  # Coordinates of boundary points of the flight area : 
  #1  :  N72° - E000�2  :  N60° - E000°3  :  N57° - E005°
  #4  :  N58°3 E010°3	5  :  N58°3 E020�6  :  N65° - E022°
  #7  :  N65° - E029°	8  :  N79° - E0299  :  N79° - E010°

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
  lat_WP = [67.5, 74, 75.5, 75.5, 77.8, 78.25] 
  lon_WP = [20.18, 19, 21,  16,   11, 15.48]
  # flight back
  lat_WP2 = [78.25, 79, 77.7, 77.65, 77.6, 77.45, 77.3, 75.5, 75.4, 67.5 ] 
  lon_WP2 = [15.48, 12, 12.5,    12.3,    9,    11.6,   9.5,   18.5, 13,   20.18]
#  lat_WP = [67.82,70.5,72.26,72.4,72.32,72.26,71.24,71.1,71.05,71.24]
#  lon_WP = [20.31,18.0,18.82,19.03,19.56,18.82,17.76,17.63,18.22,17.76]
  #Kiruna;67.8222;20.31891
#  lat_FAAM = [67.82,73.0,75.0, 72.75,72.75]
#  lon_FAAM = [20.31,19.0,22.0, 18.5, 21.0]
#
#  lat_HALO = [67.821,69.687,70.302,72.156,73.164,72.255,71.198,72.271,73.018,75.16,72.239,71.278,72.394,74.427,75.492,73.372,83.5,83.667,84.167,84.5,84.167,83.667,83.5,69.801,67.821]
#
#  lon_HALO = [20.336, 21.62, 22.172, 8.539, 9.758, 17.987, 17.421, 18.013, 19.001, 22.319, 17.962, 23.788, 25.836, 11.753, 13.498, 28.037, 18.5, 22.333, 22.667, 17.667, 13.833, 14.333, 18.333, 20.244, 20.336]

  # cross-section coordinates
  lon_s1=[12.1,12.23520606,12.65944174,13.21400478,13.58760356,13.94094611,14.33871462,
      14.65491706,15.03016339,15.37328566,15.64192952,15.89794657,16.19885511,
      16.4963865 ,16.71676549,16.98148237,17.18253143,17.43431235,17.66910028,
      17.84448999,18.01302922,18.27786088,18.43192251,18.58029156,18.76995314,
      18.95664375,19.08803228,19.25955956,19.42811494,19.54508722,19.70103341,
      19.85399797,19.95863716,20.06000296,20.24056098,20.18]
  lat_s1=[78.93,78.9201046 ,78.57888799,78.24135548,77.89877258,77.5557036 ,77.22982397,
      76.88585191,76.5273737 ,76.20003903,75.85499837,75.50972759,75.18141804,
      74.82113513,74.47523171,74.14616744,73.80000724,73.43894861,73.10939372,
      72.763008  ,72.4166229 ,72.07180271,71.72538868,71.37903104,71.04917438,
      70.68772372,70.3415569 ,70.01178108,69.6505138 ,69.3046819 ,68.97513427,
      68.61420174,68.26883362,67.92366473,67.57918425,67.5]
  lon_s2=[20.18,20.28568132,20.49090086,20.73650784,21.02311294,21.26856487,21.47283874,
      21.71810188,21.96334317,22.2486139 ,22.45183297,22.69661679,22.9413448,
      23.2252967 ,23.46970516,23.67152584,23.91560771,24.19809262,24.44178398,
      24.68535977,24.88561967,25.12879474,25.4093394 ,25.65203706,25.85088216,
      26.09312171,26.37188085,26.61357582,26.85508309,27.05209295,27.29307225,
      27.56950503,27.80985806,28.04998496,28.24501692,28.27]
  lat_s2=[67.5,67.46953351,67.45478106,67.45554947,67.44025305,67.44016759,67.42410167,
      67.42329479,67.42209443,67.40450638,67.38713581,67.38475896,67.38198961,
      67.36258133,67.35896473,67.33997587,67.33564455,67.31442918,67.30925487,
      67.30369047,67.28309965,67.27682545,67.2533688 ,67.24625761,67.22440035,
      67.21658417,67.19135463,67.18270726,67.17367437,67.15024994,67.14051858,
      67.11309344,67.10253877,67.09160174,67.06663179,67.1]
  lon_s3=[4.41, 4.44953925 ,5.0407378  ,5.53749556 ,6.08798687 ,6.5772435 , 7.11912821,
      7.63533681, 8.16884073, 8.64304204, 9.16768893, 9.68759595,10.18598354,
     10.69721632,11.15171019,11.65391332,12.10042551,12.63114747,13.06987576,
     13.55419437,13.98490735,14.49862724,14.92151709,15.38792679,15.80283199,
     16.29943515,16.75224481,17.15515267,17.59904567,18.03386792,18.46903825,
     18.8563932 ,19.2828046 ,19.70266439,20.12049871,20.18]
  lat_s3=[70.55,70.51902294,70.46049417,70.3917416 ,70.31032248,70.23864125,70.15399274,
     70.09841456,70.01050593,69.93307996,69.84210155,69.74953743,69.68640955,
     69.59078609,69.50653244,69.40804363,69.3212549 ,69.23784325,69.14848771,
     69.04439499,68.95264677,68.86334481,68.7691707 ,68.65979221,68.5633659,
     68.46848725,68.35525772,68.25542252,68.13984352,68.05444846,67.93648026,
     67.83245017,67.71228247,67.62237927,67.49997071,67.5]

  # plot WPs with connecting line
  with ax.hold_limits():
    ax.plot(lon_WP,lat_WP,linewidth=2.0,color='w',linestyle='solid',zorder=12,transform=ccrs.PlateCarree())
    ax.plot(lon_WP,lat_WP,linewidth=1.0,color='c',linestyle='dashed',zorder=12,transform=ccrs.PlateCarree())
    ax.plot(lon_WP2,lat_WP2,linewidth=2.0,color='w',linestyle='solid',zorder=12,transform=ccrs.PlateCarree())
    ax.plot(lon_WP2,lat_WP2,linewidth=1.8,color='violet',linestyle=':',zorder=12,transform=ccrs.PlateCarree())
   # ax.plot(lon_FAAM,lat_FAAM,linewidth=2.0,color='w',linestyle='solid',zorder=12,transform=ccrs.PlateCarree())
   # ax.plot(lon_FAAM,lat_FAAM,linewidth=1.0,color='g',linestyle='dashed',zorder=12,transform=ccrs.PlateCarree())
   # ax.plot(lon_HALO,lat_HALO,linewidth=2.0,color='w',linestyle='solid',zorder=12,transform=ccrs.PlateCarree())
   # ax.plot(lon_HALO,lat_HALO,linewidth=1.0,color='r',linestyle='dashed',zorder=12,transform=ccrs.PlateCarree())
    ax.scatter(lon_WP,lat_WP,s=35,color='c',marker='x',zorder=13,transform=ccrs.PlateCarree())
    ax.scatter(lon_WP2,lat_WP2,s=35,color='violet',marker='x',zorder=13,transform=ccrs.PlateCarree())

  # plot domain outline
  with ax.hold_limits():
    ax.plot(lon,lat,linewidth=1.5,color=col,linestyle='dashed',zorder=12,transform=ccrs.PlateCarree())

  # plot CR22 outline
  with ax.hold_limits():
    ax.plot(lon_f1,lat_f1,linewidth=1.0,color=col,linestyle='dashed',zorder=12,transform=ccrs.PlateCarree())
  with ax.hold_limits():
    ax.plot(lon_f2,lat_f2,linewidth=1.0,color=col,linestyle='solid',zorder=12,transform=ccrs.PlateCarree())

  # plot cross sections
  with ax.hold_limits():
    ax.plot(lon_s1,lat_s1,linewidth=1.5,color=col,linestyle='dotted',zorder=12,transform=ccrs.PlateCarree())
  with ax.hold_limits():
    ax.plot(lon_s2,lat_s2,linewidth=1.5,color=col,linestyle='dotted',zorder=12,transform=ccrs.PlateCarree())
  with ax.hold_limits():
    ax.plot(lon_s3,lat_s3,linewidth=1.5,color=col,linestyle='dotted',zorder=12,transform=ccrs.PlateCarree())

  # add forecasting locations
  sites="../../data/sites.csv"
  locs = pd.read_csv(sites,sep=';')
  with ax.hold_limits():
    ax.scatter(locs["lon"],locs["lat"],s=30,color=col,marker='.',zorder=12,transform=ccrs.PlateCarree())
    #ax.text(locs["lon"],locs["lat"],s=20,color='red',marker='+',zorder=12,transform=ccrs.PlateCarree())

  # add forecasting locations
  sites="../../data/airports.csv"
  locs = pd.read_csv(sites,sep=';')
  with ax.hold_limits():
    ax.scatter(locs["lon"],locs["lat"],s=100,color=col,marker='+',zorder=12,transform=ccrs.PlateCarree())
    #ax.text(locs["lon"],locs["lat"],s=20,color='red',marker='+',zorder=12,transform=ccrs.PlateCarree())

  # add ship track
  ship="/home/centos/ship/marinetraffic_positions"
  locs = pd.read_csv(ship,sep=',',skipinitialspace=True)
  with ax.hold_limits():
    ax.plot(locs["LON"],locs["LAT"],'-',color=col,marker='.',zorder=12,transform=ccrs.PlateCarree())
 
  # add range circle for Kiruna
  p = sh.geometry.Point([20.31891,67.8222]) # location
  n_points = 50
  angles = np.linspace(0, 360, n_points)
  d = 300 * 1000 * 1.852  # nautical miles
  polygon1 = gg.propagate(p, angles, d) # draws a 20 point circle around the location
  d = 450 * 1000 * 1.852  # nautical miles
  polygon2 = gg.propagate(p, angles, d) # draws a 20 point circle around the location
  with ax.hold_limits():
    ax.plot(polygon1[:,0],polygon1[:,1],linewidth=1.0,color=col,linestyle='dashdot',zorder=12,transform=ccrs.PlateCarree())
    ax.plot(polygon2[:,0],polygon2[:,1],linewidth=1.0,color=col,linestyle='dotted',zorder=12,transform=ccrs.PlateCarree())
'''

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import pandas as pd
import shapely as sh
import numpy as np
import geog as gg

def add_ISLAS_overlays(ax,col='red'):

    lon=[10, 0, 0, 5,10.5,20.1,22,29,29.5,10]
    lat=[79,72,60,57,58.5,58.5,65,65,79  ,79]
    # Coordinates of boundary points of the flight area : 
    #1  :  N72Â° - E000Â2  :  N60Â° - E000Â°3  :  N57Â° - E005Â°
    #4  :  N58Â°3 E010Â°3	5  :  N58Â°3 E020Â6  :  N65Â° - E022Â°
    #7  :  N65Â° - E029Â°	8  :  N79Â° - E0299  :  N79Â° - E010Â°

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

    # plot domain outline
    #with ax.hold_limits():
     #   ax.plot(lon,lat,linewidth=1.5,color=col,linestyle='dashed',zorder=12,transform=ccrs.PlateCarree())

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

    # add forecasting locations
    #sites="../../data/sites.csv"
    #locs = pd.read_csv(sites,sep=';')
    #with ax.hold_limits():
     #   ax.scatter(locs["lon"],locs["lat"],s=30,color=col,marker='.',zorder=12,transform=ccrs.PlateCarree())
        #ax.text(locs["lon"],locs["lat"],s=20,color='red',marker='+',zorder=12,transform=ccrs.PlateCarree())

    # add forecasting location
    sites="../../data/airports.csv"
    locs = pd.read_csv(sites,sep=';')
    with ax.hold_limits():
        ax.scatter(locs["lon"],locs["lat"],s=50,color=col,marker='+',zorder=12,transform=ccrs.PlateCarree())
        #ax.text(locs["lon"],locs["lat"],s=20,color='red',marker='+',zorder=12,transform=ccrs.PlateCarree())

    #### add ship track
    ###ship="/home/centos/ship/marinetraffic_positions"
    ###locs = pd.read_csv(ship,sep=',',skipinitialspace=True)
    ###with ax.hold_limits():
    ###    ax.plot(locs["LON"],locs["LAT"],'-',color=col,marker='.',zorder=12,transform=ccrs.PlateCarree())
     
    # add range circle for Andenes
    p = np.array([16.12,69.31]) # location
    n_points = 50
    angles = np.linspace(0, 360, n_points)


    #d = 300 * 1000 * 1.852  # nautical miles
    d = 1100 * 1000 # 2207km = 1000 km one way (transit speed 5 hours)
    polygon1 = gg.propagate(p, angles, d) # draws a 20 point circle around the location
    #d = 450 * 1000 * 1.852  # nautical miles
    d = 430 * 1000 # hypothetical 1 h science time (1.5 h transit each way)

    polygon2 = gg.propagate(p, angles, d) # draws a 20 point circle around the location
    with ax.hold_limits():
        ax.plot(polygon1[:,0],polygon1[:,1],linewidth=1.0,color=col,linestyle='dashdot',zorder=12,transform=ccrs.PlateCarree())
        #ax.plot(polygon2[:,0],polygon2[:,1],linewidth=1.0,color=col,linestyle='dotted',zorder=12,transform=ccrs.PlateCarree())
        ax.scatter([16.0], [69.2667], c='red', marker='^', s = 120, zorder=15, transform = ccrs.PlateCarree())

    # add range circles for coastal weather radars
    # maximum coverage for the reflectivity mode: 240 km (radius of the circle on the maps)
    
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



