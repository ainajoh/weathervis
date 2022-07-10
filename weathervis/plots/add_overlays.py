import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import shapely as sh
import numpy as np

def add_overlay(types=[None], **kwargs): #for quick on/off in default map
  print("in add_overlays")
  for ty in types:
    if ty=="ISLAS":
      add_ISLAS_overlays(**kwargs)
    #if ty=="topinfotext":
    #  add_topinfotext(**kwargs)
    #  #add_topinfotext(ax, model, datetime, leadtime, plot_name="missin_name", **kwargs)

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

  lon=[10, 0, 0, 5,10.5,20.1,22,29,29.5,10]
  lat=[79,72,60,57,58.5,58.5,65,65,79  ,79]
  # Coordinates of boundary points of the flight area :
  #1  :  N72Â° - E000Â2  :  N60Â° - E000Â°3  :  N57Â° - E005Â°
  #4  :  N58Â°3 E010Â°3	5  :  N58Â°3 E020Â6  :  N65Â° - E022Â°
  #7  :  N65Â° - E029Â°	8  :  N79Â° - E0299  :  N79Â° - E010Â°

  # plot domain outline
  with ax.hold_limits():
    ax.plot(lon,lat,linewidth=1.5,color=col,linestyle='dashed',zorder=12,transform=ccrs.PlateCarree())

  # add forecasting locations
  sites="../data/sites.csv"
  locs = pd.read_csv(sites,sep=';')
  with ax.hold_limits():
    ax.scatter(locs["lon"],locs["lat"],s=30,color=col,marker='.',zorder=12,transform=ccrs.PlateCarree())
    #ax.text(locs["lon"],locs["lat"],s=20,color='red',marker='+',zorder=12,transform=ccrs.PlateCarree())

  # add forecasting locations
  sites="../data/airports.csv"
  locs = pd.read_csv(sites,sep=';')
  with ax.hold_limits():
    ax.scatter(locs["lon"],locs["lat"],s=100,color=col,marker='+',zorder=12,transform=ccrs.PlateCarree())
    #ax.text(locs["lon"],locs["lat"],s=20,color='red',marker='+',zorder=12,transform=ccrs.PlateCarree())

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