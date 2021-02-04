# plotting of maps for Dec 2016 AR

from imetkit.domain import *  # require netcdf4
from imetkit.check_data import *
from imetkit.get_data import *
from imetkit.calculation import *
#from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import warnings
model = "AromeArctic"
param = ["x_wind_10m", "y_wind_10m","wind_speed","wind_direction"]
param_ml = ["x_wind_ml", "y_wind_ml"]
check_all = check_data(date="2020091200", model=model, param=param)
check_all_ml = check_data(date="2020091200", model=model, param=param_ml, levtype="ml")
tmap_meps = get_data(model=model, param=param, file=check_all.file.loc[0], step=0, date="2020091200",levtype="ml")
tmap_meps.retrieve()
print(tmap_meps.height3)

ml_map_meps = get_data(model=model, param=param_ml, file=check_all_ml.file.loc[0], step=0,
                       date="2020091200",levtype="ml", ml_level=[60,64])
print(ml_map_meps.url)
ml_map_meps.retrieve()

u10,v10 = xwind2uwind(tmap_meps.x_wind_10m,tmap_meps.y_wind_10m, tmap_meps.alpha)
wdir10 = wind_dir(tmap_meps.x_wind_10m,tmap_meps.y_wind_10m, tmap_meps.alpha)
ws10 = np.sqrt(u10*u10 + v10*v10)
u,v = xwind2uwind(ml_map_meps.x_wind_ml,ml_map_meps.y_wind_ml, ml_map_meps.alpha)

#wdir = wind_dir(ml_map_meps.x_wind_ml,ml_map_meps.y_wind_ml, ml_map_meps.alpha)180 + (180/np.pi )*np.arctan2(v10,u10)
wdir = ml_map_meps.alpha - (180/np.pi )*np.arctan2(tmap_meps.y_wind_10m[0, 0, :, :], tmap_meps.x_wind_10m[0, 0, :, :]) + 90
print(wdir[0,0])

wind_abs = np.sqrt(u10*u10 + v10*v10)
wind_dir_trig_to = np.arctan2(u10/wind_abs, v10/wind_abs)
wind_dir_trig_to_degrees = wind_dir_trig_to * 180/np.pi
wind_dir_trig_from_degrees = wind_dir_trig_to_degrees + 180 ## 68.38 degrees
wind_dir_cardinal = 90 - wind_dir_trig_from_degrees
print("webb")
print(wind_abs[0,0,0,0])
print(wind_dir_trig_to[0,0,0,0])
print(wind_dir_trig_to_degrees[0,0,0,0])
print(wind_dir_trig_from_degrees[0,0,0,0])
print(wind_dir_cardinal[0,0,0,0])


wdir102 = 180 + (180/np.pi )*np.arctan2(v10,u10)
print("calc from 10m")
print(u10[0,0,0,0])
print(v10[0,0,0,0])
print(ws10[0,0,0,0])
print(wdir10[0,0,0,0])
print(wdir102[0,0,0,0])

ua = -tmap_meps.wind_speed *np.sin(tmap_meps.wind_direction)
va = - tmap_meps.wind_speed *np.cos(tmap_meps.wind_direction)
print("actual")
print(ua[0,0,0,0])
print(va[0,0,0,0])
print(tmap_meps.wind_speed[0,0,0,0])
print(tmap_meps.wind_direction[0,0,0,0])
