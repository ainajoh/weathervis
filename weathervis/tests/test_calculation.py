from weathervis.calculation import *
from weathervis.get_data import *

param = ["surface_air_pressure"]
surface_air_pressure = 1000
alt_gl = 300

tv_level = 288
pl = alt_gl2pl(surface_air_pressure, alt_gl, outshape=None )
