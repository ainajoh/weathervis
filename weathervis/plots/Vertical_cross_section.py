
"""
Find model data along points by interpolating between points. 
Points can either be read from a file (if many), or given as input when run (if only two points)
-Given two or more points. 
- Given one point and a degree line

"""

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
from mpl_toolkits.axes_grid1 import make_axes_locatable  ##__N
from weathervis.checkget_data_handler import checkget_data_handler
import warnings
import gc
from util.make_cross_section import *


def plot_Vertical_cross_section():
    pass

def Vertical_cross_section(kwargs):
    H = [
    24122.6894480669,
    20139.2203688489,
    17982.7817599549,
    16441.7123200128,
    15221.9607620438,
    14201.9513633491,
    13318.7065659522,
    12535.0423836784,
    11827.0150898454,
    11178.2217936245,
    10575.9136768674,
    10010.4629764989,
    9476.39726730647,
    8970.49319005479,
    8490.10422494626,
    8033.03285976169,
    7597.43079283063,
    7181.72764002209,
    6784.57860867911,
    6404.82538606181,
    6041.46303718354,
    5693.61312218488,
    5360.50697368367,
    5041.46826162131,
    4735.90067455394,
    4443.27792224573,
    4163.13322354697,
    3895.05391218293,
    3638.67526925036,
    3393.67546498291,
    3159.77069480894,
    2936.71247430545,
    2724.28467132991,
    2522.30099074027,
    2330.60301601882,
    2149.05819142430,
    1977.55945557602,
    1816.02297530686,
    1664.38790901915,
    1522.61641562609,
    1390.69217292080,
    1268.36594816526,
    1154.95528687548,
    1049.75817760629,
    952.260196563843,
    861.980320753114,
    778.466725603312,
    701.292884739207,
    630.053985133223,
    564.363722589458,
    503.851644277509,
    448.161118360263,
    396.946085973573,
    349.869544871297,
    306.601457634038,
    266.817025119099,
    230.194566908004,
    196.413229972062,
    165.151934080260,
    136.086183243070,
    108.885366240509,
    83.2097562375566,
    58.7032686584901,
    34.9801888163106,
    11.6284723290378,
]
    param = ["air_temperature_ml","surface_geopotential","surface_air_pressure"]
    level =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64],

    kwargs.m_level = np.arange(0, 65, 1) #[0,1,2]
    nya = (78.9243,11.9312)
    ands = (69.310, 16.120)
    coos2 = [((nya[1], nya[0]), (ands[1], ands[0]))]
    coos2 =[((-4.770555, 64.6152131), (41.700291, 79.477055))]
    print(coos2[0])
    K = [0]
    stnam = ['nya']
    endnam = ['ands']
    model = "AromeArctic"
    clat = -23.6
    plon = -25.0
    plat = 77.5
    fignam = ['nya-ands']


    dmet, data_domain, bad_param = checkget_data_handler(all_param=param, model=kwargs.model,  date=kwargs.datetime,
                                                         step=kwargs.steps, m_level=kwargs.m_level,
                                                         point_name=kwargs.point_name, num_point=kwargs.num_point)
    # give the height interval that the field is goid to be interpolated to. currently lowest 5000 m in 100m steps


    heights = np.arange(0, 5000, 100) 
    dmet.air_temperature_ml = dmet.air_temperature_ml[0,:,:,:]

    P = ml2pl(dmet.ap, dmet.b, dmet.surface_air_pressure)
    P = P[0, :, :, :]
    # calculate the crosssection
    re = 6.3781 * 10**6
    gph = dmet.surface_geopotential[0, 0, :, :]
    g = 9.81  # gravitational acceleration in m s^-2
    hsurf = re * gph / (g * re - gph)
    dimz = np.shape(dmet.air_temperature_ml)[0]
    print(dimz)
    HH = np.zeros(np.shape(dmet.air_temperature_ml))
    for i, z in enumerate(H[65 - dimz : 65]):
            HH[i, :, :] = z

    cross = CrossSection(
                {
                    "lat": dmet.latitude,
                    "rlat": dmet.y,
                    "lon": dmet.longitude,
                    "rlon": dmet.x,
                    "TT": dmet.air_temperature_ml,
                    "z": HH + hsurf,
                    "p": P / 100.0
                },
                coos2[0],
                heights,
                version="rotated",
                pollon=plon,
                pollat=plat,
                flip=True,
                int2z=True,
                model=model,
            )  # polgam=180,

    x, zi = np.meshgrid(cross.lon, cross.pressure)
    fig, ax = plt.subplots()    
    pc = ax.contourf( x, zi, cross.TT, cmap="gnuplot2_r", extend="both" )
    plt.show()

if __name__ == "__main__":
    args = default_arguments()

    chunck_func_call(func=Vertical_cross_section,chunktype= args.chunktype, chunk=args.chunks,  kwargs=args)
    gc.collect()

