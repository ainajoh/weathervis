'''
Containing useful functions
'''

import datetime as dt
import numpy as np
import math
import cartopy.crs as ccrs  #add_point_on_map, default_map_projection
from pyproj import Geod

####################################################################################################################
# UTILITIES
#####################################################################################################################
def great_circle_distance(lon1, lat1, lon2, lat2, R=6378.137):
    """Calculate the great circle distance between two points

    based on : https://gist.github.com/gabesmed/1826175


    Parameters
    ----------

    lon1: float
        longitude of the starting point
    lat1: float
        latitude of the starting point
    lon2: float
        longitude of the ending point
    lat2: float
        latitude of the ending point

    Returns
    -------

    distance (km): float

    Examples
    --------

    >>> great_circle_distance(0, 55, 8, 45.5)
    1199.3240879770135
    """

    #R = 6371.0 # 6378137  # earth circumference in km

    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = (np.sin(dLat / 2) * np.sin(dLat / 2) +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dLon / 2) * np.sin(dLon / 2))
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    return distance 
def richardson_bulk(potential_temperature, height, u, v):
    b=100; ust = 0.1 #just to play with flexpart addons
    g=9.81
    #potential_temperature = potential_temperature.squeeze()
    #u = u.squeeze()
    #v = v.squeeze()
    #height = height.squeeze()


    pt_ref = potential_temperature[:,-1,:,:]
    height_ref = height[:,-1,:,:]
    u_ref = u[:,-1,:,:]
    v_ref = v[:,-1,:,:]
    
    buoy =  g/pt_ref * (potential_temperature-pt_ref)* (height-height_ref) 
    shear = (u - u_ref)**2 + (v-v_ref)**2
    shear_addon =  b*ust**2
    ri = buoy/shear
    #*(height(indzp)-height(indzp-1))/ g/pt_ref * 
    #max((uprof(indzp)-uprof(indzp-1))**2 + (vprof(indzp)-vprof(indzp-1))**2 + b*ust**2, 0.1)
    return ri


def calc_obukhov(T1M,ps,tsurf,H, ustress,vstress,ak1,bk1 ):#P1M ):
    r_air=284.
    cpa=1004.
    karman=0.4
    ga=9.81
    tv=   tsurf            #temp at surface-best if virtual
    rhoa=ps/(r_air*tv)   #air density
    #plev=P1M  #pressure at 1 model level: =ak(1)+bk(1)*ps 
    plev = ak1+bk1*ps 
    theta=T1M*(100000./plev)**(r_air/cpa) # potential temperature
    stress = np.sqrt(ustress**2 + vstress**2)  #surface stress N/m^2
    ustar = np.sqrt(abs(stress)/rhoa)
    #ustar = ustar.where(ustar >= 0, 1e-8)
    ustar[ustar < 0] =  1e-8
    thetastar=-H/(rhoa*cpa*ustar)

    #obukhov=obukhov.where()theta*ustar**2/(karman*ga*thetastar)
    #if(abs(thetastar) >.1e-10): 
    ol=theta*ustar**2/(karman*ga*thetastar)
    ol[abs(thetastar)<1e-10] = np.nan
    #ol=ol.where(abs(thetastar)>1e-10, np.nan)

    return ol


def richardson(potential_temperature, height, u, v):
    g=9.81
    ri = np.zeros(np.shape(potential_temperature))
    print(len(height[0,:,0,0]))
    print(np.shape(height))
    for k in range(0, len( height[0,:,0,0] )-1): #remember k direction is from top to down so k=0 is top
        #so k+1 is under k
        buoy =  (g/potential_temperature[:,k+1,:,:]) * (potential_temperature[:,k,:,:]-potential_temperature[:,k+1,:,:])* ( height[:,k,:,:] - height[:,k+1,:,:] ) 
        shear = ( u[:,k,:,:] - u[:,k+1,:,:])**2 + ( v[:,k,:,:] - v[:,k+1,:,:] )**2
        ri[:,k,:,:] = buoy/shear
    #*(height(indzp)-height(indzp-1))/ g/pt_ref * 
    #max((uprof(indzp)-uprof(indzp-1))**2 + (vprof(indzp)-vprof(indzp-1))**2 + b*ust**2, 0.1)
    return ri

def nearest_neighbour_idx(plon,plat, longitudes, latitudes, nmin=1):
    """
    Parameters
    ----------
    plon: longitude of a specific location [degrees]
    plat: latitude of a specific location  [degrees]
    longitudes: all longitudes of the model[degrees]
    latitudes: all latitudes of the model  [degrees]
    nmin: number of points you want nearest to your specific location

    Returns
    -------
    indexes as tuples in array for the closest gridpoint near a specific location.
    point = [(y1,x1),(y2,x2)]. This format is done in order to ease looping through points.
    for p in point:
        #gies p = (y1,x1)
        xatlocation = x_wind_10m[:,0,p]
    """
    #source https://github.com/metno/NWPdocs/wiki/From-x-y-wind-to-wind-direction
    print("inside nearest_neighbour_idx")
    d = great_circle_distance(plon,plat, longitudes, latitudes, R=6371.0)
    dsort = np.sort(d,axis=None)
    closest_idx = np.where(np.isin(d,dsort[0:nmin]))
    #point = [(x,y) for x,y in zip(closest_idx[0],closest_idx[1])]
    return closest_idx

def nearest_neighbour(plon,plat, longitudes, latitudes, nmin=1):
    """
    Parameters
    ----------
    plon: longitude of a specific location [degrees]
    plat: latitude of a specific location  [degrees]
    longitudes: all longitudes of the model[degrees]
    latitudes: all latitudes of the model  [degrees]
    nmin: number of points you want nearest to your specific location

    Returns
    -------
    indexes as tuples in array for the closest gridpoint near a specific location.
    point = [(y1,x1),(y2,x2)]. This format is done in order to ease looping through points.
    for p in point:
        #gives p = (y1,x1)
        xatlocation = x_wind_10m[:,0,p]
    """
    #source https://github.com/metno/NWPdocs/wiki/From-x-y-wind-to-wind-direction
    closest_idx = nearest_neighbour_idx(plon,plat, longitudes, latitudes, nmin=nmin)
    point = [(x,y) for x,y in zip(closest_idx[0],closest_idx[1])]

    return point

def rotate_points(lon, lat, center_longitude, center_latitude, parallels, model='MEPS', direction='n2r', pollon=None,pollat=None ):
    """Rotate lon, lat from/to a rotated system

    Parameters
    ----------
    center_longitude: float
        longitudinal coordinate of the rotated center
    center_latitude: float
        latitudinal coordinate of the rotated center
    lon: array (1d)
        longitudinal coordinates to rotate
    lat: array (1d)
        latitudinal coordinates to rotate
    direction: string, optional
        direction of the rotation;
        n2r: from non-rotated to rotated (default)
        r2n: from rotated to non-rotated
    model: string, optional
    	used model

    Returns
    -------
    rlon: array
    rlat: array
    """
    lon = np.array(lon)
    lat = np.array(lat)

    if model =="MEPS":
        parallels = (63.3,63.3)
    elif model== "AromeAecric":
        parallels= (77.5,77.5)

    globe = ccrs.Globe(semimajor_axis=6371000.)

    rotatedgrid = ccrs.LambertConformal(central_longitude=center_longitude, central_latitude=center_latitude, standard_parallels=parallels,
                                globe=globe)

    ''' If u do not have center lon lat and parallels, we can rotate with knowing the lonlat of the poles. 
     In that case alter this function to allow this and uncomment bellow
    rotatedgrid = ccrs.RotatedPole(
        pole_longitude=pollon,
        pole_latitude=pollat
    )
    '''
    standard_grid = ccrs.Geodetic()

    if direction == 'n2r':
        rotated_points = rotatedgrid.transform_points(standard_grid, lon, lat)
    elif direction == 'r2n':
        rotated_points = standard_grid.transform_points(rotatedgrid, lon, lat)

    rlon, rlat, _ = rotated_points.T
    return rlon, rlat

def find_cross_points(coo, nbre=1):
    """ give nbre points along a great circle between coo[0] and coo[1]

    Parameters
    ----------
    coo : list or numpy.ndarray
        If `coo` is a list of coordinates, it is used of the starting and
        end points: [(startlon, startlat), (endlon, endlat)]
        If `coo` is a numpy.ndarray it is used as the cross-section points.
        coo need then to be similar to :
        np.array([[10, 45], [11, 46], [12, 47]])
    Returns
    ----------
    """
    g = Geod(ellps='WGS84')
    cross_points = g.npts(coo[0][0], coo[0][1], coo[1][0], coo[1][1], nbre)
    lat = np.array([point[1] for point in cross_points])
    lon = np.array([point[0] for point in cross_points])
    distance = great_circle_distance(coo[0][0], coo[0][1],
                                     coo[1][0], coo[1][1])

    return lon, lat, distance
    
def get_AllPointsBetween2PosOnGrid(coo, center_longitude, center_latitude, parallels, pollon=None,pollat=None, model=None, nbre=10,version="regular"):
    """
    Both adds mode points alonga great circle between two locations and
     rotates (depending on input values; read about each parameter under)

    Gets the points in between two points if specified, otherwise uses the raw values of coo
    Return rotated or regular(no change of coo) gridpoints coordinates from original coordinate points coo
    Returns distance of the entire line that coo spans out. 
    
    Parameters
    ----------
    coo : list or numpy.ndarray
        If `coo` is a list of coordinates, it is used of the starting and
        end points: [(startlon, startlat), (endlon, endlat)]
        If `coo` is a numpy.ndarray it is used as the cross-section points.
        coo need then to be similar to :
        np.array([[10, 45], [11, 46], [12, 47]])
    """
    if type(coo) in [list, tuple]:
        # find the coordinate of the cross-section
        lon, lat, distance = find_cross_points(coo, nbre)
        #add end points
        lon=np.append(coo[0][0],lon)
        lon=np.append(lon,coo[1][0])
        lat=np.append(coo[0][1],lat)
        lat=np.append(lat,coo[1][1] )
        if version == 'rotated':
            crlon, crlat = rotate_points(lon=lon, lat=lat,model=model,center_longitude=center_longitude,center_latitude=center_latitude, parallels=parallels,pollon=pollon, pollat=pollat  ) 
        elif version == 'regular':
            crlon, crlat = lon, lat
        
    elif type(coo) == np.ndarray: #already defined points that we want with no need for finding points along lines
        lon = coo[:, 0]
        lat = coo[:, 1]
        nbre = coo.shape[0]
        if version == 'rotated':
            crlon, crlat = rotate_points(lon=lon, lat=lat,model=model,center_longitude=center_longitude,center_latitude=center_latitude, parallels=parallels,pollon=pollon, pollat=pollat  ) 
        elif version == 'regular':
            crlon, crlat = lon, lat
        #TODO: if points are not along a great circle then the distance is not really correct. 
        # More correct would be calculating great circle distance between each point and adding upp. 
        distance = great_circle_distance(coo[0, 0], coo[0, 1],
                                        coo[-1, 0], coo[-1, 1])
    else:
        msg = '<coo> should be of type list, tuple or nump.ndarray'
        raise TypeError(msg)


    distances = np.linspace(0, distance, nbre+2) # distance from start to every point
    query_points = [[lat, lon] for lat, lon in zip(crlat, crlon)]
    return query_points, distances

def interpolate(data, grid, interplevels):
    """interpolate `data` on `grid` for given `interplevels`

    Interpolate the `data` array at every given levels (`interplevels`) using
    the `grid` array as reference.

    The `grid` array need to be increasing along the first axis.
    Therefore ERA-Interim pressure must be flip: p[::-1, ...], but not
    COSMO pressure since its level 0 is located at the top of the atmosphere.

    Parameters
    ----------
    data : array (nz, nlat, nlon)
        data to interpolate
    grid : array (nz, nlat, nlon)
        grid use to perform the interpolation
    interplevels: list, array
        list of the new vertical levels, in the same unit as the grid

    Returns
    -------
    interpolated array: array (len(interplevels), nlat, nlon)

    Examples
    --------

    >>> print(qv.shape)
    (60, 181, 361)
    >>> print(p.shape)
    (60, 181, 361)
    >>> levels = np.arange(200, 1050, 50)
    (17,)
    >>> qv_int = interpolate(qv, p, levels)
    >>> print(qv_int.shape)
    (17, 181, 361)

    """
    data = data.squeeze()
    grid = grid.squeeze()
    shape = list(data.shape)
    if (data.ndim > 3) | (grid.ndim > 3):
        message = "data and grid need to be 3d array"
        raise IndexError(message)

    try:
        nintlev = len(interplevels)
    except:
        interplevels = [interplevels]
        nintlev = len(interplevels)
    print(shape)
    shape[-3] = nintlev

    outdata = np.ones(shape) * np.nan
    if nintlev > 20:
        for idx, _ in np.ndenumerate(data[0]):
            column = grid[:, idx[0], idx[1]]
            column_GRID = data[:, idx[0], idx[1]]

            value = np.interp(
                interplevels,
                column,
                column_GRID,
                left=np.nan,
                right=np.nan)
            outdata[:, idx[0], idx[1]] = value[:]
    else:
        for j, intlevel in enumerate(interplevels):
            for lev in range(grid.shape[0]):
                cond1 = grid[lev, :, :] > intlevel
                cond2 = grid[lev - 1, :, :] < intlevel
                right = np.where(cond1 & cond2)
                if right[0].size > 0:
                    sabove = grid[lev, right[0], right[1]]
                    sbelow = grid[lev - 1, right[0], right[1]]
                    dabove = data[lev, right[0], right[1]]
                    dbelow = data[lev - 1, right[0], right[1]]
                    result = (intlevel - sbelow) / (sabove - sbelow) * \
                             (dabove - dbelow) + dbelow
                    outdata[j, right[0], right[1]] = result
    return outdata

def interpolate_grid(**kwargs):
    from util.intergrid import Intergrid
    intfunc = Intergrid(**kwargs)
    return intfunc

def CAO_index(air_temperature_pl, pressure, SST,air_pressure_at_sea_level, p_level=850):
    #pressure in hpa
    #pt = potential_temperatur(air_temperature_pl, pressure)
    #pt_sst = potential_temperatur(SST, air_pressure_at_sea_level)

    #dpt_sst = pt_sst[:, :, :] - pt[:, np.where(pressure == p_level)[0], :, :].squeeze()
    
    pt = potential_temperature(air_temperature_pl, pressure*100)  #4, 2, 36, 36)

    #try: 
    #     air_pressure_at_sea_level= air_pressure_at_sea_level.squeeze(axis=1)
    #except: 
    #    air_pressure_at_sea_level = air_pressure_at_sea_level
    pt_sst = potential_temperature(SST, air_pressure_at_sea_level)
    #print(np.shape(SST))   #(4, 36, 36)
    #print(np.shape(air_pressure_at_sea_level))  #(4, 1, 36, 36)
    #print(np.shape(pt_sst))   #(4, 4, 36, 36) but #(4, 36, 36) if squeezing air.pressure
    dpt_sst = pt_sst[:, :, :] - pt[:, np.where(pressure == p_level)[0], :, :].squeeze()
    return dpt_sst

def get_samplesize(q, rho, a=0.5, b = 0.95, acc = 3):
    """
    Estimating sample size on field campaign
    Parameters
    ----------
    q   [Specific humidity kg/kg]
    rho [density [kg/m^3]]
    a   [Andrew Seidl provided this factor]
    b   [Andrew Seidl provided this factor]
    acc [x hour acc precip]

    Returns
    -------
    DOUBLE CHECK ALL THE UNITS AFTER UPDATE BEFORE USE.
    """
    #rho = rho #/ 1000 #kg/m3 to kg/L
    q = q #*1000# g/g to g/kg
    samplesize = q * rho * a * b * 60 #per hour
    samplesize=samplesize# /1000 #per Liter?
    samplesize_acc = np.full(np.shape(samplesize), np.nan)
    for step in range(acc-1, np.shape(samplesize)[0]):
        s_acc = 0
        i = 0
        while i < acc:
            s_acc += samplesize[ step + i - ( acc - 1 ),:]
            i+=1
        samplesize_acc[step,:,:,:] = s_acc #g

    return samplesize_acc#.squeeze()# samplesize_acc.squeeze()
def precip_acc(precip, acc=1):
    """

    Parameters
    ----------
    precip: model precip that is accumulating with respect to the forecast
    acc: The prefered precip accumulation

    Returns
    -------
    precip [ mm / acc hours]
    """
    precipacc = np.full(np.shape(precip), np.nan)
    #precipacc = np.zeros(np.shape(precip))
    for t in range(0 + acc, np.shape(precip)[0] ):
        precipacc[t, 0, :, :] = precip[t, 0, :, :] - precip[t - acc, 0, :, :]
        #Set negative values to 0, but I fixed it in plot instead.
    #precipacc = np.ma.masked_where(precipacc ==np.nan, precipacc)

    return precipacc
def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier
def timestamp2utc(timestamp):
    time_utc = [dt.datetime.utcfromtimestamp(x) for x in timestamp]
    return time_utc
####################################################################################################################
# THERMODYNAMICS
#####################################################################################################################
def potential_temperature(temperature, pressure):
    """
    Parameters
    ----------
    temperature [K]
    pressure    [Pa]

    Returns
    -------
    Potential temperature [K]
    """
    p0 = 100000  #[Pa]
    Rd = 287.05  #[J/kg K] Gas constant for dry air
    cp = 1004.  #[J/kg] specific heat for dry air (WH)
    theta = np.full(np.shape(temperature), np.nan)
    #print(np.shape(theta))
    print("hey")
    print(len(np.shape(theta))) #3
    print(len(np.shape(pressure))) #4
    print(np.shape(theta)) #3 (6, 36, 35)
    print(np.shape(pressure)) #4 (6, 1, 36, 35)
    if len(np.shape(theta)) ==4:
        if len(np.shape(pressure)) ==1:
            for i in range(0,len(pressure)):
                theta[:,i,:,:] = temperature[:,i,:,:]  * (p0 / pressure[i]) ** (Rd/cp) #[K]
        else:
            for i in range(0,np.shape(pressure)[1]):
                theta[:,i,:,:] = temperature[:,i,:,:]  * (p0 / pressure[:,i,:,:]) ** (Rd/cp) #[K]

    
    elif len(np.shape(theta)) ==1:
        for i in range(0,len(pressure)):
            theta[i] = temperature[i]  * (p0 / pressure[i]) ** (Rd/cp) #[K]
    elif len(np.shape(theta)) ==3:
        if len(np.shape(pressure)) == 4:
            pressure=pressure.squeeze(axis=1)
            theta = temperature  * (p0 / pressure) ** (Rd/cp) #[K]
        else:
            for i in range(0,np.shape(pressure)[1]):
                theta[:,i,:,:] = temperature[:,i,:,:]  * (p0 / pressure[:,i,:,:]) ** (Rd/cp) #[K]


    #print(theta)
    return theta
def density(Tv, p):
    """
    Parameters
    ----------
    Tv: [K]: Virual temp
    p: [Pa] Pressure on FULL MODELLEVELS, on pressure levels, or surface.

    Returns
    -------
    rho: [kg/m^3]
    """
    Rd = 287.06       #[J/kg K] Gas constant for dry air
    rho = p/(Rd*Tv)   #kg/m^3
    #for k in levels_r:
    #    t_v_level[:, k, :, :] = air_temperature_ml[:, k, :, :] * (1. + 0.609133 * specific_humidity_ml[:, k, :, :])
    return rho

def specific_humidity(T, rh,p):
    #T in K
    #p in Pa
    #rh in frac no %
    Rd = 287.0
    Rv=461.0
    TC = T-273.15
    es = 6.1094 * np.exp(17.625 * TC / (TC + 243.04))*100 #pa
    e = rh * es
    w= e * Rd / ( Rv * ( p - e ) )
    q = w/(w+1)
    return q

def dexcess(mslp,SST, q2m):

    Q = q2m#.squeeze()
    mslp=mslp#.squeeze()
    SST=SST#.squeeze()

    SST = SST - 273.15
    mslp = mslp / 100

    es = 6.1094 * np.exp(17.625 * SST / (SST + 243.04)) #pa
    Qs = 0.622 * es / (mslp - 0.37 * es)
    # RH_2m = dmap_meps.relative_humidity_2m[tidx,:,:].squeeze()*100
    RH = Q / Qs * 100
    # print("RH: {}".format(RH_2m[0][0]))
    d = 48.2 - 0.54 * RH
    return d#.squeeze()

def virtual_temp(air_temperature_ml=None, specific_humidity_ml=None, dmet=None):
    """

    Parameters
    ----------
    air_temperature_ml [K]
    specific_humidity_ml [kg/kg]

    Returns
    -------
    Vituel temp [K] on full modellevels.
    """
    #todo: adjust so u can send in either multidim array, lesser dim, or just point numbers
    #https://confluence.ecmwf.int/pages/viewpage.action?pageId=68163209
    if dmet is not None:
        dim_dict = dict( zip(dmet.dim.air_temperature_ml, np.shape(dmet.air_temperature_ml)))
        levelSize = np.shape(dmet.air_temperature_ml)[1]
        air_temperature_ml=dmet.air_temperature_ml
    else:
        timeSize, levelSize, ySize, xSize = np.shape(air_temperature_ml)


    t_v_level = np.zeros(shape= np.shape(air_temperature_ml))
    levels = np.arange(0, levelSize)
    levels_r = levels[::-1]  # bottom (lvl=64) to top(lvl = 0) of atmos
    Rd = 287.06
    #for idx in np.ndindex(t_v_level.shape):
        #t_v_level[idx] = air_temperature_ml[idx] * (1. + 0.609133 * specific_humidity_ml[idx])
    for k in levels_r:
        #print(k)
        #print( air_temperature_ml[:, k, :])
        t_v_level[:, k, :] = air_temperature_ml[:, k, :] * (1. + 0.609133 * specific_humidity_ml[:, k, :])
    return t_v_level

def lapserate(T_ml, z, srf_T = None):
    """
    AINA:todo IDEA look at the diana code for comparison. They make dt/dz, but from specific arome files vc I think.
    NB! understand before use. This takes dt and dz over some define modelstep.
    Since each modellevels is further apart higher up it means its courser and courser defined.
    It is only used on low levels where dz is small, but still there is differences in dz.
    Parameters
    ----------
    air_temperature_ml [K] temp on full modelelevl
    heighttoreturn: [m]: height on full model levels from ground / sea level: does not matter since we are after dz

    Returns
    -------
    lapserate [K/km].
    """


    timeSize, levelSize, ySize, xSize = np.shape(T_ml)

    dt_levels = np.full((timeSize, levelSize, ySize, xSize), np.nan)
    dz_levels = np.full((timeSize, levelSize, ySize, xSize), np.nan)
    dtdz = np.full( (timeSize, levelSize, ySize, xSize), np.nan)
    step = 1 #5 before average over
    for k in range(0, levelSize - step):
        k_next = k + step

        dt_levels[:, k, :, :] = T_ml[:, k, :, :] - T_ml[:, k_next, :,:]  # over -under
        dz_levels[:, k, :, :] = z[:, k, :, :] - z[:, k_next, :, :]  # over -under



    dtdz[:, :, :, :] = np.divide(dt_levels, dz_levels) * 1000  # /km

    if srf_T is not None:
        ii = levelSize - step
        dt = T_ml[:, ii, :, :] - srf_T[:, 0, :, :]
        dz = z[:, ii, :, :] - 0
        val = np.divide(dt, dz) * 1000

        for k in range(ii,levelSize):
            #print("inside level")
            dtdz[:, k, :, :] =  val

    #from scipy.interpolate import griddata

    # target grid to interpolate to
    #zi = np.arange(0, 1.01, 0.01)
    #dtdz_i = griddata(zi, dtdz, (i, method='linear')

    #xi, yi = np.meshgrid(xi, yi)



    return dtdz

####################################################################################################################
# HEIGHT HANDLING
#####################################################################################################################
#model levels to pressure levels
def _ml2pl_full2full( ap, b, surface_air_pressure ):
    """
    Calculate pressure on the same modellevels 
    Parameters
    ----------
    ap: [Pa]
    b
    surface_air_pressure: [Pa]]
    timeSize: int,
            time steps
    levelSize: int,
            vertical levels
    
    Returns
    -------
    p(n,k,j,i) = ap(k) + b(k)*ps(n,j,i)
    p [Pa] pressure on each half modellevel.
    """
    timeSize = np.shape(surface_air_pressure)[0]
    levelSize = np.shape(ap)[0]
    if len(np.shape(surface_air_pressure)) !=1: #if we calculate for multiple positions
        sap = surface_air_pressure[:, 0, :, :] if len(np.shape(surface_air_pressure)) ==4 else surface_air_pressure #removes unwanted dimention
        ySize = np.shape(surface_air_pressure)[-2] #lat in y
        xSize = np.shape(surface_air_pressure)[-1] #lon in x
        if len(np.shape(ap))==1: #if one dimentional: most often the case
            levelSize = np.shape(ap)[0]
            p = np.zeros(shape = (timeSize, levelSize, ySize, xSize))
            for k in range(0,levelSize):
                p[:, k, :, :] = ap[k] + b[k] * sap[:,:,:]  #(10,1,1,1) = (10,1) + (10,1) * (10,1,1)
        else: #if ap is not one dimentional but changes with time, like if we glue together different dataset at different times 
            levelSize = np.shape(ap)[1] 
            p = np.zeros(shape = (timeSize, levelSize, ySize, xSize))
            for k in range(0,levelSize):
                for t in range(0,timeSize):
                    p[t, k, :, :] = ap[t, k] + b[t, k] * sap[t,:,:]  #(10,1,1,1) = (10,1) + (10,1) * (10,1,1)
    else: #if we calculate for just one positions
        p = np.zeros(shape = (timeSize, levelSize))
        for k in range(0,levelSize):
            p[:,k] = ap[:,k] + b[:,k] * surface_air_pressure
    return p
    

    """
    NB NOT USED
    Parameters
    ----------
    ap: [Pa]
    b
    surface_air_pressure: [Pa]]

    Returns
    -------
    p(n,k,j,i) = ap(k) + b(k)*ps(n,j,i)
    p [Pa] pressure on each half modellevel.
     Ah(k-1) = 2*Af(k) - Ah(k)
     Bunnen i halvnivåer er Ps, da må Ah(65) = 0 og Bh(65) = 1
    """
    levels = np.arange(0, 64)  # index of heighlevels from top(lvl = 0) to bottom(lvl=64)
    levels_r = levels[::-1]
    timeSize = np.shape(surface_air_pressure)[0]
    levelSize = np.shape(ap)[0]
    ySize = np.shape(surface_air_pressure)[2]  # lat in y
    xSize = np.shape(surface_air_pressure)[3]  # lon in x

    ah = np.zeros(shape=np.shape(ap))
    bh = np.zeros(shape=np.shape(b))
    ph = np.zeros(shape=(timeSize, levelSize, ySize, xSize))

    ah[64] = 0
    bh[64] = 1
    ph[:,64,:,:] = surface_air_pressure[:, 0, :, :]
    for k in levels_r:
        #print(k)
        ah[k-1]= 2*ap[k]/101320 - ah[k]
        bh[k-1]= 2*b[k] - bh[k]
        #print(ah[k-1])
        #print(bh[k - 1])
        #print(b[k])
        #pfull[:, k, :, :] = 0.5*( phalf[:, k-1, :, :] + phalf[:, k, :, :] )

        ph[:,k-1,:,:]= ah[k-1]*101320 + bh[k-1]*surface_air_pressure[:, 0, :, :]

    return ph





    print("Not implemented yet")
def _ml2pl_full2half( ap, b, surface_air_pressure ):
    """
    Calculates pressure on half levels
    """
    print("in ml2pl_full2half")
    ap_0=0
    b_0=0
    ap_65 = 0
    b_65 = 1
    levelsize = np.shape(ap)[0]
    #print( levelsize)
    ah = np.zeros(levelsize+1)
    bh = np.zeros(levelsize+1)
    ah[0] = ap_0
    bh[0] = b_0
    ah[65] = ap_65
    bh[65] = b_65
    for k in range(1,levelsize):
        #ah[k-1] = 2*ap[k] - ah(k)   #bh[k-1] = 2*b[k] - bh(k)
        ah[k] = 2*ap[k] - ah[k-1]
        bh[k] = 2*b[k] - bh[k-1]
    ph = _ml2pl_full2full( ah, bh, surface_air_pressure)
    return ph
def ml2pl( ap=None, b=None, surface_air_pressure=None, inputlevel="full", returnlevel="full", dmet=None, dim=None):
    """
    Check if pressure is on half or full levels, and calls the appropriate function for this.

    Parameters
    ----------
    ap: [Pa]
    b
    surface_air_pressure: [Pa]]
    inputlevel = "full" or "half": if input data is on full or half levels
    returnlevel="full" or "half" : if return data should be on full or half levels.

    Returns
    -------
    Pressure levels on full levels if returnlevel="full"
    or
    Pressure levels on half levels if returnlevel="half"

    INFO:
    -------
    Model: Metcoop produces netcdf files that has given out ap,b so that p is on full levels imediately: See email.
    Source: https://github.com/metno/NWPdocs/wiki/Calculating-model-level-height/_compare/041362b7f5fdc02f5e1ee3dea00ffc9d8d47c2bc...f0b453779e547d96f44bf17803d845061627f7a8
    """
    if inputlevel=="full" and returnlevel=="full":
        p = _ml2pl_full2full( ap, b, surface_air_pressure)
    return p

#model/pressure levels to geopotential height
def cheat_alt(m_level):
    """
    Predifened heights that I belive is on half levels.
    """
    H = [24122.6894480669, 20139.2203688489, 17982.7817599549, 16441.7123200128,
     15221.9607620438, 14201.9513633491, 13318.7065659522, 12535.0423836784,
     11827.0150898454, 11178.2217936245, 10575.9136768674, 10010.4629764989,
     9476.39726730647, 8970.49319005479, 8490.10422494626, 8033.03285976169,
     7597.43079283063, 7181.72764002209, 6784.57860867911, 6404.82538606181,
     6041.46303718354, 5693.61312218488, 5360.50697368367, 5041.46826162131,
     4735.90067455394, 4443.27792224573, 4163.13322354697, 3895.05391218293,
     3638.67526925036, 3393.67546498291, 3159.77069480894, 2936.71247430545,
     2724.28467132991, 2522.30099074027, 2330.60301601882, 2149.05819142430,#30=2149
     1977.55945557602, 1816.02297530686, 1664.38790901915, 1522.61641562609,
     1390.69217292080, 1268.36594816526, 1154.95528687548, 1049.75817760629,
     952.260196563843, 861.980320753114, 778.466725603312, 701.292884739207,
     630.053985133223, 564.363722589458, 503.851644277509, 448.161118360263,
     396.946085973573, 349.869544871297, 306.601457634038, 266.817025119099,
     230.194566908004, 196.413229972062, 165.151934080260, 136.086183243070,
     108.885366240509, 83.2097562375566, 58.7032686584901, 34.9801888163106,
     11.6284723290378]

    alt = np.array([H[i] for i in m_level])
    return alt
def _pl2alt_half2full_gl( air_temperature_ml, specific_humidity_ml, p): #or heighttoreturn
    """
    Parameters
    ----------
    p: [Pa] pressure on each half modellevel: p = ap + b * surface_air_pressure
    surface_geopotential:[m^2/s^2] Surface geopotential (fis)
    air_temperature_ml: [K] temperature on every model level.
    specific_humidity_ml: [kg/kg] specific humidity on everymodellevel

    Returns
    -------
    geotoreturn_m: [m] geopotential height from groundlevel on each full modellevel

    Calculations:
     ------------------
     The Euler equations are formulated in a terrain-following pressure-based sigma-coordinate system (Simmons and Burridge 1981)
     U,V,T are on full levels, while p on half levels.

     Using hydpsometric equation, but in pressure-based sigma-coordinate system system following (Simmons and Burridge 1981):
     ###################################################################################################################
     * Psi(full_level) = Psi(top half_level) - alpha(full levels)*Rd*Tv
        *Psi(top half_level) = Psi(lower half level) -RdTvln( p(top half_level)/p(lower half level))
        *alpha(full levels) = 1-  p(top half_level)/ (p(lower half level)- p(top half_level)) * ln( p(top half_level)/p(lower half level))
    ###################################################################################################################
    Sources:
    ------------------
    #Arome config
    https://www.researchgate.net/publication/313544695_The_HARMONIE-AROME_model_configuration_in_the_ALADIN-HIRLAM_NWP_system
    #Equation Source#:
    Simmons, A. J. and Burridge, D. M. (1981).
     An energy and angular momentum conserving vertical finite difference scheme and hybrid vertical coordinates.
      Mon. Wea. Rev., 109, 758–766.
    #Implementation source#:
    https://confluence.ecmwf.int/pages/viewpage.action?pageId=68163209
    """
    #https://confluence.ecmwf.int/pages/viewpage.action?pageId=68163209
    #https://www.ecmwf.int/sites/default/files/elibrary/2015/9210-part-iii-dynamics-and-numerical-procedures.pdf
    #http://www.dca.ufcg.edu.br/mna/Anexo-MNA-modulo02b.pdf
    #https://www.ecmwf.int/sites/default/files/elibrary/1981/12284-energy-and-angular-momentum-conserving-finite-difference-scheme-hybrid-coordinates-and-medium.pdf

    Rd = 287.06 #[J/kg K] Gas constant for dry air
    g = 9.80665
    z_h = 0  # 0 since geopotential is 0 at sea level
    #if p == None:
    #    p = ml2plhalf( ap, b, surface_air_pressure )


    timeSize, levelSize, ySize, xSize = np.shape(p)
    geotoreturn_m = np.zeros(shape=(timeSize, levelSize, ySize, xSize))
    t_v_level = np.zeros(shape=(timeSize, levelSize, ySize, xSize))

    levels = np.arange(0, levelSize-1)  #index of heighlevels from top(lvl = 0) to bottom(lvl=64)
    levels_r = levels[::-1]           #index of heighlevels from bottom(lvl=64) to top(lvl = 0)
    p_low = p[:, levelSize - 1, :, :] # Pa lowest modelcell is 64

    for k in levels_r:                #loops through all levels from bottom to top #64,63,63.....3,2,1,0 #
        p_top = p[:, k - 1, :, :]     #Pressure at the top of that layer
        t_v_level[:, k, :, :] = air_temperature_ml[:, k, :, :] * (1. + 0.609133 * specific_humidity_ml[:, k, :, :])

        if k == 0:  # top of atmos, last loop round
            dlogP = np.log(p_low / 0.1)
            alpha = np.log(2)
        else:
            dlogP = np.log(np.divide(p_low, p_top))
            dP = p_low - p_top  #positive
            alpha = 1. - ((p_top / dP) * dlogP)

        TRd = t_v_level[:, k, :, :] * Rd
        
        z_f = z_h + (TRd * alpha)

        geotoreturn_m[:, k, :, :] = z_f #+ surface_geopotential[:, 0, :, :]

        geotoreturn_m[:, k, :, :] = geotoreturn_m[:, k, :, :]/g

        z_h = z_h + (TRd * dlogP)
        p_low = p_top
    print("")
    return geotoreturn_m

def _pl2alt_full2full_gl(dmet=None, dim=None, ap=None, b=None, surface_air_pressure=None, air_temperature_ml=None, specific_humidity_ml=None,pressure=None): #or heighttoreturn
    """
    when pressure comes in full levels, this works best anyway. 50 meter difference at highest levels. 
    """

    if dmet is not None:
        dim_dict = dict( zip(dmet.dim.air_temperature_ml, np.shape(dmet.air_temperature_ml)))
        levelSize = np.shape(dmet.air_temperature_ml)[1]
    if len(np.shape(air_temperature_ml)) ==4:
        timeSize, levelSize, ySize, xSize = np.shape(air_temperature_ml)
    else:
        timeSize, levelSize, point= np.shape(air_temperature_ml)

    Rd = 287.06 #[J/kg K] Gas constant for dry air
    g0 = 9.80665
    z_h = 0  # 0 since geopotential is 0 at sea level
    Tv = virtual_temp(air_temperature_ml, specific_humidity_ml, dmet)
    p = ml2pl( ap, b, surface_air_pressure, inputlevel="full", returnlevel="full") if pressure is None else pressure
    h = np.zeros(shape = np.shape(air_temperature_ml)) #p = np.zeros(shape = (timeSize, levelSize, ySize, xSize))
    #print(np.shape(surface_air_pressure))
    #print(np.shape(p))
    #print(np.shape(Tv))
    #exit(1)
    #h[:,-1,:,:] = Rd*Tv[:,-1,:,:]/g0 * np.log(surface_air_pressure[:,:,:,:]/p[:,-1,:,:]) + z_h
    h[:,-1,:] = Rd*Tv[:,-1,:]/g0 * np.log(surface_air_pressure[:,0,:]/p[:,-1,:]) + z_h #(3, 3, 1, 1)
    
    for i in range(0,levelSize-1):
        i =levelSize-2-i
        #print(i)
        Tv_mean = (Tv[:,i+1,...]+ Tv[:,i,...]) /2#np.average([Tv[:,i+1,...],Tv[:,i,...]], axis=0)
        g = g0*(1-2*h[:,i+1,...]/6378137) #just for fun. no effect
        h[:,i,...] = Rd*Tv_mean/g * np.log(p[:,i+1,...]/p[:,i,...]) + h[:,i+1,...]
    gph = h
    #exit(1)
    return gph
def ml2alt( air_temperature_ml, specific_humidity_ml, ap, b, surface_air_pressure, surface_geopotential=None, inputlevel="full", returnlevel="full",pressure=None ):     #https://confluence.ecmwf.int/pages/viewpage.action?pageId=68163209
    if inputlevel == "full" and returnlevel == "full": #default
        p     = _ml2pl_full2full( ap=ap, b=b,surface_air_pressure= surface_air_pressure ) if pressure is None else pressure
        gph_m = _pl2alt_full2full_gl( ap=ap, b=b,surface_air_pressure= surface_air_pressure, air_temperature_ml=air_temperature_ml, specific_humidity_ml=specific_humidity_ml, pressure=p ) #
    if inputlevel == "half" and returnlevel == "full": #If staggered
        p     = _ml2pl_full2half( ap, b, surface_air_pressure ) if pressure is None else pressure
        gph_m = _pl2alt_half2full_gl( air_temperature_ml, specific_humidity_ml, p ) #
    gph_m = gph_m + surface_geopotential/9.81 if surface_geopotential is not None else gph_m
    return gph_m

def pl2alt(ap=None, b=None, surface_air_pressure=None, air_temperature_ml=None, specific_humidity_ml=None,pressure=None,surface_geopotential=None, dmet=None):
    alt = _pl2alt_full2full_gl(ap=ap, b=b, surface_air_pressure=surface_air_pressure, air_temperature_ml=air_temperature_ml, specific_humidity_ml=specific_humidity_ml,pressure=None, dmet=dmet)
    alt = alt + surface_geopotential/9.81 if surface_geopotential is not None else alt
    return alt

#ground level to sealevel and vicaverca
def gl2sl(surface_geopotential, gph_m_gl):
    g = 9.80665
    gph_m_sl = gph_m_gl + surface_geopotential/g
    return gph_m
def sl2gl(surface_geopotential, gph_m_sl):
    g = 9.80665
    gph_m_gl = gph_m_sl - surface_geopotential / g
    return gph_m_gl

#altitude to pressure level
def alt_gl2pl(surface_air_pressure,tv, alt_gl, outshape=None ):
    Rd = 287.06
    g = 9.80665
    #    hsl = hgl + (surface_geopotential / g)    #

    #if type(alt_gl)==float or type(alt_gl) == int or type(alt_gl)==str:#if height is constant with time
    #    alt_gl = np.full(np.shape(data_altitude_sl[:, :, jindx, iindx]), float(alt_gl))
    #else: #if height is changes with time it comes as a array or list
    #    alt_gl = np.repeat(point_alt, repeats=len(data_altitude_sl[0, :, jindx, iindx]), axis=0).reshape(np.shape(data_altitude_sl[:, :, jindx, iindx]))
    #virtual_temp(air_temperature_ml, specific_humidity_ml)
    tvdlogP = np.multiply(tv, dlogP)
    T_vmean = np.divide( np.nansum(tvdlogP, axis=1), np.nansum(dlogP, axis=1) )
    H = Rd * T_vmean / g  # scale height
    pl = surface_air_pressure[:, -1, jindx, iindx] * np.exp(-(np.array(alt_gl) / H))
    return pl
def alt_sl2pl(surface_air_pressure, alt_sl ):
    Rd = 287.06
    g = 9.80665

    tvdlogP = np.multiply(tv, dlogP)
    T_vmean = np.divide( np.nansum(tvdlogP, axis=1), np.nansum(dlogP, axis=1) )
    point_alt_gl = alt_sl[:, 0] - surface_geopotential[:, 0, jindx, iindx] / g  # convert to height over surface.
    H = Rd * T_vmean / g  # scale height
    pl = surface_air_pressure[:, -1, jindx, iindx] * np.exp(-(np.array(point_alt_gl) / H))
    return pl
def point_alt_sl2pres_old(jindx, iindx, point_alt, data_altitude_sl, t_v_level, p, surface_air_pressure, surface_geopotential):
    """
    Converts height from sealevel to pressure.
    Parameters
    ----------
    jindx
    iindx
    point_alt
    data_altitude_sl
    t_v_level
    p
    surface_air_pressure
    surface_geopotential

    Returns
    -------
    """
    Rd = 287.06
    g = 9.80665

    if type(point_alt)==float or type(point_alt) == int or type(point_alt)==str:#if height is constant with time
        point_alt = np.full(np.shape(data_altitude_sl[:, :, jindx, iindx]), float(point_alt))
    else: #if height is changes with time it comes as a array or list
        point_alt = np.repeat(point_alt, repeats=len(data_altitude_sl[0, :, jindx, iindx]), axis=0).reshape(np.shape(data_altitude_sl[:, :, jindx, iindx]))

    timeSize, levelSize = np.shape(p[:, :, jindx, iindx])

    #max index when we have an altitude in our dataset lett or equal to point_altitude
    idx_tk = np.argmax( (data_altitude_sl[:, :, jindx, iindx] <= point_alt[:]), axis=1)
    tv = t_v_level[:, :, jindx, iindx]
    #######################################
    dp = np.zeros(shape=(timeSize, levelSize))
    dlogP = np.zeros(shape=(timeSize, levelSize))
    levels_r = np.arange(0, levelSize)[::-1] # bottom (lvl=64) to top(lvl = 0) of atmos
    p_low = p[:, levelSize - 1, jindx, iindx]  # Pa lowest modellecel is 64
    for k in levels_r:
        p_top = p[:, k - 1, jindx, iindx]

        if k == 0:  # top of atmos, last loop round
            dlogP_p = np.log(p_low / 0.1)  # 0.1 why? -->todo
            alpha = np.log(2)  # 0.3 why?       -->todo
        else:
            dlogP_p = np.log(np.divide(p_low, p_top))
            dP_p = p_low - p_top  # positive
            alpha = 1. - ((p_top / dP_p) * dlogP_p)
        dlogP[:, k] = dlogP_p
        dp[:, k] = dP_p
    ########################################
    for t in range(0, np.shape(tv)[0]):  # 0,1,2
        tv[t, 0:idx_tk[t]] = np.nan
        dp[t, 0:idx_tk[t]] = np.nan
        dlogP[t, 0:idx_tk[t]] = np.nan
    tvdlogP = np.multiply(tv, dlogP)

    T_vmean = np.divide( np.nansum(tvdlogP, axis=1), np.nansum(dlogP, axis=1) )
    H = Rd * T_vmean / g  # scale height
    point_alt_gl = point_alt[:, 0] - surface_geopotential[:, 0, jindx, iindx] / g  # convert to height over surface.
    p_point = surface_air_pressure[:, -1, jindx, iindx] * np.exp(-(np.array(point_alt_gl) / H))

    return p_point
#Special height calc
def BL_height_sl(atmosphere_boundary_layer_thickness, surface_geopotential):
    """
    Parameters
    ----------
    atmosphere_boundary_layer_thickness: Height of the PBL [m] over sealevel
    surface_geopotential: Surface geopotential (fis):  [m^2/s^2]
    Returns
    -------
    BL height over surface
    """
    g=9.80665 #m/s^2
    hgl = atmosphere_boundary_layer_thickness #groundlevel
    hsl = hgl + (surface_geopotential / g)    #
    return hsl





####################################################################################################################
# WIND HANDLING
#####################################################################################################################
# The wind calculations are validated with the use of parameters:
# x_wind_10m, y_wind_10m, wind_speed(height3=10m), wind_direction(height3=10m)
#   u10, v10  = xwind2uwind(tmap_meps.x_wind_10m, tmap_meps.y_wind_10m, tmap_meps.alpha)
#   wsfromuv = wind_speed(u10,v10)
#   wsfromxy = wind_speed(tmap_meps.x_wind_10m, tmap_meps.y_wind_10m)
#   wdfromuv = (np.pi/2 - np.arctan2(v10,u10) + np.pi)*180/np.pi %360
#   wdfromxy =  wind_dir(tmap_meps.x_wind_10m,tmap_meps.y_wind_10m,tmap_meps.alpha)
# Result is true with approx deviaton error of 0.002 or less.
# wsfromuv[0,0,0,0] == wsfromxy[0,0,0,0] == wind_speed[0,0,0,0]
# wdfromuv[0,0,0,0] == wdfromxy[0,0,0,0] == wind_direction[0,0,0,0]
#todo: AINA: alpha: for changing domain, for changing height?
#####################################################################################################################
def windfromspeed_dir(wind_speed,wind_direction ):
    u = -wind_speed * np.sin(np.deg2rad(wind_direction))  # m/s u wind
    v = -wind_speed * np.cos(np.deg2rad(wind_direction))  # m/s v wind

    return u,v
def xwind2uwind( xwind, ywind, alpha=None ):
    # u,v = xwind2uwind( data.xwind, data.ywind, data.alpha )
    #source: https://www-k12.atmos.washington.edu/~ovens/wrfwinds.html
    #source: https://github.com/metno/NWPdocs/wiki/From-x-y-wind-to-wind-direction

    if alpha==None:
        os.system('python wind2alpha.py')


    u = np.zeros(shape=np.shape(xwind))
    v = np.zeros(shape=np.shape(ywind))
    for t in range(0,np.shape(xwind)[0]):
        for k in range(0, np.shape(xwind)[1]):
            #absdeg2rad = np.abs((alpha)*np.pi/180)
            #alpha = alpha#360
            #absdeg2rad = alpha*np.pi/180
            absdeg2rad = np.abs((alpha%360)*np.pi/180)


            u[t, k, :, :] = xwind[t, k, :, :] * np.cos(absdeg2rad[:,:]) - ywind[t, k, :, :] * np.sin(absdeg2rad[:,:])
            v[t, k, :, :] = ywind[t, k, :, :] * np.cos(absdeg2rad[:,:]) + xwind[t, k, :, :] * np.sin(absdeg2rad[:,:])

    return u,v
def wind_speed(xwind,ywind):
    #no matter if in modelgrid or earthrelativegrid
    ws = np.sqrt(xwind**2 + ywind**2)
    return ws
def relative_humidity(temp,q,p):
    temp = temp-273.15 #Kelvin to Celcius
    w=q/(1-q)
    es =  6.112 * np.exp((17.67 * temp)/(temp + 243.5))*100
    ws = 0.622*es/p
    rh = w/ws
    rh[rh > 1] =  1
    rh[rh < 0] =  0
    return rh*100
def wind_dir(xwind,ywind, alpha=None):
    #source: https://www-k12.atmos.washington.edu/~ovens/wrfwinds.html
    #https://github.com/metno/NWPdocs/wiki/From-x-y-wind-to-wind-direction
    #https://stackoverflow.com/questions/21484558/how-to-calculate-wind-direction-from-u-and-v-wind-components-in-r
    #u = np.zeros(shape=np.shape(xwind))
    #v = np.zeros(shape=np.shape(ywind))
    wdir = np.empty( shape=np.shape(xwind) )

    if len(np.shape(xwind)) <= 1: #if calculate for just one dimless value

        a = np.arctan2(ywind, xwind) #radianse
        b = a + np.pi
        c = np.pi/2. - b
        wdir = np.degrees(c)
        wdir = wdir % 360
        eps = 0.5 * 10**(-10)
        wdir[(abs(xwind) < eps) & (abs(ywind) < eps) ] = np.nan

    if len(np.shape(xwind)) > 1:  #if wind has multiple dimentions
        for t in range(0,np.shape(wdir)[0]):
            for k in range(0, np.shape(wdir)[1]):
                #websais:#wdir[t,k,:,:] =  alpha[:,:] + 90 - np.arctan2(ywind[t,k,:,:],xwind[t,k,:,:])
                #Me:
                a =  np.arctan2( ywind[t,k,:,:], xwind[t,k,:,:] )  #mathematical wind angle in modelgrid pointing with the wind
                #a = a * (a >= 0) + (a + 2 * np.pi) * (a < 0)
                #a =  np.mod(a,np.pi)
                b = a*180./np.pi + 180.  # mathematical wind angle pointing where the wind comes FROM
                c = 90. - b   # math coordinates(North is 90) to cardinal coordinates(North is 0).
                if alpha !=None: #or alpha !=0:
                    wdir[t,k,:,:] =  c[:,:] - alpha[:,:] #add rotation of modelgrid(alpha).
                    #wdir[t,k,:,:] = np.subtract(c%360, alpha%360)
                wdir = wdir % 360  # making sure is between 0 and 360 with Modulo
    return wdir


def add_radiuskm2latlon(lat1_deg,lon1_deg, distancekm=15 , R = 6371.0 ):
    #R = 6378.1 #Radius of the Earth #model has 6371000.0
    d = distancekm #Distance in km
    brng = math.radians(45) #upper right corner

    lat1 = math.radians(lat1_deg) #Current lat point converted to radians
    lon1 = math.radians(lon1_deg) #Current long point converted to radians

    lat2 = math.asin( math.sin(lat1)*math.cos(d/R) +
        math.cos(lat1)*math.sin(d/R)*math.cos(brng))

    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(d/R)*math.cos(lat1),
                math.cos(d/R)-math.sin(lat1)*math.sin(lat2))

    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)

    lonlat2 = [lon2,lat2]

    brng = math.radians(225) #lower left  corner

    lat3 = math.asin( math.sin(lat1)*math.cos(d/R) +
        math.cos(lat1)*math.sin(d/R)*math.cos(brng))

    lon3 = lon1 + math.atan2(math.sin(brng)*math.sin(d/R)*math.cos(lat1),
                math.cos(d/R)-math.sin(lat1)*math.sin(lat3))

    lat3 = math.degrees(lat3)
    lon3 = math.degrees(lon3)
    lonlat3 = [lon3,lat3]

    return lonlat2, lonlat3
