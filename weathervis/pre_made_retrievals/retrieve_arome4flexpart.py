from weathervis.domain import * 
from weathervis.check_data import *
from weathervis.utils import *
from weathervis.get_data import *
from weathervis.calculation import *
from netCDF4 import Dataset
import os
import datetime
import platform
from weathervis.checkget_data_handler import *


def retrieve_arome4flexpart(outputpath, modelruntime,steps,lvl,xres,yres,model, use_latest):
    """
Retrieves arome model data to be used as inputs for flexpart-arome
--------------------------------------------------------------------
The function is an adapted version of
"FA_to_nc_2S_3D.py" made by Jerome BRIOUDE (jerome.brioude@univ-reunion.fr) in 2019
This script however used retrievals from thredds netcdf files, instead of from direct model output FA files

flexpart uses "name" parameter of the variable, so forexample surface_air_pressure is the parameter used for retrieving
from arome, but in flexpart it is called "SP". So "SP" is important to keep like it is.
    """
    #due to different attributes for 2d, 3d values they are split up
    variable2d_arome = {}  # dictionary containing 2dimentional variables
    variable3d_arome = {}  # dictionary containing 3dimentional variables
    variable2d_sfx = {}    # dictionary containing specific surfex variables
    resol = 7              # digital precision of number in decimal
    #INITIATING PARAMETER (e.g air_temperature_2m) INFO.
    # Check that unit and description is correct with your data
    #OK
    #SURFPRESSION in arome france
    variable2d_arome['surface_air_pressure'] = {}                                           # _FillValue: 9.96921E36
    variable2d_arome['surface_air_pressure']['name'] = 'SP'                                 # long_name: Surface air pressure, standard_name: surface_air_pressure
    variable2d_arome['surface_air_pressure']['units'] = 'Pa'                                # units: Pa
    variable2d_arome['surface_air_pressure']['description'] ='surface pressure'  # orginal'log of surface pressure'
    variable2d_arome['surface_air_pressure']['precision'] = resol
    #OK
    #CLSTEMPERATURE in arome france
    variable2d_arome['air_temperature_2m'] = {}                                             # _FillValue: 9.96921E36
    variable2d_arome['air_temperature_2m']['name'] = 'T2m'                                  # long_name: Screen level temperature (T2M), standard_name: air_temperature
    variable2d_arome['air_temperature_2m']['units'] = 'K'                                   # units: K
    variable2d_arome['air_temperature_2m']['description'] = 'Temperature at 2m'
    variable2d_arome['air_temperature_2m']['precision'] = resol
    #OK, NB not sure if uses
    #SPECSURFGEOPOTEN ina arome france 
    variable2d_arome['surface_geopotential'] = {}                                           #_FillValue: 9.96921E36
    variable2d_arome['surface_geopotential']['name'] = 'Zg'                                 #long_name: Surface geopotential (fis), standard_name: surface_geopotential
    variable2d_arome['surface_geopotential']['units'] = 'm^2/s^2'                           #units: m^2/s^2
    variable2d_arome['surface_geopotential']['description'] = 'surface geopotential'
    variable2d_arome['surface_geopotential']['precision'] = resol
    #OK, NB not sure if used
    #SURFIND.TERREMER in arome france
    variable2d_arome['land_area_fraction'] = {}                                             #_FillValue: -32767, scale_factor: 0.001
    variable2d_arome['land_area_fraction']['name'] = 'LS'                                   #long_name: Land-Sea Mask (LSM), standard_name: land_area_fraction
    variable2d_arome['land_area_fraction']['units'] = 'none'                                #units: 1
    variable2d_arome['land_area_fraction']['description'] = 'land sea mask'
    variable2d_arome['land_area_fraction']['precision'] = resol
    #OK
    # CLSVENT.ZONAL in arome france
    variable2d_arome['x_wind_10m'] = {}                                                     #_FillValue: 9.96921E36
    variable2d_arome['x_wind_10m']['name'] = 'U_lon_10m'                                    #long_name: Zonal 10 metre wind (U10M), standard_name: x_wind
    variable2d_arome['x_wind_10m']['units'] = 'm/s'                                         #units: m/s
    variable2d_arome['x_wind_10m']['description'] = 'zonal wind at 10m'
    variable2d_arome['x_wind_10m']['precision'] = resol
    #OK
    #CLSVENT.MERIDIEN ina rome france
    variable2d_arome['y_wind_10m'] = {}                                                     #_FillValue: 9.96921E36
    variable2d_arome['y_wind_10m']['name'] = 'V_lat_10m'                                    # long_name: Meridional 10 metre wind (V10M), standard_name: y_wind
    variable2d_arome['y_wind_10m']['units'] = 'm/s'                                         # units: m/s
    variable2d_arome['y_wind_10m']['description'] = 'meriodional wind at 10m'
    variable2d_arome['y_wind_10m']['precision'] = resol
    #OK
    #CLSHUMI.SPECIFIQ in arome france
    variable2d_arome['specific_humidity_2m'] = {}                                           # _FillValue: 9.96921E36
    variable2d_arome['specific_humidity_2m']['name'] = 'Q2m'                                # long_name: Screen level specific humidity, standard_name: specific_humidity
    variable2d_arome['specific_humidity_2m']['units'] = 'kg/kg'                             # units: kg/kg
    variable2d_arome['specific_humidity_2m']['description'] = 'specific humidity at 2m'
    variable2d_arome['specific_humidity_2m']['precision'] = resol
    #not read by flexpart, but added in order to put into attributes
    #variable2d_arome['projection_lambert'] = {}                                           # _FillValue: 9.96921E36
    #variable2d_arome['projection_lambert']['name'] = 'projlamb'                           # long_name: Screen level specific humidity, standard_name: specific_humidity
    #variable2d_arome['projection_lambert']['units'] = 'none'                             # units: kg/kg
    #variable2d_arome['projection_lambert']['description'] = 'projection_lambert'
    #variable2d_arome['projection_lambert']['precision'] = resol
    #OK NB
    #SURFFLU.CHA.SENS ina rome france
    variable2d_arome['integral_of_surface_downward_sensible_heat_flux_wrt_time'] = {}
    variable2d_arome['integral_of_surface_downward_sensible_heat_flux_wrt_time']['name'] = 'SSHF_CUM'
    variable2d_arome['integral_of_surface_downward_sensible_heat_flux_wrt_time']['units'] = 'J.m-2'
    variable2d_arome['integral_of_surface_downward_sensible_heat_flux_wrt_time']['description'] = 'Cum.Sensible heat flux positive downward' #original 'Cum.Sensible heat flux'
    variable2d_arome['integral_of_surface_downward_sensible_heat_flux_wrt_time']['precision'] = resol

    #NB CHANGED FROM ORIG#################### AND FROM the External Surface model SURFEX
    #OK NB Hour averaged and not accumulated like arome france therefor unit changes also. You can double check this by comparing it to accumulated land variable FMUC_ISBA
    #   NB2 This comes from the SURFEX model
    #   NB3 in 2020 it was named veraged_Zonal_Wind_Stress, from feb 2022 it was named 'Averaged screen level zonal wind stress (u), screen level usually is the height at witch it can be #            compared to measurents
    #SURFTENS.TURB.ZO in arome france
    variable2d_sfx['FMU'] = {}                                                              # scale_factor: 0.001
    variable2d_sfx['FMU']['name'] = 'USTRESS'                                               # long_name: Averaged_Zonal_Wind_Stress
    variable2d_sfx['FMU']['units'] = 'Kg.m-1.s-2' #NB original 'Kg.m-1.s-1'                 #  kg/ms2
    variable2d_sfx['FMU']['description'] = 'Averaged screen level zonal wind stress (u)' #NB original 'Surface wind stress (u)'
    variable2d_sfx['FMU']['precision'] = resol
    #OK NB see same as FMU
    #SURFTENS.TURB.ME in arome france
    variable2d_sfx['FMV'] = {}                                                              # scale_factor: 0.001
    variable2d_sfx['FMV']['name'] = 'VSTRESS'                                               # long_name: Averaged_Merid_Wind_Stress
    variable2d_sfx['FMV']['units'] = 'Kg.m-1.s-2' #NB original 'Kg.m-1.s-1'                 # units: kg/ms2
    variable2d_sfx['FMV']['description'] = 'Averaged screen level meridional wind stress (v)' #NB original 'Surface wind stress (v)'
    variable2d_sfx['FMV']['precision'] = resol
    #OK NB Hour averaged and not accumulated like arome france therefor unit changes also. You can double check this by comparing it to accumulated                
    #       integral_of_surface_downward_sensible_heat_flux_wrt_time
    #   NB2 from surfex model- AA uses integral_of_surface_downward_sensible_heat_flux_wrt_time. Compared to they are the same
    #   NB3 The benefit from this is that we get the flux at leadtime 0 which is not the case from AA. It is often smaller and for a quick look could be 20 W/m2 smaller
    # SURFFLU.CHA.SENS in arome france
    variable2d_sfx['H'] = {}                                                                # scale_factor: 0.1
    variable2d_sfx['H']['name'] = 'SSHF'                                                    # standard_name: surface_upward_sensible_heat_flux, long_name: Averaged_Sensible_Heat_Flux
    variable2d_sfx['H']['units'] = 'W.m-2'  #NB original  'J.m-2                            # units: W/m2
    variable2d_sfx['H']['description'] = 'Averaged_Sensible_Heat_Flux'
    variable2d_sfx['H']['precision'] = resol
    ########################################

    #OK
    #TEMPERATURE in arome france
    variable3d_arome['air_temperature_ml'] = {}                                                       # _FillValue: -2147483647, scale_factor: 0.001, 
    variable3d_arome['air_temperature_ml']['name'] = 'T'                                              # long_name: Air temperature model levels, standard_name: air_temperature
    variable3d_arome['air_temperature_ml']['units'] = 'K'                                             # units: K
    variable3d_arome['air_temperature_ml']['description'] = 'temperature on pressure sigmal levels'   # grid_mapping: projection_lambert
    variable3d_arome['air_temperature_ml']['precision'] = resol                                       # digit precision
    
    
    #OK NB: divergence vetrical was used in FLEXPART-AROME original, but since we have W we use that instead...but should check effect as they are different
    #VERTIC.DIVER in arome france
    variable3d_arome['divergence_vertical'] = {}                                                      # _FillValue: 9.96921E36
    variable3d_arome['divergence_vertical']['name'] = 'NH_dW'                                         # long_name: Vertical divergence, metno_name: divergence_vertical
    variable3d_arome['divergence_vertical']['units'] = 's-1'                                          # units: s-1 ---> = 'm/s * g'
    variable3d_arome['divergence_vertical']['description'] = 'Non Hydrostatic divergence of vertical velocity: D = -g(w(i) - w(i-1))'
    variable3d_arome['divergence_vertical']['precision'] = resol
    #OK NB Not used in original, but used in AA. 
    variable3d_arome['upward_air_velocity_ml'] = {}
    variable3d_arome['upward_air_velocity_ml']['name'] = 'W'
    variable3d_arome['upward_air_velocity_ml']['units'] = 'm/s'
    variable3d_arome['upward_air_velocity_ml']['description'] = 'Vertical vind model levels'
    variable3d_arome['upward_air_velocity_ml']['precision'] = resol
    
    #OK
    #'WIND.U.PHYS' in arome france
    variable3d_arome['x_wind_ml'] = {}                                                              # _FillValue: -2147483647, scale_factor: 0.001
    variable3d_arome['x_wind_ml']['name'] = 'U_X'                                                   # long_name: Zonal wind model levels, standard_name: x_wind
    variable3d_arome['x_wind_ml']['units'] = 'm/s'                                                  # units: m/s
    variable3d_arome['x_wind_ml']['description'] = 'U wind along x axis on pressure sigmal levels'  # Zonal wind model levels, standard_name: x_wind
    variable3d_arome['x_wind_ml']['precision'] = resol
    #OK
    #'WIND.V.PHYS' in arome france
    variable3d_arome['y_wind_ml'] = {}                                                              # _FillValue: -2147483647
    variable3d_arome['y_wind_ml']['name'] = 'V_Y'                                                   # long_name: Meridional wind model levels, standard_name: y_wind
    variable3d_arome['y_wind_ml']['units'] = 'm/s'                                                  # units: m/s
    variable3d_arome['y_wind_ml']['description'] = 'V wind along y axis on pressure sigmal levels'  # Meridional wind model levels
    variable3d_arome['y_wind_ml']['precision'] = resol

    ############JB says not needed --> see readwind.f90 in flexpart
    ## variable3d_arome['PRESS.DEPART']={}
    ## variable3d_arome['PRESS.DEPART']['name'] = 'NH_dP'
    ## variable3d_arome['PRESS.DEPART']['units'] = 'Pa'
    ## variable3d_arome['PRESS.DEPART']['description'] = 'NH departure from pressure'
    ## variable3d_arome['PRESS.DEPART']['precision'] = 1
    ##############

    #OK
    # HUMI.SPECIFI in arome france
    variable3d_arome['specific_humidity_ml'] = {}                                                   # _FillValue: -2147483647, scale_factor: 1.0E-7
    variable3d_arome['specific_humidity_ml']['name'] = 'Q'                                          # long_name: Specific humidity model levels, standard_name: specific_humidity
    variable3d_arome['specific_humidity_ml']['units'] = 'kg/kg'                                     # units: kg/kg
    variable3d_arome['specific_humidity_ml']['description'] = 'specific humidity on pressure sigmal levels'
    variable3d_arome['specific_humidity_ml']['precision'] = resol
    #OK
    #TKE in Arome france
    variable3d_arome['turbulent_kinetic_energy_ml'] = {}                                            # _FillValue: -2147483647, scale_factor: 1.0E-6
    variable3d_arome['turbulent_kinetic_energy_ml']['name'] = 'TKE'                                 # long_name: Turbulent kinetic energy (TKE), metno_name: turbulent_kinetic_energy
    variable3d_arome['turbulent_kinetic_energy_ml']['units'] = 'm^2/s^2'                            # units: m^2/s^2
    variable3d_arome['turbulent_kinetic_energy_ml']['description'] = 'Turbulent kinetic energy on pressure sigmal levels'
    variable3d_arome['turbulent_kinetic_energy_ml']['precision'] = resol
    #OK NB! Not used in flexpart, but it could perhaps be usefull for moist processes of haratu
    #CLOUD_FRACTI in arome france
    variable3d_arome['cloud_area_fraction_ml'] = {}                                                 # _FillValue: 9.96921E36
    variable3d_arome['cloud_area_fraction_ml']['name'] = 'CLDFRA'                                   # long_name: Cloud cover in model levels, standard_name: cloud_area_fraction
    variable3d_arome['cloud_area_fraction_ml']['units'] = 'none'                                    # units: %
    variable3d_arome['cloud_area_fraction_ml']['description'] = 'cloud fraction'
    variable3d_arome['cloud_area_fraction_ml']['precision'] = 1


    param2d_arome = [*variable2d_arome.keys()]
    dmap_arome2d, data_domain, bad_param = checkget_data_handler(date=modelruntime, use_latest=use_latest,
                                                                        model=model, step=steps, all_param=param2d_arome)
    print("2d done nicely")
   

    #UNCOMMENT IF U EVER NEED TO DEACCUMULATE SURFACE FLUX form aa instead of surfex
    #NBNBNB NOT USED JUST GOOD TO HAVE IN CASE WE CHANGE THE MIND OF WHAT TO USE IN FLEXPaRT
    deaccumulate = True #euther deaccumulate here or in flexpart
    if deaccumulate:    #handle integral_of_surface_downward_sensible_heat_flux_wrt_time to deaccumulate
        print("start deaccumulating")
        averaged_sensible_heat_flux=deepcopy(dmap_arome2d.integral_of_surface_downward_sensible_heat_flux_wrt_time)
        variable2d_arome['averaged_sensible_heat_flux'] = {}
        variable2d_arome['averaged_sensible_heat_flux']['name'] = 'SSHF_calcAA'
        variable2d_arome['averaged_sensible_heat_flux']['units'] = 'J.m-2.s-1'
        variable2d_arome['averaged_sensible_heat_flux']['description'] = 'One hour Averaged Sensible heat flux positive upward'
        variable2d_arome['averaged_sensible_heat_flux']['precision'] = resol
        param2d_arome = [*variable2d_arome.keys()]
    
        if len(steps) > 1:
            if steps[0] <= 1 :
                print("A")
                deaccum = dmap_arome2d.integral_of_surface_downward_sensible_heat_flux_wrt_time[1:,:] -  dmap_arome2d.integral_of_surface_downward_sensible_heat_flux_wrt_time[0:-1,:]
            elif steps[0] > 1:
                deaccum = np.empty((np.shape(dmap_arome2d.integral_of_surface_downward_sensible_heat_flux_wrt_time)))
                dmet_surf, data_domain, bad_param = checkget_data_handler(date=modelruntime, use_latest=use_latest,
                                                                        model=model, step=steps[0]-1, all_param=["integral_of_surface_downward_sensible_heat_flux_wrt_time"])
                deacum1= dmap_arome2d.integral_of_surface_downward_sensible_heat_flux_wrt_time[0,:] - dmet_surf.integral_of_surface_downward_sensible_heat_flux_wrt_time[0,:]
                deaccum2 = dmap_arome2d.integral_of_surface_downward_sensible_heat_flux_wrt_time[1:,:] -  dmap_arome2d.integral_of_surface_downward_sensible_heat_flux_wrt_time[0:-1,:]
                deaccum[0,:]  = deacum1
                deaccum[1:,:] = deaccum2
            else:
                print("B")
                deaccum = dmap_arome2d.integral_of_surface_downward_sensible_heat_flux_wrt_time[2:,:] -  dmap_arome2d.integral_of_surface_downward_sensible_heat_flux_wrt_time[1:-1,:]
    
        elif len(steps) == 1:
            if steps[0] == 1: #first hour do not need de-accumulation
                print("C")
                deaccum = dmap_arome2d.integral_of_surface_downward_sensible_heat_flux_wrt_time[0:,:] 
                pass
            elif steps[0] !=0: #ok
                print("D")
                deaccum = np.empty((np.shape(dmap_arome2d.integral_of_surface_downward_sensible_heat_flux_wrt_time)))
                dmet_surf, data_domain, bad_param = checkget_data_handler(date=modelruntime, use_latest=use_latest,
                                                                        model=model, step=steps[0]-1, all_param=["integral_of_surface_downward_sensible_heat_flux_wrt_time"])
                deaccum[0,:] = dmap_arome2d.integral_of_surface_downward_sensible_heat_flux_wrt_time[0,:] -  dmet_surf.integral_of_surface_downward_sensible_heat_flux_wrt_time[0,:]
            
            elif steps[0] == 0: #ok
                print("E")
                print("ERROR: leadtime 0 not possible due to lack of accumulated sensibile heat at initial time")
                exit(1)
        
        if steps[0] == 0 and steps[1] != 1:
            averaged_sensible_heat_flux[1:] = deaccum[:]
        elif steps[0] == 0 and steps[1] == 1:
            averaged_sensible_heat_flux[2:] = deaccum[:]
        elif steps[0] == 1:
            averaged_sensible_heat_flux[1:] = deaccum[:]
        else:
            averaged_sensible_heat_flux[:] = deaccum[:]
    
        dmap_arome2d.averaged_sensible_heat_flux= -averaged_sensible_heat_flux[:]/3600.
    
    #    #just to compare calc
    #    print(np.shape(deaccum))
    #    print(dmap_arome2d.integral_of_surface_downward_sensible_heat_flux_wrt_time[:,0,308,262])
    #    print(deaccum[:,0,308,262])
    #    print(averaged_sensible_heat_flux[:,0,308,262])
    #    print(dmap_arome2d.averaged_sensible_heat_flux[:,0,308,262])
    #    print("donde deaccumulating")
    
    param3d_arome = [*variable3d_arome.keys()]
    print(param3d_arome)
    dmap_arome3d, data_domain, bad_param = checkget_data_handler(date=modelruntime, m_level=lvl, use_latest=use_latest,
                                                         model=model, step=steps, all_param=param3d_arome)
    print("3d done nicely")
   
    print("DONE")
    print(bad_param)
    for bp in bad_param:
        param3d_arome.remove(bp)

    print("retrive sfxarome")
    param2d_sfx = [*variable2d_sfx.keys()]
    print(param2d_sfx)
    dmap_sfx2d, data_domain, bad_param = checkget_data_handler(date=modelruntime, use_latest=use_latest,
		                                                             model=model, step=steps, all_param=param2d_sfx)
    print("badparam") 
    print(bad_param)

    #attr
    checkproj = check_data(date=modelruntime, model=model, param=["projection_lambert","x","y"], use_latest=use_latest)
    url= checkproj.file["url"][0]
    dataset = Dataset(url)
    attr = {}
    proj = dataset.variables["projection_lambert"]

    for tidx in np.arange(0, len(steps), 1):
        t = steps[tidx]
        print("Inside for loop")
        output = outputpath + modelruntime + "/"
        print(output)
        if not os.path.exists(output):
            os.makedirs(output)
            print("Directory ", output, " Created ")
        else:
            print("Directory ", output, " exist ")
        print("####################################################################################")
        print(t)
        print("####################################################################################")
        validdate = datetime.datetime(int(modelruntime[0:4]), int(modelruntime[4:6]), int(modelruntime[6:8]), int(modelruntime[8:10])) + datetime.timedelta(hours=int(t))
        date_time = validdate.strftime("%Y%m%d_%H")
        print("####################################################################################")
        print(date_time)
        print("####################################################################################")
        #flexpart dont like 00, want 24
        if validdate.hour==0:
            dateminus1d=validdate - datetime.timedelta(days=1)
            date_time=dateminus1d.strftime("%Y%m%d") + "_24"
            #d = datetime.today() - timedelta(days=days_to_subtract)

        print(date_time)
        ncid = Dataset(output+ 'AR' +  date_time + '.nc', 'w')
        attr['reference_lon'] = proj.getncattr("longitude_of_central_meridian")
        attr['ydim'] = np.long(len(dmap_arome2d.y[::yres]))#np.long(dataset.variables["y"].getncattr("_ChunkSizes"))  # Use: None
        attr['forecast'] = validdate.strftime("%H")  # 23
        attr['x_resolution'] = np.double("2500.0") #np.double("2500.0")*xres  # Use: None
        attr['center_lon'] = proj.getncattr("longitude_of_central_meridian")
        attr['rotation_radian'] = 0.0
        attr['xdim'] = np.long(len(dmap_arome2d.x[::xres]))#np.long(dataset.variables["x"].getncattr("_ChunkSizes"))  # Use: None
        attr['input_lat'] = proj.getncattr("latitude_of_projection_origin")  # Use: None
        attr['reference_lat'] = proj.getncattr("latitude_of_projection_origin")
        attr['y_resolution'] = np.double("2500.0") # 
        attr['date'] =  validdate.strftime("%Y%m%d") # "20180331"
        attr['input_lon'] = proj.getncattr("longitude_of_central_meridian")  # Use: None
        print(attr['input_lon'])
        print(attr['input_lat'])
        #exit(1)
        find_inputpos = nearest_neighbour(plat=attr['input_lat'], plon=attr['input_lon'],longitudes=dmap_arome2d.longitude, latitudes=dmap_arome2d.latitude )
        print(find_inputpos[0])
        attr['input_position'] = 'none'#np.double(find_inputpos[0])#find_inputpos[0][0] # 359??  # Use: None
        attr['geoid'] = proj.getncattr("earth_radius") #6370000#6371229.0 #
        attr['center_lat'] = proj.getncattr("latitude_of_projection_origin")

        print("Create netcdf")
        ncid.setncatts(attr)
        time= ncid.createDimension('time', 1)
        x = ncid.createDimension('X', len(dmap_arome2d.x[::xres]))
        y = ncid.createDimension('Y', len(dmap_arome2d.y[::yres]))
        level = ncid.createDimension('level', len(dmap_arome3d.hybrid))
        times = ncid.createVariable('time', 'i4', ('time',)) #AMJ
        times[:] = dmap_arome2d.time[tidx]
        times.units = dmap_arome2d.units.time

        xs = ncid.createVariable('X', 'i4', ('X',))
        xs.units = 'none'
        xs[:] = range(1, len(dmap_arome2d.x[::xres]) + 1)
        ys = ncid.createVariable('Y', 'i4', ('Y',))
        ys.units = 'none'
        ys[:] = range(1, len(dmap_arome2d.y[::yres]) + 1)

        levels = ncid.createVariable('level', 'i4', ('level',))
        levels.units = 'none'
        levels[:] = range(1,  len(dmap_arome3d.hybrid) + 1 )

        nc_ak = ncid.createVariable('Ak', 'f4', ('level',))
        nc_ak.units = 'none'
        nc_ak[:] = dmap_arome3d.ap
        nc_bk = ncid.createVariable('Bk', 'f4', ('level',))
        nc_bk.units = 'none'
        nc_bk[:] = dmap_arome3d.b

        vid = ncid.createVariable('LON', 'f4', ('Y', 'X'), zlib=True)
        vid.description = 'longitude of the center grid'
        vid[:] = dmap_arome2d.longitude[::xres,::yres]
        vid = ncid.createVariable('LAT', 'f4', ('Y', 'X'), zlib=True)
        vid.description = 'latitude of the center grid'
        vid[:] = dmap_arome2d.latitude[::xres,::yres]
        print(param3d_arome)

        for param in param3d_arome:
            vid = ncid.createVariable(variable3d_arome[param]['name'], 'f4',('level','Y','X'),zlib=True)
            vid.units = variable3d_arome[param]['units']
            vid.description = variable3d_arome[param]['description']
            expressiondata = f"dmap_arome3d.{param}[{tidx},:,::{xres},::{yres}]"
            print(expressiondata)
            data = eval(expressiondata)
            vid[:] = data

        print(param2d_arome)
        for param in param2d_arome:
            vid = ncid.createVariable(variable2d_arome[param]['name'], 'f4', ('Y', 'X'), zlib=True)
            vid.units = variable2d_arome[param]['units']
            vid.description = variable2d_arome[param]['description']
            expressiondata = f"dmap_arome2d.{param}[{tidx},0,::{xres},::{yres}]"
            print(expressiondata)
            data = eval(expressiondata)
            if param =="surface_air_pressure":
                print(param)
                #data = np.log(data) Only needed if log oressure is expected
            vid[:] = data
        print(param2d_sfx)
        for param in param2d_sfx:
            vid = ncid.createVariable(variable2d_sfx[param]['name'], 'f4', ('Y', 'X'), zlib=True)
            vid.units = variable2d_sfx[param]['units']
            vid.description = variable2d_sfx[param]['description']
            expressiondata = f"dmap_sfx2d.{param}[{tidx},::{xres},::{yres}]"
            print(expressiondata)
            data = eval(expressiondata)
            vid[:] = data


        ncid.close()

def fix(outputpath, modelruntime, steps=[0, 64], lvl=[0, 64], archive=1):
    print(modelruntime)

    if "cyclone.hpc.uib.no" in platform.node() and outputpath == None:
        print("detected cyclone")
        # outputpath="/Data/gfi/work/cat010/flexpart_arome/input/"
        # outputpath="/Data/gfi/work/hso039/flexpart_arome/input/"
        user = os.environ['USER']
        outputpath = "/Data/gfi/projects/isomet/projects/ISLAS/flexpart-arome_forecast/data/{0}/input/".format(user)
    elif outputpath != None:
        outputpath = outputpath
    else:
        outputpath = "./"
        print("LOCAL")

    model = "AromeArctic"
    xres = 1
    yres = 1
    use_latest = False if archive == 1 else True
    variable2d = retrieve_arome4flexpart(outputpath, modelruntime, steps, lvl, xres, yres, model, use_latest)


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--datetime", help="YYYYMMDDHH for modelrun", required=True, type=str)
  parser.add_argument("--steps", default=[0,64], nargs="+", type=int,help=" j")
  parser.add_argument("--m_levels", default=[0,64], nargs="+", type=int,help="model level, 64 is lowest")
  parser.add_argument("--archive", default=1, type=int,help="fetch from archive if 1")
  parser.add_argument("--outputpath", default=None, type=str,help="where to save")

  args = parser.parse_args()
  m_levels = list(np.arange(args.m_levels[0], args.m_levels[-1]+1, 1))

  if args.steps[0]==0: 
      print("NB 0 ledtime can not be used due to sensibile heat not available here")
      args.steps.remove(0)
      if not args.steps:
          exit(1)
  steps = list(np.arange(args.steps[0], args.steps[-1]+1, 1)) if (len(args.steps)>1 and args.steps[0]!=args.steps[-1]) else [args.steps[0]]
  print(steps)
  #exit(1)
  fix(args.outputpath, args.datetime, steps, m_levels, args.archive)
