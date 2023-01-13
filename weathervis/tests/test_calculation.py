# - 20220307: no error.


import unittest
from weathervis.checkget_data_handler import *
import warnings
import sys, os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

#from tests import *
#MethodName_StateUnderTest_ExpectedBehavior

class test_calculation(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self.to_old_date = "1999010100"
        self.archive_date = "2020010100" #strftime('%Y%m%d')
        self.latest_date = (datetime.today() - timedelta(hours=12)).strftime('%Y%m%d') + "00"
        self.to_futuristic_date = "2030010100"
        self.bad_format_date = "20200100"
        self.any_type_date = 2020010100

        self.model_meps="MEPS"
        self.model_aa="AromeArctic"
        self.model_meps_anyformat ="mePs"
        self.model_aa_anyformat ="aRomeArcTic"
        self.bad_model="coffeepot"

        self.one_good_param = ['air_pressure_at_sea_level']
        self.multiple_good_param = ['air_pressure_at_sea_level', 'pressure','specific_humidity_pl']

        self.one_bad_param = ['coffeepot']
        self.multiple_bad_param = ['coffeepot_small', 'large_bird', "dog"]

        self.multiple_goodandbad_param = ['air_pressure_at_sea_level', 'coffeepot', 'dog',' pressure' ]
        self.multiple_good_param_pl_ml_sfx_sfc = ["specific_humidity_pl","air_pressure_at_sea_level", "mass_fraction_of_graupel_in_air_ml", "SIC","LE_SEA"]

        self.one_step=0
        self.one_step_in_array = [0]
        self.two_step_far_appart = [3,9]
        self.multiple_step = [0,1,2,3,4,7]

        self.url_base = "https://thredds.met.no/thredds/dodsC/alertness/users/marvink/CAO2015/fc2015122400_fp.nc"

    def test_ml2gph(self):
        url = "https://thredds.met.no/thredds/dodsC/alertness/users/marvink/AMS_publication/CAO2015_cy40ref.nc"
        param = ["geoph_ml", "ap", "b","surface_air_pressure","p0","air_temperature_ml","specific_humidity_ml","surface_geopotential"]
        point_name=["Arcticocean"]
        m_levels=np.arange(0,65,1)
        dmet, data_domain, bad_param = checkget_data_handler(url =url, all_param=param, point_name=point_name )
        #mlz = ml2alt_gl(dmet.air_temperature_ml, dmet.specific_humidity_ml, dmet.ap, dmet.b, dmet.surface_air_pressure, inputlevel="full", returnlevel="full") #ml2pl( dmet.ap, dmet.b, 
        mlz = ml2alt(air_temperature_ml=dmet.air_temperature_ml, specific_humidity_ml=dmet.specific_humidity_ml, ap= dmet.ap, b=dmet.b, surface_air_pressure=dmet.surface_air_pressure)

        #mlz = ml2alt_mk(dmet.surface_geopotential, m_levels)
        #mlz = tmp_aina( dmet.air_temperature_ml, dmet.specific_humidity_ml, dmet.ap, dmet.b, dmet.p0,  #inputlevel="half", returnlevel="full") #ml2pl( dmet.ap, dmet.b, 
        mlz_cheat=cheat_alt(dmet.m_level)
        print("test shape")
        print(np.shape(mlz))
        mlz=mlz
        print(np.shape(mlz))
        print(mlz)
        mlz_cheat = mlz


        dmet.geoph_ml/= 9.80665
        dmet.surface_geopotential = dmet.surface_geopotential.squeeze() /9.80665
        dmet.geoph_ml = dmet.geoph_ml.squeeze()[::-1]
        mlz = mlz.squeeze()[::-1]  + dmet.surface_geopotential
        mlz_cheat = mlz_cheat.squeeze()[::-1] + dmet.surface_geopotential
        levels = dmet.m_level#[::-1]
        mean_z = np.mean(np.array([ mlz,mlz_cheat ]), axis=0 )
        
        diff_calc = np.subtract(mlz, dmet.geoph_ml)
        diff_cheat = mlz_cheat-dmet.geoph_ml
        diff_mean = mean_z-dmet.geoph_ml
        diff_depth = dmet.geoph_ml[1:] - dmet.geoph_ml[0:-1]
        #mlz[1:] = mlz[1:] - diff_depth[:]

        fig, ax = plt.subplots(figsize=(10,6)) #    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 9), subplot_kw={'projection': crs})
        #ax2=ax.twinx(); ax2.set_ylim(0,100)
        ax.plot(levels, dmet.geoph_ml, "o" ,label="geoph_ml" )
        ax.plot(levels, mlz,"o", label = "calc_geoph")
        #ax.plot(levels, mlz_cheat, label = "calc_cheat")
        ax.grid()
        ax.legend()
        ax.set_xlabel("Vertical levels")
        ax.set_ylabel("Height [m]")
        plt.show()

        fig2, ax2 = plt.subplots(figsize=(10,6))
        start =0
        end = 64
        sep = 5
        #fig2.subplots_adjust(bottom=0.75)
        ax2.plot(levels[start:end],diff_calc[start:end],"o", label = "diff_calc")
        #ax2.plot(dmet.geoph_ml[0:15],diff_calc[0:15], label = "diff_calc") diff_mean
        #ax2.plot(levels[start:end],diff_cheat[start:end], label = "diff_cheat")
        #ax2.plot(levels[start:end],diff_mean[start:end], label = "diff_mean")


        ax2_2=ax2.twiny()
        ax2_2.set_xlim(ax2.get_xlim())
        ax2_2.set_xticks(levels[start:end:sep])
        ax2_2.set_xticklabels(np.rint(diff_depth[start:end:sep]).astype('i'))

        ax2_3=ax2.twiny()
        ax2_3.set_xlim(ax2.get_xlim())
        ax2_3.set_xticks(levels[start:end:sep])
        ax2_3.set_xticklabels(np.rint(dmet.geoph_ml[start:end:sep]).astype('i'))
        ax2_3.tick_params(axis='x',  pad=20)
        ax2_3.xaxis.set_ticks_position("bottom")
        ax2_3.xaxis.set_label_position("bottom")
        ax2.grid()
        ax2.legend()

        ax2.set_ylabel("Height differenc [m]")
        ax2_3.set_xlabel("Vertical levels & Height [m]")
        ax2_2.set_xlabel("Depth between model levels in [m]")
        plt.show()

    def test_ml2pl(self):
        param = [ "ap", "b","surface_air_pressure","air_temperature_ml","surface_geopotential", "specific_humidity_ml", "p0"]
        #param= ["surface_geopotential"]
        point_name=["plmeasure"]
        m_levels=np.arange(0,65,1)
        datetime="2020020112"
        datetime="2020032012"

        step = 0
        #dmet, data_domain, bad_param = checkget_data_handler(model="AromeArctic", all_param=param, point_name=point_name, m_level=m_levels, date=datetime)
        dmet, data_domain, bad_param =checkget_data_handler(model=self.model_aa, point_name=point_name,date=datetime, all_param=param, step= 0, use_latest=False,m_level=m_levels)
        print(dmet.p0)
        print(dmet.surface_air_pressure)
        ml2pl_data = ml2pl(ap= dmet.ap, b=dmet.b, surface_air_pressure=dmet.surface_air_pressure)
        print(np.shape(ml2pl_data))
        #print(ml2pl_data[-1])
        #mlz = ml2alt_gl(dmet.air_temperature_ml, dmet.specific_humidity_ml, dmet.ap, dmet.b, dmet.surface_air_pressure, inputlevel="full", returnlevel="full") #ml2pl( dmet.ap, dmet.b, 
        mlz_gl = ml2alt(air_temperature_ml=dmet.air_temperature_ml, specific_humidity_ml=dmet.specific_humidity_ml, ap= dmet.ap, b=dmet.b, surface_air_pressure=dmet.surface_air_pressure)
        mlz_sl=mlz_gl + dmet.surface_geopotential/9.81

        #print(mlz[-1])
        ml2pl_data=ml2pl_data.squeeze()
        mlz_gl=mlz_gl.squeeze()
        mlz_sl=mlz_sl.squeeze()

        print(ml2pl_data[-1])
        print(mlz_gl[-1])
        print(mlz_sl[-1])
        print(dmet.surface_geopotential/9.81)





        

    def test_great_circle_distance(self):
        d= great_circle_distance(0, 55, 8, 45.5)
        #        self.assertEqual(np.shape(dmet.specific_humidity_pl), (1, 2, 949, 739))
        assert d == 1199.3240879770135

    def test_nearest_neighbour_idx(self):
        pass
    def test_nearest_neighbour(self):
        pass

    def test_rotate_points(self):
        dmet, data_domain, bad_param = checkget_data_handler(model=self.model_aa, date=self.archive_date, all_param=self.one_good_param, step= self.one_step, use_latest=False)
        parallels = dmet.standard_parallel_projection_lambert
        center_longitude = dmet.longitude_of_central_meridian_projection_lambert
        center_latitude = dmet.latitude_of_projection_origin_projection_lambert
        lon = 77.5 
        lat = 77.5 
        rlon,rlat = rotate_points(lon, lat, center_longitude, center_latitude, parallels,direction='n2r' )
        print(rlat[0])
        assert rlon == np.array([1610358.0024465572])
        assert rlat == np.array([1655116.700232346])
        lon = rlon[0]
        lat = rlat[0]
        rlon,rlat = rotate_points(lon, lat, center_longitude, center_latitude, parallels,direction='r2n' )
        assert rlon == np.array([77.5]) 
        assert rlat == np.array([77.5])

    def test_find_cross_points(self):
        dmet, data_domain, bad_param = checkget_data_handler(model=self.model_aa, date=self.archive_date, all_param=self.one_good_param, step= self.one_step, use_latest=False)
        parallels = dmet.standard_parallel_projection_lambert
        center_longitude = dmet.longitude_of_central_meridian_projection_lambert
        center_latitude = dmet.latitude_of_projection_origin_projection_lambert
        nya = (78.9243,11.9312)
        ands = (69.310, 16.120)
        coo = [( (nya[1], nya[0]), (ands[1], ands[0]) )]   
        plon,plat,linedist= find_cross_points(coo[0], nbre=1)
        #Double checked that this is in fact correct giving midle point between two positions
        assert plon == np.array([14.644598831763899])
        assert plat == np.array([74.12741997379624])
        assert linedist == 1077.1607866053605

    def test_get_AllPointsBetween2PosOnGrid(self):
        dmet, data_domain, bad_param = checkget_data_handler(model=self.model_aa, date=self.archive_date, all_param=self.one_good_param, step= self.one_step, use_latest=False)
        parallels = dmet.standard_parallel_projection_lambert
        center_longitude = dmet.longitude_of_central_meridian_projection_lambert
        center_latitude = dmet.latitude_of_projection_origin_projection_lambert
        nya = (78.9243,11.9312)
        ands = (69.310, 16.120)
        coo = [( (nya[1], nya[0]), (ands[1], ands[0]) )]   
        query_points, distances= get_AllPointsBetween2PosOnGrid(coo[0], center_longitude, center_latitude, parallels,
                                       pollon=None,pollat=None, model=None, 
                                       nbre=1,version="regular")
        print(query_points)
        print(distances)
        points = 1
        assert query_points[points][0] == 74.12741997379624 #lat
        assert query_points[points][1] == 14.644598831763899 #lon
        assert distances[0] == 0
        assert distances[1] == 538.5803933026802
        assert distances[2] == 1077.1607866053605

        query_points, distances= get_AllPointsBetween2PosOnGrid(coo[0], center_longitude, center_latitude, parallels,
                                       pollon=None,pollat=None, model=None, 
                                       nbre=1,version="rotated")
        
    def test_interpolate(self):
        param = ["air_temperature_ml","surface_air_pressure"]
        m_level=np.arange(0,64,1)
        dmet, data_domain, bad_param = checkget_data_handler(model=self.model_aa, m_level = m_level, date=self.archive_date, all_param=param, step= self.one_step, use_latest=False)
        p = ml2pl(dmet.ap, dmet.b, dmet.surface_air_pressure)
        dmet.air_temperature_ml = dmet.air_temperature_ml[0,:,:,:]
        p = p[0,:,:,:]
        grid = p
        interplevels = np.arange(0, 5000, 100)
        dataint = interpolate(dmet.air_temperature_ml, grid, interplevels)
        print(np.shape(dataint))

    def test_interpolate_grid(self):
        param = ["air_temperature_ml","surface_air_pressure"]
        m_level=np.arange(0,64,1)
        dmet, data_domain, bad_param = checkget_data_handler(model=self.model_aa, m_level = m_level, date=self.archive_date, all_param=param, step= self.one_step, use_latest=False)
        p = ml2pl(dmet.ap, dmet.b, dmet.surface_air_pressure)
        dmet.air_temperature_ml = dmet.air_temperature_ml[0,:,:,:]
        p = p[0,:,:,:]
        lat = dmet.latitude
        lon=dmet.longitude
        rlat=dmet.y
        rlon=dmet.x
        _lo = np.array([rlat.min(), rlon.min()])
        _hi = np.array([rlat.max(), rlon.max()])
        print("hi1")
        print(_lo)
        print("hi2")
        print(np.shape(dmet.air_temperature_ml.squeeze()))
        print(_lo)
        print(_hi)
        intfunc = interpolate_grid(griddata=dmet.air_temperature_ml.squeeze(), lo=_lo, hi=_hi,
                                    verbose=False, order=1)

    def tearDown(self):
        # Restore
        #remove logging perhaps
        #runs after every test
        #you could delete all created files forexample.
        pass

if __name__ =='__main__':
    unittest.main()
    #suite = unittest.TestSuite()
    #suite.addTest(checkgetdata("test_date_good___archive"))
    #runner = unittest.TextTestRunner()
    #runner.run(suite)
    #run one func:
    #python tests/test_calculation.py test_calculation.test_find_cross_points

