# - 20220307: no error.


import unittest
from weathervis.checkget_data_handler import *
import warnings
import sys, os
from datetime import datetime, timedelta
#from tests import *
#MethodName_StateUnderTest_ExpectedBehavior

class plots(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self.to_old_date = ["1999010100"]
        self.archive_date = ["2020010100"] #strftime('%Y%m%d')
        self.latest_date = (datetime.today() - timedelta(hours=12)).strftime('%Y%m%d') + "00"
        self.to_futuristic_date = ["2030010100"]
        self.bad_format_date = ["20200100"]
        self.any_type_date = [2020010100]

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

    def test_BLH_map(self):
        #self.archive_date
        from plots import BLH_map
        BLH_map.BLH(datetime=self.archive_date , model="AromeArctic", domain_name=["Svalbard"], coast_details="auto", use_latest = None, delta_index=None,steps=[0], domain_lonlat=None, legend=False, info=False, grid=True,
        runid=None, outpath=None, url=None, point_lonlat =None,overlays=None, point_name=None)

        BLH_map.BLH(datetime=self.archive_date , model="MEPS", domain_name=["South_Norway"], coast_details="auto", use_latest = None, delta_index=None,steps=[0], domain_lonlat=None, legend=False, info=False, grid=True,
        runid=None, outpath=None, url=None, point_lonlat =None,overlays=None, point_name=None)

    def test_BLH_map_anymodelformat(self):
        #self.archive_date
        from plots import BLH_map
        BLH_map.BLH(datetime=self.archive_date , model="aromeArctic", domain_name=["Svalbard"], coast_details="auto", use_latest = None, delta_index=None,steps=[0], domain_lonlat=None, legend=False, info=False, grid=True,
        runid=None, outpath=None, url=None, point_lonlat =None,overlays=None, point_name=None)

        #BLH_map.BLH(datetime="2022030300", model="MEPS", domain_name=["South_Norway"], coast_details="auto", use_latest = None, delta_index=None,steps=[0], domain_lonlat=None, legend=False, info=False, grid=True,
        #runid=None, outpath=None, url=None, point_lonlat =None,overlays=None, point_name=None)
    
    
    def test_OLR_map(self):
        #self.archive_date
        from plots import OLR_map
        OLR_map.OLR(datetime=self.archive_date , model="AromeArctic", domain_name=["Svalbard"], coast_details="auto", use_latest = None, delta_index=None,steps=[0], domain_lonlat=None, legend=False, info=False, grid=True,
        runid=None, outpath=None, url=None, point_lonlat =None,overlays=None, point_name=None)

        OLR_map.OLR(datetime=self.archive_date , model="MEPS", domain_name=["South_Norway"], coast_details="auto", use_latest = None, delta_index=None,steps=[0], domain_lonlat=None, legend=False, info=False, grid=True,
        runid=None, outpath=None, url=None, point_lonlat =None,overlays=None, point_name=None)

    def test_CAOindex_map(self):
        #self.archive_date
        from plots import CAOindex_map
        CAOindex_map.CAO(datetime=self.archive_date , model="AromeArctic", domain_name=["Svalbard"], coast_details="auto", use_latest = None, delta_index=None,steps=[0], domain_lonlat=None, legend=False, info=False, grid=True,
        runid=None, outpath=None, url=None, point_lonlat =None,overlays=None, point_name=None)

        CAOindex_map.CAO(datetime=self.archive_date , model="MEPS", domain_name=["South_Norway"], coast_details="auto", use_latest = None, delta_index=None,steps=[0], domain_lonlat=None, legend=False, info=False, grid=True,
        runid=None, outpath=None, url=None, point_lonlat =None,overlays=None, point_name=None)


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
    #python tests/test_checkget_data.py checkgetdata.test_date_good___archive

