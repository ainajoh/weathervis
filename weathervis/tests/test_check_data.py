import unittest
from weathervis.checkget_data_handler import *
import warnings
import sys, os
from datetime import datetime, timedelta
from tests import *

#MethodName_StateUnderTest_ExpectedBehavior

class checkgetdata(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)


        self.to_old_date = "1999010100"
        self.archive_date = "2020010100" #strftime('%Y%m%d')
        self.latest_date = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d') + "00"
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



        #self.checkMEPSonDate = check_data(model="MEPS", date="2020010100")

    #DATES
    def test_most_used_variables(self):
        check = check_data(model=self.model_aa, numbervar=5)
        print(check)
        print(check.param)

    def test_search_variable_name(self):
        check = check_data(model=self.model_aa, search="wind")
        print(check)
        print(check.param)

    def test_search_variable_name_on_date(self):
        check = check_data(model=self.model_aa, date=self.archive_date, search="wind", use_latest=False)
        print(check.param)

    def test_find_available_dates(self):
        check = check_data(model=self.model_aa)
        print(check.date)

    def test_find_allfiles_for_date(self):
        check = check_data(date=self.archive_date, model=self.model_aa)
        print(check.file)

    def test_find_allfiles_for_date_param(self):
        check = check_data(date=self.archive_date, model=self.model_aa, param=self.one_good_param)
        print(check.file)

    def BAD_find_allfiles_for_date_plevel(self):
        check = check_data(date=self.archive_date, model=self.model_aa, p_level=[500])
        print(check.file)

    def BAD_find_allfiles_for_date_mlevel(self):
        check = check_data(date=self.archive_date, model=self.model_aa, m_level=[5,10])
        print(check.file)

    def BAD_find_allfiles_for_date_ens_mbrs(self):
        check = check_data(date=self.archive_date, model=self.model_aa, mbrs=[1, 5])
        print(check.file)





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

