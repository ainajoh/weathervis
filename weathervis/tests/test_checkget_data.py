#Log:
# - 20220303: no error.
# - 20220304: no error.
# - 20220307: no error.
import unittest
from weathervis.checkget_data_handler import *
import warnings
import sys, os
from datetime import datetime, timedelta

#from tests import *

#MethodName_StateUnderTest_ExpectedBehavior

class checkgetdata(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        self.to_old_date = ["1999010100"]
        self.archive_date = ["2020010100"] #strftime('%Y%m%d')
        self.latest_date =  (datetime.today() - timedelta(hours=12)).strftime('%Y%m%d%H') #+ "00"
        self.latest_date = self.latest_date[:-2] + str(int(self.latest_date[-2:])-int(self.latest_date[-2:])%6).zfill(2)
        self.to_futuristic_date = "2030010100"
        self.bad_format_date = "20200100"
        self.any_type_date = 2020010100

        self.model_meps="MEPS"
        self.model_aa="AromeArctic"
        self.model_meps_anyformat ="mePs"
        self.model_aa_anyformat ="aRomeArcTic"
        self.bad_model="coffeepot"

        self.one_good_param = ['air_temperature_2m']
        self.multiple_good_param = ['air_pressure_at_sea_level', 'pressure','specific_humidity_pl']

        self.one_bad_param = ['coffeepot']
        self.multiple_bad_param = ['coffeepot_small', 'large_bird', "dog"]

        self.multiple_goodandbad_param = ['air_pressure_at_sea_level', 'coffeepot', 'dog',' pressure' ]
        self.multiple_good_param_pl_ml_sfx_sfc = ["specific_humidity_pl","air_pressure_at_sea_level", "mass_fraction_of_graupel_in_air_ml", "SIC","LE_SEA"]

        self.one_point_name=["Tromso"]

        self.one_step=0
        self.int64_step = np.int64(self.one_step)
        self.one_step_in_list = [self.one_step]
        self.one_step_in_nparray = np.array(self.one_step_in_list)

        self.multiple_step_in_list = [0,1,2,3,4,7]
        self.multiple_step_in_array = np.array(self.multiple_step_in_list)


        self.two_step_far_appart = [3,9]

        self.url_base = "https://thredds.met.no/thredds/dodsC/alertness/users/marvink/CAO2015/fc2015122400_fp.nc"



        #self.checkMEPSonDate = check_data(model="MEPS", date="2020010100")
    #TESTING STEP INPUT TYPES
    def test_one_step_type(self):
        dataint64, bad, no = checkget_data_handler(model=self.model_aa, date=self.archive_date, all_param=self.one_good_param, step= self.int64_step, use_latest=False, point_name=self.one_point_name)
        dataint, bad, no = checkget_data_handler(model=self.model_aa, date=self.archive_date, all_param=self.one_good_param, step= self.one_step, use_latest=False,point_name=self.one_point_name)
        datalist, bad, no = checkget_data_handler(model=self.model_aa, date=self.archive_date, all_param=self.one_good_param, step= self.one_step_in_list, use_latest=False,point_name=self.one_point_name)
        datanparray, bad, no = checkget_data_handler(model=self.model_aa, date=self.archive_date, all_param=self.one_good_param, step= self.one_step_in_nparray, use_latest=False,point_name=self.one_point_name)
        valueint = getattr(dataint, self.one_good_param[0])
        valueint64= getattr(dataint64, self.one_good_param[0])
        valuelist = getattr(datalist, self.one_good_param[0])
        valuearray = getattr(datanparray, self.one_good_param[0])

        self.assertEqual(valueint,valueint64, "int problem")
        self.assertEqual(valueint64,valuelist, "list problem")
        self.assertEqual(valuelist,valuearray, "array problem")

    def test_multiple_step_type(self):
        datalist, bad, no = checkget_data_handler(model=self.model_aa, date=self.archive_date, all_param=self.one_good_param, step= self.multiple_step_in_list, use_latest=False,point_name=self.one_point_name)
        datanparray, bad, no = checkget_data_handler(model=self.model_aa, date=self.archive_date, all_param=self.one_good_param, step= self.multiple_step_in_array, use_latest=False,point_name=self.one_point_name)
        valuelist = getattr(datalist, self.one_good_param[0])
        valuearray = getattr(datanparray, self.one_good_param[0])
        print(valuearray.tolist())
        print(valuelist.tolist())
        print(type(valuearray))
        print(type(valuelist))
        self.assertEqual(valuelist.tolist(),valuearray.tolist(), "array problem")


    def test_array_step_type(self):
        step = np.array([0,1])
        checkget_data_handler(model=self.model_aa, date=self.archive_date, all_param=self.one_good_param, step= step, use_latest=False)
   
    def test_list_step_type(self):
        step = [0,1]
        checkget_data_handler(model=self.model_aa, date=self.archive_date, all_param=self.one_good_param, step= step, use_latest=False)

    #DATES
    def test_date_good___archive(self): #OK
        checkget_data_handler(model=self.model_aa, date=self.archive_date, all_param=self.one_good_param, step= self.one_step, use_latest=False)

    def test_date_good___archive_meps(self):
        checkget_data_handler(model=self.model_meps, date=self.archive_date, all_param=self.one_good_param,
                              step=self.one_step, use_latest=False)

    def test_date_good___archive_url(self):
        checkget_data_handler(model=self.model_meps, date=self.archive_date, all_param=self.one_good_param,
                              step=self.one_step, use_latest=False, url=self.url_base)

    def test_date_correct___archive_url(self):
        #test missing model, date, step, and use_latest
        print(self.url_base)
        checkget_data_handler(all_param=self.one_good_param, url=self.url_base)

    def test_date_good___latest(self): #error
        checkget_data_handler(model=self.model_aa, date=self.latest_date, all_param=self.one_good_param, step= self.one_step, use_latest=True)
        checkget_data_handler(model=self.model_meps, date=self.latest_date,all_param=self.one_good_param,use_latest=True,  step=self.one_step)
    
    def test_date_good___forget_use_latest(self):#ERROR
        checkget_data_handler(model=self.model_aa, date=self.latest_date, all_param=self.one_good_param, step=self.one_step)
    
    
    def test_date_good___anytype(self): #ERROR
        checkget_data_handler(model=self.model_aa, date=self.any_type_date, all_param=self.one_good_param, step= self.one_step, use_latest=False)
    def test_date_bad___archived_with_uselatest(self):
        with self.assertRaises(ValueError) as error:
            checkget_data_handler(model=self.model_aa, date=self.archive_date, all_param=self.one_good_param,
                                    step=self.one_step, use_latest=True)
        self.assertIn('Not able to find any file', str(error.exception) )

    def date_bad___to_old_date(self):
        with self.assertRaises(ValueError) as error:
            checkget_data_handler(model=self.model_aa, date=self.to_old_date, all_param=self.one_good_param,
                                      step=self.one_step, use_latest=False)
        errormsg ='request is for a date earlier than what is' #This is deprecated for now.
        errormsg="Modelrun wrong: Either;"
        self.assertIn(errormsg, str(error.exception))

    def test_date_bad___to_futuristic_date(self):
        print("########test_date_bad___to_futuristic_date")
        with self.assertRaises(ValueError) as error:
            checkget_data_handler(model=self.model_aa, date=self.to_futuristic_date, all_param=self.one_good_param,
                                  step=self.one_step, use_latest=False)
        self.assertIn('latest or date on the form: YYYYMMDDHH', str(error.exception)) # does not match format '%Y%m%d%H%M'
    def test_date_bad___formatdate(self):
        print("test_date_bad___formatdate")
        with self.assertRaises(ValueError) as error:
            checkget_data_handler(model=self.model_aa, date=self.bad_format_date , all_param=self.multiple_good_param_pl_ml_sfx_sfc,
                                  step=self.one_step, use_latest=False)
        self.assertIn('latest or date on the form: YYYYMMDDHH', str(error.exception)) # does not match format '%Y%m%d%H%M'

    #MODELS
    def test_model_good___anyformat_aromearctic(self):
        checkget_data_handler(model=self.model_aa_anyformat, date=self.archive_date, all_param=self.one_good_param,
                              step=self.one_step, use_latest=False)
    def test_model_bad(self):
        with self.assertRaises(ValueError) as error:
            checkget_data_handler(model=self.bad_model, date=self.archive_date, all_param=self.one_good_param,
                              step=self.one_step, use_latest=False)
        self.assertIn('Model not found', str(error.exception))

    #parameters
    def test_parameter_good___multiple(self):
        checkget_data_handler(model=self.model_aa, date=self.archive_date, all_param=self.multiple_good_param,
                                  step=self.one_step, use_latest=False)
    def test_parameter_good___multiple_goodbad(self):
        #made such that it does not create an error if some parameters ar not found.. it just uses the good parameters.
        checkget_data_handler(model=self.model_aa, date=self.archive_date, all_param=self.multiple_goodandbad_param,
                                  step=self.one_step, use_latest=False)
    def test_parameter_bad___one(self):
        with self.assertRaises(ValueError) as error:
            checkget_data_handler(model=self.model_aa, date=self.archive_date, all_param=self.one_bad_param,
                                  step=self.one_step, use_latest=False)
        self.assertIn('No matches for your parameter found', str(error.exception)) # does not match format '%Y%m%d%H%M'
    def test_parameter_bad___multiple(self):
        with self.assertRaises(ValueError) as error:
            checkget_data_handler(model=self.model_aa, date=self.archive_date, all_param=self.multiple_bad_param,
                                  step=self.one_step, use_latest=False)
        self.assertIn('No matches for your parameter found', str(error.exception)) # does not match format '%Y%m%d%H%M'


    def test_parameter_good___pl_ml_sfx_sfc(self):
        print("#####################################################################################")
        print("test_parameter_good___pl_ml_sfx_sfc")
        dmet, data_domain, bad_param = checkget_data_handler(model=self.model_aa, date=self.latest_date, all_param=self.multiple_good_param_pl_ml_sfx_sfc,
                                  step=self.one_step, use_latest=False)

        self.assertEqual(np.shape(dmet.specific_humidity_pl), (1, 13, 949, 739))
        self.assertEqual(np.shape(dmet.air_pressure_at_sea_level), (1, 1, 949, 739))
        self.assertEqual(np.shape(dmet.mass_fraction_of_graupel_in_air_ml), (1, 65, 949, 739))
        self.assertEqual(np.shape(dmet.SIC), (1, 1,949, 739))
        #self.assertEqual(np.shape(dmet.LE_SEA), (1, 949, 739))
    def test_parameter_good___pl_ml_sfx_sfc_specificlevels(self):
        print("#####################################################################################")
        print("test_parameter_good___pl_ml_sfx_sfc_specificlevels")
        print(self.latest_date)
        #this is something i moght want to change, from and to VS values specific i belive abouut this centos might be different. should check
        dmet, data_domain, bad_param = checkget_data_handler(model=self.model_aa, date=self.latest_date, all_param=self.multiple_good_param_pl_ml_sfx_sfc,
                                  step=self.one_step, use_latest=False, m_level=[1,2,5], p_level=[1000,500])

        self.assertEqual(np.shape(dmet.specific_humidity_pl), (1, 2, 949, 739))
        self.assertEqual(np.shape(dmet.air_pressure_at_sea_level), (1, 1, 949, 739))
        self.assertEqual(np.shape(dmet.mass_fraction_of_graupel_in_air_ml), (1, 3, 949, 739))
        self.assertEqual(np.shape(dmet.SIC), (1,1, 949, 739))
        #self.assertEqual(np.shape(dmet.LE_SEA), (1, 1,949, 739)) = () error



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

