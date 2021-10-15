#######################################################################
# File name: check_data.py
# This file is part of: weathervis
########################################################################
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import logging
from collections import Counter
import os
import gc
"""
###################################################################
This module checks what data is available, gives information on the dataset 
and choose the user prefered dataset.
------------------------------------------------------------------------------
Usage:
------
check = check_data(model, date = None, param=None, mbrs=None, file = None, numbervar = 100, search = None)
Returns:
------
Object with properties
"""
package_path = os.path.dirname(__file__)

all_models = ["aromearctic", "meps"] #ECMWF later
source = ["thredds"] # later"netcdf", "grib" 2019120100

#use_latest = True

logging.basicConfig(filename="get_data.log", level = logging.INFO, format = '%(levelname)s : %(message)s')

def SomeError( exception = Exception, message = "Something did not go well" ):
    #source: https://softwareengineering.stackexchange.com/questions/222586/how-should-you-cleanly-restrict-object-property-types-and-values-in-python
    logging.error(exception(message))
    if isinstance( exception.args, tuple ):
        raise exception
    else:
        raise exception(message)

def filter_param(file,param):
    """Used by check_data: Remove files not containing a givet set of parameters.
    Returns only files containing all the user defined parameters."""
    if param:  #If a user param is given
        for i in range(0, len(file)): # go through all files,
            param_bool = np.array([key in file.loc[i,"var"].keys() for key in param])
            if all(param_bool) == False:  #remove those files not having that parameter.
                file.drop([i], inplace=True)
    file.reset_index(inplace=True, drop=True)
    logging.info(file)
    return file

def filter_type(file,mbrs, p_level,m_level):
    """Used by check_data: Remove files not having the userdefined mbrs or levels
    Returns only files containing all the user defined preferences."""
    if mbrs != 0 and mbrs != None:
        #file = file[file["mbr_bool"] == True] deprecated
        file = file[~file.mbr_ens.isnull()] #not needed?
        file.reset_index(inplace=True)      #not needed?
        def find_mbrs(value):
            df_val = pd.DataFrame(value).isin(mbrs).sum(axis=0)
            if len(df_val) >= 1:
                return True
            else:
                return False
        ll = file["mbr_bool"].apply(find_mbrs)
        file = file[ll]


    if m_level != None:
        print("FILT")

        file = file[~file.m_levels.isnull()] #not needed?
        file.reset_index(inplace=True)       #not needed?
        print(file.columns)
        #file = file[file["ml_bool"] == True]  #deprecated


        #print(file.dim)

        #exit(1)
        #def find_mlevel(value):
        #   #print()
        #
        #    print("find_mlevel")
        #    val = value.values() #            df_val = pd.DataFrame(value).isin(p_level).sum(axis=0)
        #
        #    dd = pd.DataFrame([val])#.index.values
        #    print(dd)

    elif p_level:
        file = file[~file.p_levels.isnull()]   #not needed?
        file.reset_index(inplace=True)         #not needed?
        def find_plevel(value):
            df_val = pd.DataFrame(value).isin(p_level).sum(axis=0)
            if len(df_val) >=1:
                return True
            else:
                return False
        ll= file["p_levels"].apply(find_plevel)
        file = file[ll]
    file.reset_index(inplace=True, drop=True)
    return file

def filter_step(file,maxstep):
    if maxstep != None:
        for i in range(0, len(file)): # go through all files,
            step_bool = int(file.loc[i,"dim"]["time"]["shape"]) >= maxstep
            if step_bool == False:
                file.drop([i], inplace=True)
    file.reset_index(inplace=True, drop=True)
    return file

def filter_function_for_date(value, check_earliest=False, model = "AromeArctic"):
    if (value != None) and (len(value) == 10) and (int(value[0:4]) in range(2000, 2030)) and (int(value[4:6]) in range(0, 13)) \
            and (int(value[6:8]) in range(1, 32)) and (int(value[9:10]) in range(0, 25)):
        pass
    elif value == None:
        pass
    else:
        SomeError(ValueError, f'Modelrun wrong: Either; "latest or date on the form: YYYYMMDDHH')

    if check_earliest and value !=None: #deprecated for now as it is a waist of time just to get nice error
        date_all = check_data.check_available_date(model=model)
        date_first = pd.to_datetime(date_all.Date[0], format='%Y%m%d')
        date_req =  pd.to_datetime(date, format='%Y%m%d%H%M')
        if date_req < date_first:
            SomeError(ValueError, f"Your request is for a date earlier than what is available. You requested {date_req}, but the earlies is {date_first}")

filter_function_for_models = lambda value: value if value.lower() in all_models else SomeError(ValueError, f'Model not found: choices:{all_models}')

def filter_function_for_models_ignore_uppercases(model):
    model_lower = model.lower()
    if model_lower in all_models:
        pass
    else:
        SomeError(ValueError, f'Model not found: choices:{all_models}')

    model = "MEPS" if model_lower == "meps" else "AromeArctic" if model_lower == "aromearctic" else None

    return model

def filter_any(file):
    """Used by check_data: Remove random files until only one left
    Returns only one file.
    Todo: find a better way as this might not be what the use expect"""
    if len(file) > 1:  # want to end up with only one file.
        file = file[0]
        file.reset_index(inplace=True, drop=True)
    return file

def filter_firstdate(file,date):
    if maxstep != None:
        for i in range(0, len(file)): # go through all files,
            step_bool = int(file.loc[i,"dim"]["time"]["shape"]) >= maxstep
            if step_bool == False:
                file.drop([i], inplace=True)
    file.reset_index(inplace=True, drop=True)
    return file


class check_data():

    def __init__(self, model=None, date = None,param=None, step=None, mbrs=None, p_level= None, m_level=None,file = None, numbervar = 100, search = None, use_latest=False,url=None):
        """
        Parameters
        ----------
        date:  Modelrun as string in format YYYYMMDDHH
        model: Weathermodel, either: MEPS, AromeArctic
        url: opendal url to a specific file
        param: Parameters as strings in a list
        mbrs:  Which ensemble member
        file: an object containing file info. The main outcome of return in this class
        numbervar: max number of listed pameter, for searching.
        search: Parameter to search for.
        p_level: Which pressure levels in hPa
        m_level: WHich model level in hPa
        step:
        use_latest: True if you want a recent date, False if u want to get from the archive
        """
        logging.info("# check_data() #\n#################")

        self.date = str(date) if date != None else None
        self.model = model
        self.url = url
        self.param = param
        self.mbrs = mbrs
        self.file = file
        self.numbervar = numbervar
        self.search = search
        self.p_level = p_level
        self.m_level = m_level

        if step != None and type(step[0]) == str:
            if len(step) == 1:
                if ":" in step[0]:
                    ss = step[0].split(":")
                    ss = [int(x.strip()) for x in ss]
                    if len(ss) == 3:
                        step = list(np.arange(ss[0], ss[1], ss[2]))
                    else:
                        step = list(np.arange(ss[0], ss[1], 1))
                elif "," in step[0]:
                    ss = step[0].split(",")
                    step = [int(x.strip()) for x in ss]
                else:
                    step = [int(x.strip()) for x in step]
            else:
                step = [int(x.strip()) for x in step]

        self.maxstep = np.max(step) if step != None or type(step) != float or type(step) != int else step
        self.use_latest=use_latest

        self.check_web_connection()
        filter_function_for_date(self.date)

        self.model = filter_function_for_models_ignore_uppercases(self.model) if self.model != None else None
        self.p_level = [p_level] if p_level != None and type(p_level) != list else p_level
        self.m_level = [m_level] if m_level != None and type(m_level) != list else m_level
        if (self.model !=None and self.date !=None) or self.url !=None:
            all_files = self.check_files(date, model, param,  mbrs, url) #the main outcome
            self.file = self.check_file_info(all_files, param, mbrs)
        ###################################################
        # SEARCH OPTIONS UNDER
        ###################################################
        #search for parameter for a specific date or url file
        if self.param == None and (self.date != None or self.url != None):
            self.param = self.check_variable(self.file, self.search, self.url)

        #search for parameter for a all dates, only possible for not userdefined url.
        if self.date == None and self.param == None and self.url == None:
            self.param = self.check_variable_all(self.model, self.numbervar, self.search)
            if self.search:
                self.date = self.check_available_date(self.model, self.search)
            else:
                self.date = self.check_available_date(self.model)
        ##################################################
        ###################################################

        self.clean_all()

    def check_files(self, date, model, param, mbrs, url=None):
        """
        Returns a dataframe containing file name and file url
        """
        logging.info("--> check_files() <---\n")

        base_url = ""

        if self.url != None:
            #if a url file is given. we dont need to look for possible files.
            df = pd.DataFrame(data=list([re.search(f'[^/]*nc$', self.url).group(0)]), columns=["File"])
            df["url"] = self.url
        else:
            date = str(date); YYYY = date[0:4]; MM = date[4:6]; DD = date[6:8]; HH = date[8:10]
            # find out where to look for data
            archive_url = "latest" if self.use_latest else "archive"
            if model.lower()=="meps":
                base_url = "https://thredds.met.no/thredds/catalog/meps25eps{0}/".format(archive_url)   #info about date, years and filname of our model
                base_urlfile = "https://thredds.met.no/thredds/dodsC/meps25eps{0}/".format(archive_url) #info about variables in each file
            elif model.lower() == "aromearctic":
                base_url = "https://thredds.met.no/thredds/catalog/aromearctic{0}/".format(archive_url)
                base_urlfile = "https://thredds.met.no/thredds/dodsC/aromearctic{0}/".format(archive_url)
            else:
                SomeError(ValueError, f"Cannot recognise the model")

            #Find what files exist at that date by scraping thredd web
            page = requests.get(base_url + "catalog.html") if self.use_latest else requests.get(base_url + YYYY+"/"+MM+"/"+DD+ "/catalog.html")
            if page.status_code != 200:  SomeError(ConnectionError, f"Error locating site, is the date correct?")
            soup = BeautifulSoup(page.text, 'html.parser')
            rawfiles = soup.table.find_all("a")
            ff =[str(i.text) for i in rawfiles]
            ff= pd.DataFrame( data = list(filter(re.compile(f'.*{YYYY}{MM}{DD}T{HH}Z.nc').match, ff )), columns=["File"])
            drop_files = ["_vc_", "thunder", "_kf_", "_ppalgs_", "_pp_", "t2myr", "wbkz", "vtk","_preop_"]
            df = ff.copy()[~ff["File"].str.contains('|'.join(drop_files))] #(drop_files)])
            df.reset_index(inplace=True, drop=True)
            df["url"] = base_urlfile + df['File'] if self.use_latest else f"{base_urlfile}/{YYYY}/{MM}/{DD}/" + df['File']
            del ff
            del rawfiles
            soup.decompose()
            page.close()
            gc.collect()
        return df

    def check_file_info(self, df, param, mbrs):
        """
        Returns a dataframe containing info about file. used in get_data
        """
        logging.info("--> check_file_info() <---\n")

        # df["var"] = None
        # df["dim"] = None
        df["mbr_bool"] = False
        df["ml_bool"] = False
        df['p_levels'] = object()  # df['p_levels'].astype(object)
        df['m_levels'] = object()  # df['m_levels'].astype(object)
        df['mbr_ens'] = object()  # df['mbr_ens'].astype(object)
        df['h_levels'] = object()  # df['h_levels'].astype(object)

        i=0
        while i<len(df):
            #For every file in dataframe, read metadata.
            dataset = Dataset(df["url"][i]) #metadata

            # Make a dataframe containing dimention variables (time, pressure, etc)
            dn = dataset.dimensions.keys()
            ds = [dataset.dimensions[d].size for d in dn]
            valued = np.full(np.shape(ds), np.nan)
            dimdic = dict(zip(dn,ds))
            dimframe = pd.DataFrame(list(zip(ds, valued, valued)),index = dn, columns=["shape","value", "unit"])

            #seperate dimentions into pressure, hybrid, height and ensmembers.
            pressure_dim = list(filter(re.compile(f'press*').match, dimframe.index))
            hybrid_dim = list(filter(re.compile(f'.*hybrid*').match, dimframe.index))
            height_dim = list(filter(re.compile(f'.*height*').match, dimframe.index))
            memb_dim = list(filter(re.compile(f'.*member*').match, dimframe.index))

            #Initiali dict that will be filled with dimention info for every parameter.
            d_p = {}
            d_ml = {}
            d_hl = {}
            d_memb = {}

            #find info for all dimentional variables.
            for dim_tmp in dimframe.iterrows():
                if dimframe["shape"][dim_tmp[0]] < 100: #as long as dim is small enough get the value
                    prminfo = dataset.variables[dim_tmp[0]]
                    dim_value=[int(x) for x in prminfo[:]]
                    dimframe.loc[dim_tmp[0],"value"] = ",".join(str(int(x)) for x in dim_value)
                    #Check if dimention is a  pressure/hybrid/height or member dimention category
                    if dim_tmp[0] in pressure_dim:
                        print("INF")
                        print(dim_tmp[0])
                        tt = [int(x) for x in dataset.variables[dim_tmp[0]][:]] if dim_tmp[0] in dimdic and dimdic[
                            dim_tmp[0]] >= 1 else None
                        d_p.update({dim_tmp[0]: tt})
                    if dim_tmp[0] in hybrid_dim:
                        tt = [int(x) for x in dataset.variables[dim_tmp[0]][:]] if dim_tmp[0] in dimdic and dimdic[
                            dim_tmp[0]] >= 1 else None
                        d_ml.update({dim_tmp[0]: tt})
                    if dim_tmp[0] in height_dim:
                        tt = [int(x) for x in dataset.variables[dim_tmp[0]][:]] if dim_tmp[0] in dimdic and dimdic[
                                dim_tmp[0]] >= 1 else None
                        d_hl.update({dim_tmp[0]: tt})
                    if dim_tmp[0] in memb_dim:
                        tt = [int(x) for x in dataset.variables[dim_tmp[0]][:]] if dim_tmp[0] in dimdic and dimdic[
                                dim_tmp[0]] >= 1 else None
                        d_memb.update({dim_tmp[0]: tt})

                    try:
                        dimframe.loc[dim_tmp[0], "unit"] = prminfo.getncattr("units")
                    except:
                        pass

            #print(d_p)
            df.at[i, "p_levels"] = d_p
            df.at[i, "m_levels"] = d_ml
            df.at[i, "h_levels"] = d_hl
            df.at[i, "mbr_ens"] = d_memb
            #Go through all variables
            dv_shape = [dataset.variables[d].shape for d in dataset.variables.keys()]    #save var shape
            dv_dim = [dataset.variables[d].dimensions for d in dataset.variables.keys()] #save var dimensions / what it depends on
            varlist = list(zip(dv_shape,dv_dim))

            varframe = pd.DataFrame(varlist, index = dataset.variables.keys() ,columns=["shape", "dim"])
            df.loc[i,"var"] = [varframe.to_dict(orient='index')]
            df.loc[i,"dim"] = [dimframe.to_dict(orient='index')]
            i+=1
            dataset.close()

        #while has ended
        print("HOLa df")
        print(df)
        file_withparam = filter_param( df.copy(), param)
        print("HOLa file_withparam")
        print(file_withparam)
        print(mbrs)
        print(self.p_level)
        print(self.m_level)
        file_corrtype = filter_type( df.copy(), mbrs, self.p_level, self.m_level)
        print("HOLa fole_corrrtyope")
        print(file_corrtype)
        file = file_withparam[file_withparam.File.isin(file_corrtype.File)]

        file.reset_index(inplace=True, drop = True)

        file = filter_step(file,self.maxstep)


        if len(file) ==0 and len(df) !=0:#SomeError(ValueError, f'Type not found: c
            SomeError( ValueError,  f"Not able to find file at {self.date} for model {self.model} for these parameters. Available files are: \n {df}")
        elif len(file) ==0 and len(df) ==0:
            dt_requested = self.date
            dt_now = pd.to_datetime('now')  # or now or today
            dt_requested = pd.to_datetime(dt_requested, format='%Y%m%d%H')
            delta_time = dt_requested - dt_now
            folder= "archive" if self.use_latest==False else " the latest data"
            SomeError( ValueError,  f"Not able to find any file at {self.date} for model {self.model}. "
                                    f"The requested file is {abs(delta_time.days)} days from current date, and you have been looking in an {folder} folder. Maybe change use_latest = True if it is a recent date, and use_latest = False for archived dates")


        del file_withparam
        del file_corrtype
        gc.collect()

        return file

    def check_available_date(self, model, search = None):
        logging.info("--> check_available_date() <---\n")

        df = pd.read_csv(f"{package_path}/data/{model}_filesandvar.csv")
        dfc = df.copy()  # df['base_name'] = [re.sub(r'_[0-9]*T[0-9]*Z.nc','', str(x)) for x in df['File']]
        drop_files = ["_vc_", "thunder", "_kf_", "_ppalgs_", "_pp_", "t2myr", "wbkz", "vtk","_preop_"]
        df_base = pd.DataFrame([re.sub(r'_[0-9]*T[0-9]*Z.nc', '', str(x)) for x in df['File']], columns=["base_name"])
        dfc["base_name"] = df_base["base_name"]

        dfc = dfc[~dfc["base_name"].str.contains('|'.join(drop_files))]  # (drop_files)])
        if search:
            dfc = dfc[dfc["var"].str.contains(search)==True]
        dfc.reset_index(inplace=True, drop=True)
        #df_base = dfc['var'].str.replace(" ", "").str.split(",")  # , expand = True)
        dateti = dfc[["Date","Hour"]].copy()
        dateti.drop_duplicates(keep='first', inplace=True)
        dateti.reset_index(inplace=True, drop=True)

        return dateti

    def check_variable(self, file, search, url):
        #url not supported yet in var search
        logging.info("--> check_variable() <---\n")

        var_dict = file.at[0, "var"]
        param = []
        for n in range(0,len(file)):
            filename =  file.at[n, "File"]
            var = file.at[n, "var"].keys()
            param.append( pd.DataFrame([x for x in var], columns=[filename]))

        param = pd.concat(param, axis = 1, join = "outer", sort=True)
        if search:
            param = param[param.apply(lambda x: x.str.contains(search))]#.any(axis=1)]
            param = param.dropna(how='all')
        return param.to_string()

    def check_variable_all(self, model, numbervar, search ):
        logging.info("--> check_variable_all <---\n")

        df = pd.read_csv(f"{package_path}/data/{model}_filesandvar.csv")
        dfc = df.copy()  # df['base_name'] = [re.sub(r'_[0-9]*T[0-9]*Z.nc','', str(x)) for x in df['File']]
        drop_files = ["_vc_", "thunder", "_kf_", "_ppalgs_", "_pp_", "t2myr", "wbkz", "vtk","_preop_"]
        df_base = pd.DataFrame([re.sub(r'_[0-9]*T[0-9]*Z.nc', '', str(x)) for x in df['File']], columns=["base_name"])
        dfc["base_name"] = df_base["base_name"]
        dfc = dfc[~dfc["base_name"].str.contains('|'.join(drop_files))]  # (drop_files)])
        df_base = dfc['var'].str.replace(" ", "").str.split(",")  # , expand = True)
        flattened = [val for sublist in df_base[:] for val in sublist]
        if search:
            flattened = [s for s in flattened if str(search) in s]
            count = Counter(flattened).most_common(len(flattened))
            param = pd.DataFrame(count, columns = ["param", "used"])["param"]
        else:
            count = Counter(flattened).most_common(numbervar)
            param = pd.DataFrame(count, columns = ["param", "used"])["param"]

        return param.to_string()

    def check_filecontainingvar(self, model, numbervar, search ):
        logging.info("--> check_filecontainingvar <---\n")

        #NOT IN USE
        #Nice to have a list of the files containing that var, but that means scraping the web too often.
        #Maybe add on file. scraping only new dates...DID It! Just need to update this function to find file containing: Then another function saying at what date.

        # Todo: update R scripts to only add new lines in filevar info
        df = pd.read_csv(f"bin/{model}_filesandvar.csv")
        dfc = df.copy()
        drop_files = ["_vc_", "thunder", "_kf_", "_ppalgs_", "_pp_", "t2myr", "wbkz", "vtk","_preop_"]
        df_base = pd.DataFrame([re.sub(r'_[0-9]*T[0-9]*Z.nc', '', str(x)) for x in df['File']], columns=["base_name"])
        dfc["base_name"] = df_base["base_name"]
        dfc = dfc[~dfc["base_name"].str.contains('|'.join(drop_files))]  # (drop_files)])

        df_base = dfc['var'].str.replace(" ", "").str.split(",")  # , expand = True)
        search = "wind"
        #param = df_base[df_base.apply(lambda x: x.str.contains(search)).any(axis=1)]
        test = ["heipadeg", "du", "hei du"]

        flattened = [val for sublist in df_base[:] for val in sublist]

    #OTHER FUNC
    def clean_all(self):
        del self.model
        del self.url
        del self.mbrs
        del self.numbervar
        del self.search
        del self.p_level
        del self.m_level
        del self.maxstep
        del self.use_latest
        gc.collect()
    def check_web_connection(self, url=None):
        url_test = "https://thredds.met.no/thredds/catalog/meps25epsarchive/catalog.html"
        try:
            webcheck = requests.head(url_test, timeout=5)
        except requests.exceptions.Timeout as e:
            print(e)
            print("There might be problems with the server; check out https://status.met.no")
        except:
            print("internet problems?")
            exit(1)

        if webcheck.status_code != 200:  # SomeError(ValueError, f'Type not found: c
            SomeError(ConnectionError,
                      f"Website {url} is down with {webcheck}; . Wait until it is up again. Check https://status.met.no")

        webcheck.close()
        gc.collect()

