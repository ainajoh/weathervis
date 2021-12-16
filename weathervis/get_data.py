########################################################################
# File name: get_data.py
# This file is part of: weathervis
########################################################################
from netCDF4 import Dataset
import numpy as np
import logging
import pandas as pd
import os
from weathervis.check_data import *  # require netcdf4
from weathervis.domain import *  # require netcdf4
import re
import pkgutil
import ast

"""
###################################################################
This module gets the data defined by the user 
------------------------------------------------------------------------------
Usage:
------
data =  get_data(model, date, param, file, step, data_domain=None, p_level = None, m_level = None, mbrs=None)
Returns:
------
data Object with properties
"""

package_path = os.path.dirname(__file__)
# Nice logging info saved to aditional file

pathhome= os.environ["HOME"]
filelog=f"{pathhome}/wv_get_data.log"

logging.basicConfig(filename=filelog, level = logging.INFO, format = '%(levelname)s : %(message)s')
if os.path.getsize(filelog) > 100 * 1024:
    os.remove(filelog)

#use_latest = True

def SomeError( exception = Exception, message = "Something did not go well" ):
    # Nice error messeges.
    logging.error(exception(message))
    #source: https://softwareengineering.stackexchange.com/questions/222586/how-should-you-cleanly-restrict-object-property-types-and-values-in-python
    if isinstance( exception.args, tuple ):
        raise exception
    else:
        raise exception(message)

# Valid options of certain variables
available_models = ["aromearctic", "meps"] #ECMWF later
# functions for filtering out unvalid uptions
check_if_thredds_is_down = lambda value:value if requests.head(value) != 200 else SomeError(ConnectionError, f"Website {value} is down;. Wait until it is up again")
filter_function_for_type= lambda value: value if value in levtype else SomeError(ValueError, f'Type not found: choices:{levtype}')
filter_function_for_models = lambda value: value if value.lower() in available_models else SomeError(ValueError, f'Model not found: choices:{model}')
filter_function_for_date = lambda value: value \
    if ( len(value) == 10 ) and ( int(value[0:4]) in range(2000,2030) ) and ( int(value[4:6]) in range(0,13) ) \
    and ( int(value[6:8]) in range(1,32)) and ( int(value[9:10]) in range(0,25)) \
    else SomeError(ValueError, f'Modelrun wrong: Either; "latest or date on the form: YYYYMMDDHH')
filter_function_for_mbrs=lambda value, file: value if max(value) < file.dim["ensemble_member"]["shape"] else SomeError(ValueError, f'Member input outside range of model')
filter_function_for_step=lambda value, file: value if np.max(value) < file.dim["time"]["shape"] else SomeError(ValueError, f' step input outside range of model')
#filter_function_for_p_level=lambda value, file: value if set(value).issubset(set(file["p_levels"])) else SomeError(ValueError, f' p_level input outside range of model')
filter_function_for_m_level=lambda value, file: value if np.max(value) < file.dim["hybrid"]["shape"] else SomeError(ValueError, f' m_level input outside range of model')
filter_function_for_param=lambda value, file: value if set(value).issubset(set(file["var"].keys())) else SomeError(ValueError, f' param input not possible for this file')
filter_function_for_file=lambda value: value if value is not None else SomeError(ValueError, f' File has to be given (see check_data(()')

#filter_function_for_domain=lambda value: value if value in np.array(dir(domain))  else SomeError(ValueError, f'Domain name not found')
#Domain filter not needed as it should be handled in domain itself
class get_data():

    def __init__(self, model=None, date=None, param=None, file=None, step=None, data_domain=None, p_level = None, m_level = None, h_level = None, mbrs=None, url=None, use_latest=False,delta_index=None):
        """
        Parameters - Type - Info - Example
        ----------
        model: - String - The weather model we want data from            - Example: model = "MEPS"
        date:  - String - date and time of modelrun in format YYYYMMDDHH - Example: date = "2020012000"
        param: - List of Strings - Parameters we want from model         - Example: param = ["wind_10m"]
        file:  - Panda Series of strings and dictionaries -  Returned from chech_data.py -
        step   - int or list of ints - Forecast time steps -
        data_domain - String - Domain name as defined in the domain.py file
        p_level - int or list of ints - Pressure levels
        m_level -  int or list of ints - model levels
        mbrs     - int or list of ints - ensemble members
        url      - url of where we can find file on thredds
        """
        print("################ __init__ in get_data.py #############################")

        # Initialising -- NB! The order matters ###
        self.model = model
        self.mbrs_bool = False if mbrs == None else True
        mbrs = [0] if mbrs == None else mbrs
        self.mbrs = list(mbrs) if type(mbrs)!=list else mbrs
        self.date = str(date)
        self.step = step if step != None else 0

        #if type(self.step) == list and type( self.step[0] ) == str:
        #    if len( self.step ) == 1:
        #        if ":" in step[0]:
        #            ss = step[0].split(":")
        #            ss = [int(x.strip()) for x in ss]
        #            if len(ss) == 3:
        #                self.step = list(np.arange(ss[0], ss[1], ss[2]))
        ##            else:
        #                self.step = list(np.arange(ss[0], ss[1], 1))
        #        elif "," in step[0]:
        #            ss = step[0].split(",")
        #            self.step = [int(x.strip()) for x in ss]
        #        else:
        #            self.step = [int(x.strip()) for x in step]
        #    else:
        #        self.step = [int(x.strip()) for x in step]

        self.h_level = h_level
        self.p_level = p_level if type(p_level)==list or p_level==None else list(p_level)
        self.m_level = m_level
        self.param = np.unique(param).tolist()
        self.data_domain = data_domain
        self.delta_index=delta_index
        #If file comes in as a dataframe with possibly multiple rows, choose one and make it a Series
        self.file = file.loc[0] if type(file) == pd.core.frame.DataFrame else file
        #If no datadomain is set, choose idx to span the entire modeldomain
        if data_domain != None:
            self.idx = data_domain.idx
        else:
            self.idx = ( np.array([0, self.file["var"]["y"]["shape"][0]-1]), np.array([0,self.file["var"]["x"]["shape"][0]-1]) )

        self.lonlat = data_domain.lonlat if data_domain else None
        #if no member is wanted initially, then we exclude an aditional dimension caused by this with mbrs_bool later

        #If No member is wanted, we take the control (mbr=0)
        self.url = url
        self.units = self.dummyobject()
        self.FillValue = self.dummyobject()
        self.use_latest=use_latest
        self.indexidct = None #updated later Contains the indexes for the dimentions we want, see adjust_user_url()

        #Check and filter for valid settings. If any of these result in a error, this script stops
        check_if_thredds_is_down("https://thredds.met.no/thredds/catalog/meps25epsarchive/catalog.html")



        if self.url is None:
            filter_function_for_file(self.file)

            if self.model: filter_function_for_models(self.model)
            filter_function_for_mbrs(np.array(self.mbrs), self.file) if self.mbrs != None and self.mbrs_bool and self.file.mbr_bool else None

            if self.date: filter_function_for_date(self.date)

            filter_function_for_step(self.step,self.file)

            #filter_function_for_p_level(np.array(self.p_level),self.file) if self.p_level != None and self.file.p_levels != None else None
            #filter_function_for_m_level(np.array(self.m_level),self.file) if self.m_level != None and self.file.ml_bool else None

            filter_function_for_param(self.param, self.file) #this is kind of checked in check_data.py already.


        #Make a url depending on preferences if no url is defined already.
        self.url = self.make_url() if self.url == None else self.url #should call an error here for else statement

        self.url = self.adjust_user_url()



    def adjust_user_url(self):
        """
        Adjust url to retrieve only specific values
        :return:
        """
        print("################ adjust_user_url in get_data.py #############################")

        jindx = self.idx[0]
        iindx = self.idx[1]
        # Sets up the userdefined range of value in thredds format [start:step:stop]

        step = f"[{np.min(self.step)}:1:{np.max(self.step)}]" #0 or no
        y = f"[{jindx.min()}:1:{jindx.max()}]"
        x = f"[{iindx.min()}:1:{iindx.max()}]"
        pl_idx = f"[0:1:0]"
        m_level = f"[0:1:0]"
        mbrs = f"[0:1:0]"
        non = f"[0:1:0]"

        indexidct = {"time": step, "y": y, "x": x}
        fixed_var = np.array(
            ["latitude", "longitude", "forecast_reference_time", "projection_lambert", "surface_air_pressure", "ap", "b", "ap2", "b2","ap0", "b0","ap1", "b1"])
        # keep only the fixed variables that are actually available in the file.
        fixed_var = fixed_var[np.isin(fixed_var, list(self.file["var"].keys()))]
        # update global variable to include fixed var
        self.param = np.array(list(set(np.append(self.param, fixed_var)) ) )# Contains absolutely all variables we want
        file = self.file.copy()
        param = self.param.copy()
        logging.info(file)
        url = f"{self.url}?"
        list_p = list(param)
        i=0# loop that updates the url to include each parameter with its dimensions
        while i < len(list_p):
            prm = param[i]
            url += f"{prm}"  # example:  url =url+x_wind_pl
            dimlist = list( file["var"][prm]["dim"] )  # List of the variables the param depends on ('time', 'pressure', 'ensemble_member', 'y', 'x')

            #########################
            # Find different dimention related to either pressure, model levels, height levels or ens members.
            # Then adjust retrieve url to only include upper and lower limit of these variables.
            #########################
            # Find different dimention related to either pressure, model levels, height levels or ens members.
            pressure_dim = list(filter(re.compile(f'press*').match, dimlist))
            model_dim = list(filter(re.compile(f'.*hybrid*').match, dimlist))
            height_dim = list(filter(re.compile(f'.*height*').match, dimlist))
            other_dim = list(filter(re.compile(f'.*top_of_atmosphere*').match, dimlist))
            ens_mbr_dim = list(filter(re.compile(f'.*ensemble*').match, dimlist))
            x_dim = list(filter(re.compile(f'x.*?(\d+)/*').match, dimlist))
            y_dim = list(filter(re.compile(f'y.*?(\d+)/*').match, dimlist))
            if x_dim:
                indexidct[x_dim[0]] = indexidct["x"]
            if y_dim:
                indexidct[y_dim[0]] = indexidct["y"]

            if pressure_dim:
                self.p_level = self.file["p_levels"][pressure_dim[0]] if self.p_level is None else self.p_level
                is_in_any = np.sum((np.array(self.file["p_levels"][pressure_dim[0]])[:, None] == np.array(self.p_level)[None, :])[:])
                if is_in_any == 0:
                    SomeError(ValueError, f'Please provide valid pressure levels for parameter {prm}. \n'
                                          f'Options are {self.file["p_levels"][pressure_dim[0]]}, but you requested { self.p_level}')

                idx = np.where(np.array(self.file["p_levels"][pressure_dim[0]])[:, None] == np.array(self.p_level)[None, :])[0]
                pl_idx = f"[{np.min(idx)}:1:{np.max(idx)}]"
                indexidct[pressure_dim[0]] = pl_idx#"[0:1:0]"

            #if other_dim:
            #    print("other_dim")
                #self.p_level = self.file["p_levels"][pressure_dim[0]] if self.p_level is None else self.p_level
                #is_in_any = np.sum((np.array(self.file["p_levels"][pressure_dim[0]])[:, None] == np.array(self.p_level)[None, :])[:])
                #if is_in_any == 0:
                #    SomeError(ValueError, f'Please provide valid pressure levels for parameter {prm}. \n'
                #                          f'Options are {self.file["p_levels"][pressure_dim[0]]}, but you requested { self.p_level}')

                #idx = np.where(np.array(self.file["p_levels"][pressure_dim[0]])[:, None] == np.array(self.p_level)[None, :])[0]
                #pl_idx = f"[{np.min(idx)}:1:{np.max(idx)}]"
                #indexidct[other_dim[0]] = pl_idx


            extra = []
            if model_dim:
                for m in model_dim:
                    dinfo = Dataset(self.url)
                    hyp = dinfo.variables[m]
                    terms = hyp.getncattr("formula_terms")
                    terms = re.sub(r'(\s\S*?)\s', r'\1, ', terms.strip())
                    terms = re.sub(r'(\w+)', r'"\1"', terms)
                    terms = ast.literal_eval("{" + terms + "}")
                    extra = np.append(extra, list(terms.values()))

                m_level = self.m_level
                lev_num = np.arange(0,len(self.file["m_levels"][model_dim[0]]))

                self.m_level = lev_num if self.m_level is None else self.m_level
                is_in_any = np.sum((np.array(lev_num)[:, None] == np.array(self.m_level)[None, :])[:])
                if is_in_any == 0:
                    SomeError(ValueError, f'Please provide valid model levels for parameter {prm}. \n'
                                          f'Options are {self.file["m_levels"][model_dim[0]]}, but you requested {self.m_level}')
                idx = np.where(np.array(lev_num)[:, None] == np.array(self.m_level)[None, :])[0]

                ml_idx = f"[{np.min(idx)}:1:{np.max(idx)}]"
                indexidct[model_dim[0]] = ml_idx

                #self.param.append(self.param, extra)
                #for e in extra:
                #    print("eeeeeeeeeeeeeeee")
                #    print(e)
                #    print("press" in e)
                #    #pressure_dim_extra = list(filter(re.compile(f'*press*').match, [e]))
                #    if "press" in e:
                #        #param = np.append(param, e)
                #        list_p.extend(e)
                #
                #    else:
                #        if e=="p0":
                #            indexidct[e] = ""
                #        else:
                #            indexidct[e] = ml_idx


                self.m_level =  m_level



            if height_dim:
                h_level = self.h_level
                self.h_level = self.file["h_levels"][height_dim[0]] if self.h_level is None else self.h_level
                is_in_any = np.sum(
                    (np.array(self.file["h_levels"][height_dim[0]])[:, None] == np.array(self.h_level)[None, :])[:])

                if is_in_any == 0:
                    SomeError(ValueError, f'Please provide valid height levels for parameter {prm}. \n'
                                          f'Options are {self.file["h_levels"][height_dim[0]]}, but you requested {self.h_level}')
                idx = \
                np.where(np.array(self.file["h_levels"][height_dim[0]])[:, None] == np.array(self.h_level)[None, :])[0]
                hl_idx = f"[{np.min(idx)}:1:{np.max(idx)}]"
                indexidct[height_dim[0]] = hl_idx
                self.h_level = h_level

            if ens_mbr_dim:
                self.mbrs = self.file["mbr_ens"][ens_mbr_dim[0]] if self.mbrs is None else self.mbrs
                is_in_any = np.sum(
                    (np.array(self.file["mbr_ens"][ens_mbr_dim[0]])[:, None] == np.array(self.mbrs)[None, :])[:])
                if is_in_any == 0:
                    SomeError(ValueError, f'Please provide valid ensemble members for parameter {prm}. \n'
                                          f'Options are {self.file["mbr_ens"][ens_mbr_dim[0]]}, but you requested {self.mbrs}')
                idx = \
                    np.where(np.array(self.file["mbr_ens"][ens_mbr_dim[0]])[:, None] == np.array(self.mbrs)[None, :])[
                        0]
                mbr_idx = f"[{np.min(idx)}:1:{np.max(idx)}]"
                indexidct[ens_mbr_dim[0]] = mbr_idx
            #Convert the dimentional variables to numbers
            newlist = [indexidct[i] for i in dimlist]  # convert dependent variable name to our set values. E.g: time = step = [0:1:0]

            #exit(1)
            startsub = ''.join(
                newlist) + ","  # example: ('time', 'pressure','ensemble_member','y','x') = [0:1:0][0:1:1][0:1:10][0:1:798][0:1:978]

            aditional = list(file["var"][prm]["dim"])# + list(set(extra))
            for dimen in np.setdiff1d( aditional, self.param ):
                # includes the dim parameters like, pressure, hybrid, height as long as we havent already gone through them
                self.param = np.append(self.param,
                                       dimen)  # update global param with the var name so that we do not go through it multiple time.
                startsub += dimen
                startsub += indexidct[dimen] + ","
            url += startsub
            i += 1


        url = url.rstrip(",")  # if url ends with , it creates error so remove.
        logging.info(url)
        self.indexidct = indexidct
        return url  # returns the url that will be set to global url.

    def make_url(self):
        """
        Runs if no url is inserted
        Makes the OPENDAP url for the user specified model and parameters in a specific domain and time
        Returns
        -------
        url for thredds
        """

        ###############################################################################
        file = self.file.copy()
        param = self.param.copy()
        logging.info(file)
        date = str(self.date)
        YYYY = date[0:4]
        MM = date[4:6]
        DD = date[6:8]
        HH = date[8:10]

        if self.use_latest==False and self.model.lower() =="aromearctic":
              url = f"https://thredds.met.no/thredds/dodsC/aromearcticarchive/{YYYY}/{MM}/{DD}/{file.loc['File']}"
        elif self.use_latest==True and self.model.lower()  =="aromearctic":
              url = f"https://thredds.met.no/thredds/dodsC/aromearcticlatest/{file.loc['File']}"
        elif self.use_latest==False and self.model.lower()  =="meps":
              url = f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{YYYY}/{MM}/{DD}/{file.loc['File']}"
        elif self.use_latest==True and self.model.lower()  =="meps":
              url = f"https://thredds.met.no/thredds/dodsC/meps25epslatest/{file.loc['File']}"
        else:
              url = self.url

        logging.info(url)
        return url #returns the url that will be set to global url.

    def thredds(self, url, file):
        """
        Retrieves the data from thredds and set it as attributes to the global object.
        Parameters
        ----------
        url:
        file
        Returns
        -------
        """
        logging.info("-------> start retrieve from thredds")
        print("################ thredds in get_data.py #############################")
        print(url)
        dataset = Dataset(url)
        self.indexidct = dict.fromkeys(self.indexidct, ":")  #reset the index dictionary
        for k in dataset.__dict__.keys(): #info of the file
            ss = f"{k}"
            self.__dict__[ss] = dataset.__dict__[k]
        logging.info("-------> Getting variable: ")
        iteration =-1
        for prm in self.param:
            iteration += 1
            logging.info(prm)
            dimlist = list(file["var"][prm]["dim"])  # List of the variables the param depends on ('time', 'pressure', 'ensemble_member', 'y', 'x')
            pressure_dim = list(filter(re.compile(f'press*').match, dimlist))
            model_dim = list(filter(re.compile(f'.*hybrid*').match, dimlist))
            #height_dim = list(filter(re.compile(f'.*height*').match, dimlist))
            #mbrs_dim = list(filter(re.compile(f'.*ensemble*').match, dimlist))

            startsub = ":" #retrieve everything if startsub = :
            t_arr = np.arange(0,int(self.file["dim"]["time"]["shape"]))
            idx = np.where(t_arr[:, None] == np.array(self.step)[None, :])[0]
            idx = ",".join([str(i) for i in idx - idx[0]])
            idx = '[{:}]'.format(idx)
            self.indexidct["time"] = ''.join(str(idx))
            newlist1 = [self.indexidct[i] for i in dimlist]
            startsub = ','.join(newlist1) if newlist1 else ":"
            if pressure_dim:
                if self.p_level is None:
                    pass
                idx = np.where(np.array(self.file["p_levels"][pressure_dim[0]])[:,None]==np.array(self.p_level)[None,:])[0]
                idx = ",".join([str(i) for i in idx -idx[0]])
                idx = '[{:}]'.format(idx)
                self.indexidct[pressure_dim[0]] = ''.join(str(idx))
                newlist1 = [self.indexidct[i] for i in dimlist]  # convert dependent variable name to our set values. E.g: time = step = [0:1:0]
                startsub = ','.join(newlist1)  # example: ('time', 'pressure','ensemble_member','y','x') = [0:1:0][0:1:1][0:1:10][0:1:798][0:1:978]
            elif model_dim:
                lev_num = np.arange(0,len(self.file["m_levels"][model_dim[0]]))
                if self.m_level is None:
                    self.m_level = lev_num
                idx = \
                np.where(np.array(lev_num)[:, None] == np.array(self.m_level)[None, :])[0]
                idx = ",".join([str(i) for i in idx - idx[0]])
                idx = '[{:}]'.format(idx)
                self.indexidct[model_dim[0]] = ''.join(str(idx))
                newlist1 = [self.indexidct[i] for i in
                            dimlist]  # convert dependent variable name to our set values. E.g: time = step = [0:1:0]
                startsub = ','.join(newlist1)  # ex

            if "units" in dataset.variables[prm].__dict__.keys():
                self.units.__dict__[prm] = dataset.variables[prm].__dict__["units"]
            if "_FillValue" in dataset.variables[prm].__dict__.keys():
                self.FillValue.__dict__[prm] = int(dataset.variables[prm].__dict__["_FillValue"])
            else:
                self.FillValue.__dict__[prm] = np.nan
            if prm == "projection_lambert":
                for k in dataset.variables[prm].__dict__.keys():
                    ss = f"{k}_{prm}"
                    self.__dict__[ss] = dataset.variables[prm].__dict__[k]

            varvar = f"dataset.variables['{prm}'][{startsub}]" ##
            varvar = eval(varvar)
            dimlist = np.array(list(file["var"][prm]["dim"]))  # ('time', 'pressure', 'ensemble_member', 'y', 'x')

            if not self.mbrs_bool and any(np.isin(dimlist, "ensemble_member")):#"ensemble_member" in dimlist:
                indxmember = np.where(dimlist == "ensemble_member")[0][0]
                varvar = dataset.variables[prm][:].squeeze(axis=indxmember)

            self.__dict__[prm] = varvar
        dataset.close()
        iteration += 1

    def retrieve(self):
        #self.url = self.make_url()
        self.thredds(self.url, self.file)

    class dummyobject(object):pass
