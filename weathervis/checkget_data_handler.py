#######################################################################
# File name: checkget_datahandler.py
# This file is part of: weathervis
########################################################################
"""
###################################################################
Combining check and get data to eliminate repeated codes for users.
Here we also combine what files we need to retrieve from most efficently inorder to get all the wanter parameters.
------------------------------------------------------------------------------
Usage:
------
dmet, data_domain, bad_param = checkget_data_handler(all_param=all_param, date=dt, model=model,
                                                     step=steps, domain_name=domain_name, use_latest=False,
                                                     m_level=[1,2,5], p_level=[1000,500])
Returns:
------
Object with properties
"""
#from memory_profiler import profile


from weathervis.config import *
from weathervis.domain import *
from weathervis.check_data import *
from weathervis.get_data import *
import itertools
import os
import pandas as pd
import sys
import numpy as np
import gc
from weathervis.utils import * #domain_input_handler

#from memory_profiler import profile

#@profile
def SomeError( exception = Exception, message = "Something did not go well" ):
    # Nice error messeges.
    logging.error(exception(message))
    #source: https://softwareengineering.stackexchange.com/questions/222586/how-should-you-cleanly-restrict-object-property-types-and-values-in-python
    if isinstance( exception.args, tuple ):
        raise exception
    else:
        raise exception(message)
#@profile
def find_best_combinationoffiles(all_param, fileobj, m_level=None, p_level=None):    #how many ways ca we split up all the possible balls between these kids?
    print("###################### find_best_combinationoffiles in checkget_data_handler.py##################################")
    m_level = 60 if m_level is None else max(m_level)
    filenames = []
    tot_param_we_want_that_are_available = []
    tot_param_we_want_that_areNOT_available = []
    for i in range(0, len(fileobj)):
        param_available = [*fileobj.loc[i].loc["var"].keys()]
        param_we_want_that_are_available = [x for x in param_available if x in all_param]
        param_we_want_that_areNOT_available = [x for x in all_param if x not in param_we_want_that_are_available]
        tot_param_we_want_that_areNOT_available += [param_we_want_that_areNOT_available]
        tot_param_we_want_that_are_available += [param_we_want_that_are_available]
        filenames += [fileobj.loc[i].loc["File"]]
    #getting the unique flattened version of the total parameter that was available and that was not.
    tot_unique_avalable = []
    for sublist in tot_param_we_want_that_are_available:
        for item in sublist:
            if item not in tot_unique_avalable:
                tot_unique_avalable.append(item)
    tot_NOTunique_avalable = []
    for sublist in tot_param_we_want_that_areNOT_available:
        for item in sublist:
            if item not in tot_NOTunique_avalable:
                tot_NOTunique_avalable.append(item)
    #Contains the parameters not found in any file.
    bad_param = [x for x in tot_NOTunique_avalable if x not in tot_unique_avalable]

    #UNCOMMENT IF YOU WANT IT TO STOP WHEN PARAM YOU WANT IS NOT FOUND AT ALL
    #if len(not_available_at_all) != 0:
    #    print(f"The requested parameters are not all available. Missing: {not_available_at_all}")
    #    raise ValueError #what if we set these variables to None such that no error eill occur with plotting, only blanks
    config_overrides_r = dict(zip(filenames, tot_param_we_want_that_are_available))

    def filer_param_by_modellevels(config_overrides_r,tot_param_we_want_that_are_available):
        print("################ filer_param_by_modellevels in checkget_data_handler.py #############################")

        for i in range(0,len(fileobj)):
            thisfileobj = fileobj.loc[i]
            varname = tot_param_we_want_that_are_available[i]
            if len(varname) == 0: #jump over file if no parameter needed in it
                continue
            var = pd.DataFrame.from_dict(thisfileobj.loc["var"], orient="index")
            varname = [varname] if type(varname) is not list else varname
            pandas_df = var.loc[varname]
            f = pandas_df[pandas_df.dim.astype(str).str.contains("hybrid")] #keep only ml variables.
            if len(f) != 0: #if var depends on hybrid.
                dimen = [f.apply(lambda row: dict(zip(row['dim'],row['shape'])), axis=1)][0]#.loc["dim"]
                dimofmodellevel = [dimen.apply(lambda row: [value for key, value in row.items() if 'hybrid' in key.lower()])][0]#.loc["dim
                removethese = dimofmodellevel[dimofmodellevel.apply(lambda row: row[0]<m_level)]#.loc["dim"]
                val = [*removethese.index]
                key = thisfileobj["File"] #arome_arctic_extracted_2_5km_20200221T00Z.nc
                config_overrides_r[key] = [x for x in config_overrides_r[key] if x not in val]
        return config_overrides_r #indent of this is impotant. resulted in error before when indent wrong

    config_overrides_r = filer_param_by_modellevels(config_overrides_r,tot_param_we_want_that_are_available)
    #flip it
    config_overrides = {}
    for key, value in config_overrides_r.items():
         for prm in value:
             config_overrides.setdefault(prm, []).append(key)

    if len(config_overrides) > 0:
        keys, values = zip(*config_overrides.items())
        possible_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)][:10]
        ppd = pd.DataFrame([], columns=["file", "keys", "len", "combo"])
        ppd.sort_values(by='len', inplace=True)

        iii = 0
        #This for loop takes time if it is many combinations, therefore reduces combinations.
        # bUT IT CAN BLE glitchy because I CAN NOT sort it based on how many files, but seems python might do this automatically..
        for combination in possible_combinations:
            filesincombination = [*combination.values()]
            uniquelistoffiles = list(set(filesincombination))
            ppd.at[iii, 'file'] = uniquelistoffiles
            ppd.at[iii, 'keys'] = iii
            ppd.at[iii, 'len'] = len(uniquelistoffiles)
            ppd.at[iii, 'combo'] = combination
            iii += 1
        ppd.sort_values(by='len', inplace=True)
        ppd.reset_index(inplace=True)
        our_choice = ppd.loc[0] #The best combination retrieving from least amount of files.
    else:
        ppd = pd.DataFrame([])
    return ppd,bad_param
#@profile
def retrievenow(our_choice,model,step, date,fileobj,m_level,p_level, domain_name=None, domain_lonlat=None,bad_param=[],bad_param_sfx=[],point_name=None, point_lonlat=None, use_latest=True,delta_index=None):
    print("################ retrievenow in checkget_data_handler.py #############################")

    fixed_var = ["ap", "b", "ap2", "b2", "pressure", "hybrid", "hybrid2","hybrid0"]  # this should be gotten from get_data

    ourfilename = our_choice.file[0]
    print("*****************333333333")
    print(ourfilename)
    ourfileobj = fileobj[fileobj["File"].isin([ourfilename])]

    ourfileobj.reset_index(inplace=True, drop=True)
    data_domain = domain_input_handler(dt=date, model=model, domain_name=domain_name, domain_lonlat=domain_lonlat, file =ourfileobj,point_name=point_name,point_lonlat=point_lonlat, use_latest=use_latest,delta_index=delta_index)#

    combo = our_choice.combo

    ourparam = [k for k, v in combo.items() if v == ourfilename]

    dmet = get_data(model=model, param=ourparam, file=ourfileobj, step=step, date=date, m_level=m_level, p_level=p_level, data_domain=data_domain, use_latest=use_latest)

    dmet.retrieve()

    for i in range(1,len(our_choice.file)):
        ourfilename = our_choice.file[i]
        ourfileobj = fileobj[fileobj["File"].isin([ourfilename])]
        ourfileobj.reset_index(inplace=True, drop=True)
        combo = our_choice.combo
        ourparam = [k for k, v in combo.items() if v == ourfilename]


        dmet_next = get_data(model=model, param=ourparam, file=ourfileobj, step=step, date=date, m_level=m_level,
                     p_level=p_level, data_domain=data_domain, use_latest=use_latest)
        dmet_next.retrieve()

        for pm in dmet_next.param:
            if pm in fixed_var:
                ap_prev = len(getattr(dmet, pm)) if pm in dmet.param else 0
                ap_next = len(getattr(dmet_next, pm)) if pm in dmet_next.param else 0
                if ap_next < ap_prev:  # if next is bigger dont overwrite with old one
                    continue
            setattr(dmet, pm, getattr(dmet_next, pm))
        del(dmet_next)

    for bparam in bad_param:
        setattr(dmet, bparam, None)

    good_sfx = np.setdiff1d(["SFX_"+b for b in bad_param],bad_param_sfx)
    for gprmsfx in good_sfx:
        gprm = gprmsfx.replace("SFX_","")
        setattr(dmet, gprm, getattr(dmet,gprmsfx))

    gc.collect()

    return dmet, data_domain,bad_param
#@profile
def checkget_data_handler(all_param, date=None,  model=None, step=[0], p_level= None, m_level=None, mbrs=None, domain_name=None, domain_lonlat=None, point_name=None,point_lonlat=None,use_latest=False,delta_index=None, url=None):
    print("################ checkget_data_handler in checkget_data_handler.py #############################")
    step = [step] if type(step) == int else step
    if url != None:
        fileobj = check_data(url=url, model=model, date=date, step=step, use_latest=use_latest).file
        data_domain = domain_input_handler(file = fileobj, url=url, dt=date, model=model, domain_name=domain_name, domain_lonlat=domain_lonlat,
                                           point_name=point_name, point_lonlat=point_lonlat, delta_index=delta_index)

        dmet = get_data(file = fileobj, url=url, model=model, param=all_param, step=step, date=date, m_level=m_level,
                         p_level=p_level, data_domain=data_domain, use_latest=use_latest)
        dmet.retrieve()
        bad_param=None
        return dmet, data_domain, bad_param

    date=str(date)
    fileobj = check_data(model=model, date=date, step=step, use_latest=use_latest).file
    all_choices, bad_param  = find_best_combinationoffiles(all_param=all_param, fileobj=fileobj,m_level=m_level,p_level=p_level)


    bad_param_sfx=[]
    if bad_param:
        new_bad = ["SFX_"+x for x in bad_param]
        all_param = all_param + new_bad
        all_choices, bad_param_sfx = find_best_combinationoffiles(all_param=all_param, fileobj=fileobj, m_level=m_level,
                                                              p_level=p_level)
    if len(all_choices)==0:
        SomeError(ValueError, f'No matches for your parameter found, try using the check_data search option')
    # RETRIEVE FROM THE BEST COMBINATIONS AND TOWARDS WORSE COMBINATION IF ANY ERROR

    for i in range(0, len(all_choices)):
        gc.collect()
        try:
            dmet, data_domain,bad_param = retrievenow(our_choice = all_choices.loc[i],model=model,step=step, date=date,fileobj=fileobj,
                                   m_level=m_level,p_level=p_level,domain_name=domain_name, domain_lonlat=domain_lonlat,
                                    bad_param = bad_param,bad_param_sfx = bad_param_sfx,point_name=point_name,point_lonlat=point_lonlat,use_latest=use_latest,
                                                     delta_index=delta_index)
            break
        except:
            print("Oops!", sys.exc_info()[0], "occurred.")
            print("Next entry.")
            print(" ")
    return dmet,data_domain,bad_param



if __name__ == "__main__": #todo add more for test functionality
    args = default_arguments()
#    import argparse
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--datetime", help="YYYYMMDDHH for modelrun", default=None, nargs="+")
#    parser.add_argument("--steps", default=[0, 10], nargs="+", type=int,
#                        help="forecast times example --steps 0 3 gives time 0 to 3")
#    parser.add_argument("--model", default="AromeArctic", help="MEPS or AromeArctic")
#    parser.add_argument("--domain_name", default=None, help="see domain.py")
#    parser.add_argument("--domain_lonlat", default=None, nargs="+", type=float, help="lonmin lonmax latmin latmax")
#    parser.add_argument("--param_all", default=None, nargs="+", type=str)
#    parser.add_argument("--point_name", default=None, help="see sites.csv")
#    parser.add_argument("--point_lonlat", default=None, nargs="+", type=float, help="lon lat")
#    parser.add_argument("--point_num", default=1, type=int)
#    parser.add_argument("--plot", default="all", help="Display legend")
#    parser.add_argument("--legend", default=False, help="Display legend")
#    parser.add_argument("--info", default=False, help="Display info")
#    args = parser.parse_args()

#    param_pl = []
#    param_ml = ["air_temperature_ml", "specific_humidity_ml"]
#    param_sfc = ["surface_air_pressure", "air_pressure_at_sea_level", "air_temperature_0m", "air_temperature_2m",
#                 "relative_humidity_2m", "x_wind_gust_10m", "y_wind_gust_10m", "x_wind_10m", "y_wind_10m",
#                 "specific_humidity_2m", "precipitation_amount_acc", "convective_cloud_area_fraction",
#                 "cloud_area_fraction", "high_type_cloud_area_fraction", "medium_type_cloud_area_fraction",
#                 "low_type_cloud_area_fraction", "rainfall_amount", "snowfall_amount", "graupelfall_amount",
#                 "land_area_fraction"]
#    param_sfc = ["specific_humidity_2m"]
#    all_param = param_sfc + param_ml + param_pl

#    fileobj = check_data(args.model, date=str(args.datetime[0]), step=args.steps, use_latest=use_latest).file
#    all_choices, bad_param = find_best_combinationoffiles(all_param, fileobj)

    #RETRIEVE FROM THE BEST COMBINATIONS AND TOWARDS WORSE COMBINATION IF ANY ERROR
#    for i in range(0, len(all_choices)):
#        try:
#            dmet = retrievenow(all_choices.loc[i],args.model, args.steps, str(args.datetime[0]))
#            break
#        except:
#            #del(dmet)
#            print("Oops!", sys.exc_info()[0], "occurred.")
#            print("Next entry.")
#            print(" ")
