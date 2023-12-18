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
import traceback
from netCDF4 import Dataset
from copy import deepcopy


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
    #getting the uniq   ue flattened version of the total parameter that was available and that was not.
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
def retrievenow(our_choice,model,step, date,fileobj,m_level,p_level, domain_name=None, num_point=1, domain_lonlat=None,bad_param=[],bad_param_sfx=[],point_name=None, point_lonlat=None, use_latest=True,delta_index=None):
    print("################ retrievenow in checkget_data_handler.py #############################")

    fixed_var = ["ap", "b", "ap2", "b2", "pressure", "hybrid", "hybrid2","hybrid0"]  # this should be gotten from get_data

    ourfilename = our_choice.file[0]
    ourfileobj = fileobj[fileobj["File"].isin([ourfilename])]

    ourfileobj.reset_index(inplace=True, drop=True)
    print(num_point)
    data_domain = domain_input_handler(dt=date, model=model, domain_name=domain_name,
                 domain_lonlat=domain_lonlat, file =ourfileobj,point_name=point_name,
                 point_lonlat=point_lonlat, use_latest=use_latest,delta_index=delta_index, num_point=num_point)#

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
            print(pm)
            if pm in fixed_var:
                ap_prev = len(getattr(dmet, pm)) if pm in dmet.param else 0
                ap_next = len(getattr(dmet_next, pm)) if pm in dmet_next.param else 0
                if ap_next < ap_prev:  # if next is bigger dont overwrite with old one
                    continue
            setattr(dmet, pm, getattr(dmet_next, pm))
            #***************
            #if pm=="SIC":
            #    print("SIC here ")
            #    print(getattr(dmet.dim, pm))
            #    print(getattr(dmet_next.dim, pm))
            #    exit(1)
            #***************
            try:
                setattr(dmet.dim, pm, getattr(dmet_next.dim, pm))
            except:
                pass

            try:
                setattr(dmet.units, pm, getattr(dmet_next.units, pm))
                #setattr(dmet.idx, pm, getattr(dmet_next.idx, pm))
            except:
                pass


        param_set = set(dmet.param)
        param_set.update(dmet_next.param)
        param_set = list(param_set)
        dmet.param = param_set
        #setattr(dmet.param, dmet_next, pm))
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
def checkget_data_handler(all_param, date=None, save=False, read_from_saved=False, model=None, num_point=1,step=[0], p_level= None, m_level=None, mbrs=None, domain_name=None, domain_lonlat=None, point_name=None,point_lonlat=None,use_latest=False,delta_index=None, url=None,first_run=True):
    print("################ checkget_data_handler in checkget_data_handler.py #############################")
    #step = [step] if type(step) == int else step #isinstance(<var>, int)
    step = step.tolist() if type(step).__module__ == np.__name__ else step
    step = [step] if not isinstance(step, list) else step #
    date=[date] if type(date) != list and first_run else date
    point_lonlat = point_lonlat.tolist() if type(point_lonlat).__module__ == np.__name__ else point_lonlat

    print(date)
    print(step)
    print(point_lonlat)

    if read_from_saved:
        dmet = read_data(read_from_saved)
        return dmet, "", ""
    
    if (point_name or point_lonlat) and first_run:
        dmet,data_domain,bad_param = points_handler(all_param=all_param, date=date, save=save, 
                                                    read_from_saved=read_from_saved, model=model,
                                                    num_point=num_point,step=step, p_level= p_level, 
                                                    m_level=m_level, mbrs=mbrs, domain_name=domain_name,
                                                    domain_lonlat=domain_lonlat, point_name=point_name,
                                                    point_lonlat=point_lonlat,use_latest=use_latest,
                                                    delta_index=delta_index, url= url)
        if save: 
            save_data(dmet,data_domain,bad_param, save)  

        return dmet,data_domain,bad_param

    if type(date) is list:
        print("date list")
        dmet,data_domain,bad_param = time_handler( all_param=all_param, date=date, save=save, 
                                                    read_from_saved=read_from_saved, model=model,
                                                    num_point=num_point,step=step, p_level= p_level, 
                                                    m_level=m_level, mbrs=mbrs, domain_name=domain_name,
                                                    domain_lonlat=domain_lonlat, point_name=point_name,
                                                    point_lonlat=point_lonlat,use_latest=use_latest,
                                                    delta_index=delta_index, url= url)
        
        #if save: 
        #    save_data(dmet,data_domain,bad_param, save)   
        # 
        if not (point_name or point_lonlat): 
            if save: 
                save_data(dmet,data_domain,bad_param, save)
        return dmet,data_domain,bad_param

    #exit(1)

    if url != None:
        print("AINA rmv")
        fileobj = check_data(url=url, model=model, date=date, step=step, use_latest=use_latest,p_level=p_level, m_level=m_level).file
        data_domain = domain_input_handler(file = fileobj, url=url, dt=date, model=model, domain_name=domain_name, domain_lonlat=domain_lonlat,
                                           point_name=point_name, point_lonlat=point_lonlat, delta_index=delta_index, num_point=num_point)

        dmet = get_data(file = fileobj, url=url, model=model, param=all_param, step=step, date=date, m_level=m_level,
                         p_level=p_level, data_domain=data_domain, use_latest=use_latest)
        dmet.retrieve()
        bad_param=None
        #if save: 
        #    save_data(dmet,data_domain,bad_param)
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
                                                     delta_index=delta_index, num_point=num_point)
            break
        except:
            print("Oops!", sys.exc_info()[0], "occurred. See info under:")

            print(traceback.format_exc())
            print("Next entry.")
            print(" ")
    #if save: 
    #    save_data(dmet,data_domain,bad_param, save)
    return dmet,data_domain,bad_param

def make_dimdict(dmet):
    pass

def save_data(dmet,data_domain,bad_param, save="Buffer.nc"):
    print("in save_data")
    #print(dmet.param)#print(dir(dmet.dim))#print(vars(dmet.attr))
    file =save
    ncid = Dataset(file, 'w')

    #print(vars(dmet).keys())
    print(dmet.lonlat)
    for key, val in vars(dmet).items():
        if key in dmet.param:
            continue
        print(key)
        print(val)
        if type(val) ==bool:
            val=int(val)
        print(type(val))
        if val is None: #type(val) is None:
            val = -999999
        try:
            setattr(ncid, key, val )
        except:
            pass
    
    for param_nom in dmet.param:
            print(param_nom)
            expression_data = f"dmet.{param_nom}"
            expression_data_unit = f"dmet.units.{param_nom}"
            expression_data_dim = f"dmet.dim.{param_nom}"
            data = getattr(dmet, param_nom) #eval(expression_data)
            try:
                dims = eval(expression_data_dim)
            except:
                dims=[]
            try:
                units = eval(expression_data_unit)
            except:
                units=None
            excisting_dim = ncid.dimensions.keys() #dimensions.values():
            for _dim in dims: # gows through dimension parameter names
                if _dim not in excisting_dim:
                    print(_dim) #point
                    print(getattr(dmet, _dim)) 
                    print(len(getattr(dmet, _dim))) #2
                    print(np.shape(getattr(dmet, _dim)))
                    ncid.createDimension(_dim, len(getattr(dmet, _dim)))
                    #vid_dim = ncid.createVariable(_dim, 'f4',(_dim),zlib=True)

            print(dims[:]) #('point',)
            vid = ncid.createVariable(param_nom, 'f4',(dims[:]),zlib=True)
            vid.units = units if units else " "
            vid.dim = dims if dims else " "
            for _dim in dims:
                try:
                    setattr(vid.dim, _dim,  getattr(dmet, _dim)) 
                except:
                    pass
            #vid.description = dmet[param_nom]['description']
            #exit(1)
            print("bf data")
            try:
                vid[:] = data
            except:
                pass
            #if param_nom=="SIC":
            #    print("SIC here ")
            #    print(dims)
            #    exit(1)
    #exit(1)
    ncid.close()
    gc.collect()

def read_data(read_from_saved="Buffer.nc"):
    print("in read data")
    file=read_from_saved
    data = Dataset(file,"a")

    print(data)
    #exit(1)
    dmet = dummyobject()
    dmet.units=dummyobject()
    dmet.dim=dummyobject()    
    for k,v in vars(data).items():
        setattr( dmet, k, v )
    for k,v in data.variables.items():
        setattr( dmet, k, data.variables[k][:])
        setattr( dmet.units, k, v.units)
        setattr( dmet.dim, k, v.dim)
    data.close()
    gc.collect()
    #fileobj = check_data(url=url, model=model, date=date, step=step, use_latest=use_latest,p_level=p_level, m_level=m_level).file
    #data_domain = domain_input_handler(file = fileobj, url=url, dt=date, model=model, domain_name=domain_name, domain_lonlat=domain_lonlat,
    #                                      point_name=point_name, point_lonlat=point_lonlat, delta_index=delta_index, num_point=num_point)

    print(dir(dmet))
    print(dir(dmet.__getattribute__))
    #data_domain = domain(dt, dmet.model, file=file, use_latest=dmet.use_latest,delta_index=dmet.delta_index, url=url, num_point=num_point)
    return dmet

class dummyobject(object):pass


def conicide_time_with_location():
    pass

def points_handler(**args): #if point_lonlat = many (later if point_name is many)
    #for index, row in obs_data.iloc[:num_p].iterrows()
    print("in points_handler")
    print(args["point_lonlat"])
    print(np.shape(args["point_lonlat"]))

    if args["point_lonlat"] !=None and np.shape(args["point_lonlat"]) == (2,):# or np.shape(point_lonlat) == (2,1)):
        args["point_lonlat"] = np.array(args["point_lonlat"]).reshape(1,2).tolist()

    points = len(args["point_name"]) if args["point_name"] !=None else len(args["point_lonlat"])
    
    all_points = args["point_lonlat"] if args["point_lonlat"] else args["point_name"]
    initial_point_lonlat = args["point_lonlat"]
    print("in points_handler")
    dmet_old = False
    num_locrun=0
    
    for p in range(0,points):
        num_locrun+=1
        point_lonlat = point_name2point_lonlat(args["point_name"][p]) if args["point_name"] else initial_point_lonlat[p]
    
        args["point_lonlat"] = point_lonlat
        
        print(args["point_lonlat"])
        dmet,data_domain,bad_param = time_handler(**args)
        setattr( dmet, "point", all_points) #str(point_lonlat)

        if dmet_old: #if we have an older retrieval then do: 
            print("dmet_old = true")
            
            for prm in dmet.param: #for every parameter 
                print(prm)
                
                prev_orig_shape = list(np.shape(getattr(dmet, prm)))
                try:
                    dims = getattr(dmet.dim, prm)
                except:
                    dims=[]
                    continue
                print(dims)
                if "x" in dims or "y" in dims:
                    prev_orig_shape[-1] =prev_orig_shape[-1]* num_locrun
                    joint = np.append( getattr(dmet_old, prm), getattr(dmet, prm),axis=-1)
                    setattr( dmet, prm,  np.array( joint )) 
                    prev_dim = getattr(dmet.dim, prm)

                    import re
                    regex = re.compile(r'^x.?$|^y.?$')
                    dim_new = [i for i in prev_dim if not regex.match(i)]
                    dim_new.append("point")
                    setattr( dmet.dim, prm, tuple(dim_new ))                    


                    fill_all = False
                    if fill_all:
                        orig_shape=prev_orig_shape.copy() 
                        orig_shape[-1] =prev_orig_shape[-1]* num_locrun
                        if len(orig_shape)>1:
                            orig_shape[-2] =prev_orig_shape[-2]* num_locrun
                        #Indent out of if statement if u want more dim
                        #orig_shape = np.append(int(num_modelrun),np.array(orig_shape, dtype=np.int)) #add more dimentions
                        print(prev_orig_shape) #[1, 65, 1, 1]
                        print(orig_shape)     #[1, 65, 2, 2]

                        orig_shape = tuple(orig_shape)
                        old_shape = list(np.shape(getattr(dmet_old, prm)))   ##[1, 65, 1, 1]

                        numdim=1
                        numdim_act=1
                        numdim_act2=1
                        for di in range(0,len(orig_shape)):
                            numdim = numdim*orig_shape[di] #shape we want
                            numdim_act= numdim_act*prev_orig_shape[di] #one fill
                            numdim_act2 = numdim_act2*old_shape[di]    #other fill
            
                        #numdim_act = 0 if numdim_act==1 else numdim_act
                        fillnum = numdim - (numdim_act + numdim_act2) #shape we want  - shape we fill it
                        setattr( dmet, prm,  np.array( np.append( np.array([getattr(dmet_old, prm)]), np.array([getattr(dmet, prm)]) )) )
                        fill_array = np.full([int(fillnum )], np.nan)
                        if len(orig_shape)>1: setattr( dmet, prm,  np.array( np.append( fill_array,         np.array([getattr(dmet, prm)]) )) ) 
                        setattr( dmet, prm, getattr(dmet, prm).reshape(orig_shape[:] ))
                        print("af reshape")
                        print(np.shape(getattr(dmet, prm)))
                    #exit(1)

        dmet_old = deepcopy(dmet)

    #print("here")
    #exit(1)

    return dmet_old,data_domain,bad_param

def time_handler(**args):

    print("in timehandler")
    print(args["point_lonlat"])

    #exit(1)
    date_list = args["date"]
    dmet_old = False
    args["first_run"]=False
    num_modelrun =0
    for dt in date_list:
        num_modelrun +=1
        args["date"]=dt
        print("loop in timehandler")
        print(dt)
        dmet, data_domain, bad_param = checkget_data_handler(**args)
        if dmet_old: #if we have an older retrieval then do: 
            
            for prm in dmet.param: #for every parameter 
                print(prm)
                orig_shape = list(np.shape(getattr(dmet, prm)))
                print(orig_shape)
                try:
                    dims = getattr(dmet.dim, prm)
                except:
                    dims=[]
                    continue
                print(dims)
                if "time" in dims:
                    orig_shape[0] =orig_shape[0]* num_modelrun
                    #Indent out of if statement if u want more dim
                    #orig_shape = np.append(int(num_modelrun),np.array(orig_shape, dtype=np.int)) #add more dimentions
                    orig_shape = tuple(orig_shape)
                    print(orig_shape)
                    print("bf set")
                    joint=np.append( getattr(dmet_old, prm), getattr(dmet, prm), axis=0)
                    print(np.shape(joint))
                    setattr( dmet, prm,  np.array( joint ))
                    print("af set")
                    #setattr(dmet, prm, getattr(dmet, prm).reshape(orig_shape[:] ))
                    print("af reshape")
                    print(np.shape(getattr(dmet, prm)))

        dmet_old = deepcopy(dmet)
            
    return dmet,data_domain,bad_param

    #if datetime == a list. 
    #depending on your forecast retrieval request
    pass

if __name__ == "__main__": #todo add more for test functionality
    args = default_arguments()