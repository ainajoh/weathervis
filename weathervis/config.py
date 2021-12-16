import platform
import os
import time
import numpy as np
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
import os.path as path
import pandas as pd
global OUTPUTPATH
global LOGFILEPATH
#LOGFILEPATH =  dname + "/../output/wv_get_data.log"
OUTPUTPATH = dname + "/../output/"

def make_data_uptodate():
    filepath=f"{dname}/data/"
    filenames= ["MEPS_filesandvar.csv", "AromeArctic_filesandvar.csv"]

    for file_nn in filenames:
        try:
            file_time = path.getmtime(filepath+ file_nn)
        except:
            Question = input(f"#################################################################################################\n"
                            f"##   You are missing model metadata for {filepath+ file_nn}.                                         \n"
                            f"##   This requires some time to generate the first time. Not having it might resolve in errors   \n"
                            f"##   You  might have to run; conda install -c r r-rvest and conda install -c r r-tidyverse        \n"
                            f"##   DO YOU WANT TO UPDATE IT NOW? ----->yes/no  ")

            if Question == ("yes") or Question == ("y"):
                print("Uppdating now....") #/Users/ainajoh/anaconda3/envs/r_env/bin/rscript

                #input_rscript= "MEPS" if file == "MEPS_filesandvar.csv" else "AromeArctic"
                #os.system(f'rscript ../util/scrap4filevariable.R {input_rscript}')

            elif Question == ("no") or Question == ("n"):
                print("NO updes")

        form_time= ((time.time() - file_time)) /(24)
        if int(form_time) > 90: #update metainfo every 3rd month
            Question = input(f"#################################################################################################\n"
                         f"## It is 3 month since model metadata was updated for {filepath + file_nn}                         \n"
                         f"## *** You  might have to run; conda install -c r r-rvest and conda install -c r r-tidyverse *** \n"
                         f"##   DO YOU WANT TO UPDATE IT NOW? ----->yes/no  ")

            if Question == ("yes") or Question == ("y"):
                print(filepath+ file_nn)
                df = pd.read_csv(filepath+ file_nn)
                print(df)
                last_date = df.loc[len(df)-1,"Date"]
                print(last_date)
                input_rscript = "MEPS" if file_nn == "MEPS_filesandvar.csv" else "AromeArctic"
                os.system(f'rscript ../util/scrap4filevariable.R {input_rscript} {last_date}')
            elif Question == ("no") or Question == ("n"):
                print("NO updates")

def setup_directory_config(OUTPUTPATH):

    if not os.path.exists(OUTPUTPATH):
        os.makedirs(OUTPUTPATH)
        print("Directory ", OUTPUTPATH, " Created ")
    else:
        print("Directory ", OUTPUTPATH, " already exists")

    return OUTPUTPATH

def cyclone():
    import importlib
    import sys
    from subprocess import call
    #module load Python/3.7.0-foss-2018b
    #source / Data / gfi / users / local / share / virtualenv / dynpie3 / bin / activate

    cyclone_conf = dname + "/data/config/config_cyclone.sh"
    call(f"source {cyclone_conf}", shell=True)
    MODULE_PATH = "/shared/apps/Python/3.7.0-foss-2018b/lib/python3.7/site-packages/netCDF4/__init__.py"
    MODULE_NAME = "netCDF4"
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    #OUTPUTPATH = dname+"/../../../../../output/weathervis/"

    username= os.environ.get('USER')
    OUTPUTPATH = "/Data/gfi/projects/isomet/projects/ISLAS/weathervis/output/"+username+"/"
    #OUTPUTPATH = setup_directory_config(OUTPUTPATH)
    USER_OUTPUTPATH = "/Data/gfi/projects/isomet/projects/ISLAS/weathervis/output/"+username+"/"

    package_path = os.path.dirname(__file__)
    # Nice logging info saved to aditional file

    pathhome = os.environ["HOME"]
    filelog = f"{pathhome}/wv_get_data.log"
    LOGFILEPATH = filelog
    return OUTPUTPATH

def islas_server():
    import importlib
    import sys
    from subprocess import call
    cyclone_conf = dname + "/data/config/config_islas_server.sh"
    call(f"source {cyclone_conf}", shell=True)
    OUTPUTPATH = dname+"/../../../../output/weathervis/"
    OUTPUTPATH = setup_directory_config(OUTPUTPATH)
    print(OUTPUTPATH)
    return OUTPUTPATH


if "cyclone.hpc.uib.no" in platform.node():
    print("detected cyclone")
    OUTPUTPATH = cyclone()
elif "islas-forecast.novalocal" in platform.node():
    print("detect islas-forecast.novalocal")
    OUTPUTPATH = islas_server()
else:
    print("local host detected")


