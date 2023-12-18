
import numpy as np
from netCDF4 import Dataset
import pandas as pd
from weathervis.calculation import *
from weathervis.check_data import *  # require netcdf4
import re
import requests
#Preset domain.

package_path = os.path.dirname(__file__)

if __name__ == "__main__":
    print("Run by itself")

def lonlat2idx(lonlat, lon, lat, num_point=1):
    #Todo: add like, when u have a domain outside region of data then return idx= Only the full data.
    # DOMAIN FOR SHOWING GRIDPOINT:: MANUALLY ADJUSTED
    print(lonlat)
    print(len(lonlat))
    #exit(1)
    if len(lonlat)>2:
        idx = np.where((lat > lonlat[2]-0.11) & (lat < lonlat[3]+0.09) & \
                       (lon >= lonlat[0]-0.32) & (lon <= lonlat[1]+0.28))
    else:
        idx = nearest_neighbour_idx(lonlat[0],lonlat[1],lon,lat, num_point)

    return idx

def idx2lonlat(idx, lon, lat, num_point=1):
    # DOMAIN FOR SHOWING GRIDPOINT:: MANUALLY ADJUSTED
    print("indise idx2lonlat")
    #lon = lon[idx[0].min(),idx[1].min(): idx[1].max()]
    #lat = lat[idx[0].min(),idx[1].min(): idx[1].max()]
    lon = lon[idx[0].min():idx[0].max()+1,idx[1].min(): idx[1].max()+1]
    lat = lat[idx[0].min():idx[0].max()+1,idx[1].min(): idx[1].max()+1]
    latlon = [lon.min(), lon.max(),lat.min(), lat.max() ]
    return latlon

def find_scale(lonlat):
    dim = ((float(lonlat[1]) - float(lonlat[0])) + (float(lonlat[3]) - float(lonlat[2])))  # / 31850.000000321685 #/ 107025.
    scale = int((1 / (dim / 124)) / 5 + 0.9) #empirical:
    if scale < 1:
        scale = 1
    elif scale > 12:
        scale = 12
    return scale

class domain():
    def __init__(self, date=None, model=None, point_lonlat=None, file=None, lonlat=None, idx=None,domain_name=None, 
    point_name=None, use_latest=True,delta_index=None, url=None, num_point=1):
        self.date = date
        self.model = model
        self.lonlat = lonlat
        self.idx = idx
        self.domain_name = domain_name
        self.point_name=point_name
        self.point_lonlat=point_lonlat
        self.use_latest = use_latest
        self.delta_index=delta_index
        self.scale = find_scale(self.lonlat) if self.lonlat else 1
        
        if (file is not None and isinstance(file, pd.DataFrame) ):  #type(file)==pd.core.frame.DataFrame):
            
            self.url = file.loc[0,'url']
        elif (file is not None):
            
            self.url = file.loc['url']
        elif (url is not None):
            
            self.url = url
        else:
            
            self.url = self.make_url_base()
        
        self.url = self.url + "?latitude,longitude"
        
        dataset = Dataset(self.url)
        
        self.lon = dataset.variables["longitude"][:]
        self.lat = dataset.variables["latitude"][:]
        dataset.close()  # self.lonlat = [0,30, 73, 82]  #

        if self.lonlat and not self.idx:
            self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)

        print(self.point_lonlat)
        
        if self.point_name != None and self.domain_name == None and self.point_lonlat==None:
            sites = pd.read_csv(f"{package_path}/data/sites.csv", sep=";", header=0, index_col=0)
            plon = float(sites.loc[self.point_name].lon)
            plat = float(sites.loc[self.point_name].lat)
            self.lonlat = [plon,plat]
            self.idx = lonlat2idx(self.lonlat, self.lon, self.lat, num_point)
            self.lonlat=idx2lonlat(self.idx,self.lon,self.lat)
            #if self.delta_index!=None:
            #    ii_max =  int(self.idx[0] + self.delta_index[0]/2)
            #    ii_min = int(self.idx[0] - self.delta_index[0]/2)
            #    jj_max = int(self.idx[1] + self.delta_index[1]/2)
            #    jj_min = int(self.idx[1] - self.delta_index[1]/2)
            #    ii=np.arange(ii_min,ii_max,1)
            #    jj=np.arange(jj_min,jj_max,1)
            #    self.idx = (ii,jj)

        if self.point_lonlat != None and self.domain_name == None:
            #plon = float(self.point_lonlat[0])
            #plat = float(self.point_lonlat[1])
            self.lonlat = self.point_lonlat
            self.idx = lonlat2idx(self.lonlat, self.lon, self.lat, num_point)
            if self.delta_index != None:
                ii_max = int(self.idx[0] + self.delta_index[0] / 2)
                ii_min = int(self.idx[0] - self.delta_index[0] / 2)
                jj_max = int(self.idx[1] + self.delta_index[1] / 2)
                jj_min = int(self.idx[1] - self.delta_index[1] / 2)
                ii = np.arange(ii_min, ii_max, 1)
                jj = np.arange(jj_min, jj_max, 1)
                self.idx = (ii, jj)






        #if self.idx:
        #    self.lonlat = idx2lonlat(self.idx, url)  # rough

        #url = ""#((YYYY==2018 and MM>=9) or (YYYY>2018)) and not (YYYY>=2020 and MM>=2 and DD>=4)
        #if self.model == "MEPS" and ( (int(YYYY)==2018 and int(MM)<9) or ( int(YYYY)<2018 ) ):
        #    url = f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{YYYY}/{MM}/{DD}/meps_mbr0_extracted_backup_2_5km_{YYYY}{MM}{DD}T{HH}Z.nc?latitude,longitude"
        #
        #elif self.model == "MEPS"
        #
        # and ( (int(YYYY)==2018 and int(MM)>=9) or (int(YYYY)>2018 )) and ((int(YYYY)==2020 and int(MM)<=2 and int(DD)<4)):
        #    url = f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{YYYY}/{MM}/{DD}/meps_mbr0_extracted_2_5km_{YYYY}{MM}{DD}T{HH}Z.nc?latitude,longitude"
        #else:
        #    url = f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{YYYY}/{MM}/{DD}/meps_det_2_5km_{YYYY}{MM}{DD}T{HH}Z.nc?latitude,longitude"
        #eval()
    def make_url_base(self): # Todo: Update for more accurate url. avoid if filename is given as input
        print("############### make_url_bas in domain.py#######################")
        date = str(self.date)
        YYYY = date[0:4]; MM = date[4:6]; DD = date[6:8] #HH = date[8:10]
        archive_url = "latest" if self.use_latest else f"archive/{YYYY}/{MM}/{DD}/"
        #exit(1)
        check=check_data()
        
        print(check.url)
    
        meta_df = check.filter_metadata(check.load_metadata(), model=self.model, use_latest=self.use_latest)
        
        base_url = meta_df[meta_df.modelinfotype]
        base_urlfile=meta_df[meta_df.source]
        base_url = base_url + "catalog.html" if self.use_latest else base_url + YYYY+"/"+MM+"/"+DD+ "/catalog.html"
        base_urlfile = base_urlfile if self.use_latest else base_urlfile + YYYY+"/"+MM+"/"+DD + "/"
        
        print(base_url)

        #if self.model.lower() == "aromearctic":
        #    base_url = "https://thredds.met.no/thredds/catalog/aromearctic{0}/".format(archive_url)
        #    base_urlfile = "https://thredds.met.no/thredds/dodsC/aromearctic{0}/".format(archive_url)
        #elif self.model.lower() == "meps":
        #    base_url = "https://thredds.met.no/thredds/catalog/meps25eps{0}/".format(archive_url)
        #    base_urlfile = "https://thredds.met.no/thredds/dodsC/meps25eps{0}/".format(archive_url)
        #else:
        #    base_url = self.url


        page = requests.get(base_url)
        soup = BeautifulSoup(page.text, 'html.parser')
        rawfiles = soup.table.find_all("a")
        
        ff = [str(i.text) for i in rawfiles]
        
        ff = pd.DataFrame(data=list(filter(re.compile(f'.*.nc$|.*.ncml$').match, ff)), columns=["File"])
        drop_files = ["_vc_", "thunder", "_kf_", "_ppalgs_", "_pp_", "t2myr", "wbkz", "vtk", "_preop_"]
        
        df = ff.copy()[~ff["File"].str.contains('|'.join(drop_files))]  # (drop_files)])
        df.reset_index(inplace=True, drop=True)
        
        df["url"] = base_urlfile + df['File']# if self.use_latest else f"{base_urlfile}/{YYYY}/{MM}/{DD}/" + df['File']
        del ff
        del rawfiles
        soup.decompose()
        page.close()
        gc.collect()
        print("eee1")
        print( df.url)
        print( df.url[0])

        
        url= df.url[0] #just random pick the first one
        return url



    def MEPS(self):
        self.lonlat = [-1, 60., 49., 72]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)
    def Finse(self):
        self.lonlat = [7.524026, 8.524026, 60, 61.5]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def South_Norway(self):
        self.domain_name = "South_Norway"
        self.lonlat = [4., 9.18, 58.01, 62.2]  # lonmin,lonmax,latmin,latmax,
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def West_Norway(self):
        # self.lonlat = [2., 12., 53., 64.]  # lonmin,lonmax,latmin,latmax,
        self.lonlat = [1.0, 12., 54.5, 64.]  # lonmin,lonmax,latmin,latmax,
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def AromeArctic(self):
        # self.lonlat = [-10,60,30,90] #lonmin,lonmax,latmin,latmax,
        self.lonlat = [-18.0, 80.0, 62.0, 88.0]  # [-30,90,10,91] #lonmin,lonmax,latmin,latmax,

        # url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)  # RIUGHNone#[0, -1, 0, -1]  # Index; y_min,y_max,x_min,x_max such that lat[y_min] = latmin
        self.scale = find_scale(self.lonlat)

    def CAO_fram(self):  # map
        url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"

        self.lonlat = [10, 10, 75, 80]  #
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)  # RIUGHNone#[0, -1, 0, -1]  # Index; y_min,y_max,x_min,x_max such that lat[y_min] = latmin
        self.scale = find_scale(self.lonlat)

    def Svalbard_z2(self):  # map
        url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"

        self.lonlat = [15, 23, 77, 82]  #
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)  # RIUGHNone#[0, -1, 0, -1]  # Index; y_min,y_max,x_min,x_max such that lat[y_min] = latmin
        self.scale = find_scale(self.lonlat)

    def Svalbard_z1(self):  # map
        self.lonlat = [4, 23, 76.3, 82]  #
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)  # RIUGHNone#[0, -1, 0, -1]  # Index; y_min,y_max,x_min,x_max such that lat[y_min] = latmin
        self.scale = find_scale(self.lonlat)

    def Svalbard(self):  # data
        # url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"
        self.domain_name = "Svalbard"
        self.lonlat = [-8, 30, 73, 82]  #
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)  # RIUGHNone#[0, -1, 0, -1]  # Index; y_min,y_max,x_min,x_max such that lat[y_min] = latmin
        self.scale = find_scale(self.lonlat)

    def North_Norway(self):  # data
        #url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"
        self.domain_name = "North_Norway"
        self.lonlat = [5, 20, 66.5, 76.2]  #
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)  # RIUGHNone#[0, -1, 0, -1]  # Index; y_min,y_max,x_min,x_max such that lat[y_min] = latmin
        self.scale = find_scale(self.lonlat)

    def KingsBay(self):  # bigger data
        url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"

        self.lonlat = [10, 13.3, 78.6, 79.3]
        self.idx = lonlat2idx(self.lonlat,  self.lon, self.lat)  # Rough
        self.scale = find_scale(self.lonlat)

    def KingsBay_Z0(self):  # map
        url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"

        self.lonlat = [11, 13., 78.73, 79.16]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)  # Rough
        self.scale = find_scale(self.lonlat)

    def Test(self):  # map
        url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"

        self.lonlat = [-10, -5, 82.0, 82.1]
        self.idx = lonlat2idx(self.lonlat, url)  # Rough
        self.scale = find_scale(self.lonlat)

    def KingsBay_Z1(self):  # smaller data
        url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"

        self.idx = np.array([[517, 517, 518, 518, 518, 518, 518, 519, 519, 519, 519, 519, 519, 520, 520, 520, 520, 520,
                              520, 520, 520, 520, 521, 521, 521, 521, 521, 521, 521, 521, 521, 522, 522, 522, 522, 522,
                              522, 522, 522, 522, 523, 523, 523, 523, 523, 523, 524, 524, 524, 524, 524, 525, 525, 525],
                             [183, 184, 182, 183, 184, 185, 186, 182, 183, 184, 185, 186, 187, 181, 182, 183, 184, 185,
                              186, 187, 188, 189, 182, 183, 184, 185, 186, 187, 188, 189, 190, 183, 184, 185, 186, 187,
                              188, 189, 190, 191, 185, 186, 187, 188, 189, 190, 186, 187, 188, 189, 190, 187, 188,
                              189]])  # y,x
        self.lonlat = idx2lonlat(self.idx, self.lon, self.lat)  # rough
        self.scale = find_scale(self.lonlat)

    def Andenes(self):
        # 16.120;69.310;10
        self.domain_name = "Andenes"
        self.lonlat = [15.8, 16.4, 69.2, 69.4]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def ALOMAR(self):
        # 16.120;69.310;10
        self.domain_name = "ALOMAR"
        self.lonlat = [15.8, 16.4, 69.2, 69.4]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def Andenes_area(self):
        self.domain_name = "Andenes_area"
        self.lonlat = [12.0, 19.5, 68.0, 70.6]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def Varlegenhuken(self):
        point_name = "Varlegenhuken"
        sites = pd.read_csv(f"{package_path}/data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.32)
        maxlon = float(plon + 0.28)
        minlat = float(plat - 0.11)
        maxlat = float(plat + 0.09)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def Longyearbyen(self):
        point_name = "Longyearbyen"
        sites = pd.read_csv(f"{package_path}/data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.32)
        maxlon = float(plon + 0.28)
        minlat = float(plat - 0.11)
        maxlat = float(plat + 0.09)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def Hopen(self):
        point_name = "Hopen"
        sites = pd.read_csv(f"{package_path}/data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.32)
        maxlon = float(plon + 0.28)
        minlat = float(plat - 0.11)
        maxlat = float(plat + 0.09)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def Bodo(self):
        point_name = "Bodo"
        sites = pd.read_csv(f"{package_path}/data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.32)
        maxlon = float(plon + 0.28)
        minlat = float(plat - 0.11)
        maxlat = float(plat + 0.09)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def Tromso(self):
        point_name = "Tromso"
        sites = pd.read_csv(f"{package_path}/data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.32)
        maxlon = float(plon + 0.28)
        minlat = float(plat - 0.11)
        maxlat = float(plat + 0.09)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def Bjornoya(self):
        point_name = "Bjornoya"
        sites = pd.read_csv(f"{package_path}/data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.32)
        maxlon = float(plon + 0.28)
        minlat = float(plat - 0.11)
        maxlat = float(plat + 0.09)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def NyAlesund(self):
        point_name = "NyAlesund"
        sites = pd.read_csv(f"{package_path}/data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.32)
        maxlon = float(plon + 0.28)
        minlat = float(plat - 0.11)
        maxlat = float(plat + 0.09)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def MetBergen(self):
        point_name = "MetBergen"
        #import os
        #abspath = os.path.abspath(__file__)
        #dname = os.path.dirname(abspath)
        sites = pd.read_csv(f"{package_path}/data/sites.csv", sep=";", header=0, index_col=0)

        #sites = pd.read_csv("./data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.32)
        maxlon = float(plon + 0.28)
        minlat = float(plat - 0.11)
        maxlat = float(plat + 0.09)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def Osteroy(self):
        point_name = "Osteroy"
        sites = pd.read_csv(f"{package_path}/data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 1.70)
        maxlon = float(plon + 1.10)
        minlat = float(plat - 0.80)
        maxlat = float(plat + 1.00)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def Olsnesnipa(self):  # PAraglidingstart
        point_name = "Olsnesnipa"
        sites = pd.read_csv(f"{package_path}/data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.22)
        maxlon = float(plon + 0.18)
        minlat = float(plat - 0.08)
        maxlat = float(plat + 0.05)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def JanMayen(self):  # PAraglidingstart
        point_name = "JanMayen"
        sites = pd.read_csv(f"{package_path}/data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.22)
        maxlon = float(plon + 0.18)
        minlat = float(plat - 0.08)
        maxlat = float(plat + 0.05)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def CAO(self):  # PAraglidingstart
        point_name = "JanMayen"
        sites = pd.read_csv(f"{package_path}/data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.22)
        maxlon = float(plon + 0.18)
        minlat = float(plat - 0.08)
        maxlat = float(plat + 0.05)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def NorwegianSea(self):  # PAraglidingstart
        point_name = "NorwegianSea"
        sites = pd.read_csv(f"{package_path}/data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.22)
        maxlon = float(plon + 0.18)
        minlat = float(plat - 0.08)
        maxlat = float(plat + 0.05)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def NorwegianSea_area(self):  # PAraglidingstart
        #url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"
        self.domain_name = "NorwegianSea_area"
        self.lonlat = [-7, 16, 69.0, 77.2]  #
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def GEOF322(self):  # PAraglidingstart
        point_name = "GEOF322"
        sites = pd.read_csv(f"{package_path}/data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.22)
        maxlon = float(plon + 0.18)
        minlat = float(plat - 0.08)
        maxlat = float(plat + 0.05)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def Iceland(self):
        #url = "https://thredds.met.no/thredds/dodsC/aromearcticlatest/arome_arctic_sfx_2_5km_latest.nc?latitude,longitude"
        self.domain_name = "Iceland"
        #self.lonlat = [12.0, 19.5, 68.0, 70.6]
        #self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        #self.lonlat = [-65, 20., 58., 85]
        self.lonlat = [-26., -8, 63., 67]

        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def pcmet1(self):
        point_name = "pcmet1"
        sites = pd.read_csv(f"{package_path}/data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.22)
        maxlon = float(plon + 0.18)
        minlat = float(plat - 0.08)
        maxlat = float(plat + 0.05)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def pcmet2(self):
        point_name = "pcmet2"
        sites = pd.read_csv(f"{package_path}/data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.22)
        maxlon = float(plon + 0.18)
        minlat = float(plat - 0.08)
        maxlat = float(plat + 0.05)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)

    def pcmet3(self):
        point_name = "pcmet3"
        sites = pd.read_csv(f"{package_path}/data/sites.csv", sep=";", header=0, index_col=0)
        plon = float(sites.loc[point_name].lon)
        plat = float(sites.loc[point_name].lat)
        minlon = float(plon - 0.22)
        maxlon = float(plon + 0.18)
        minlat = float(plat - 0.08)
        maxlat = float(plat + 0.05)
        self.lonlat = [minlon, maxlon, minlat, maxlat]
        self.idx = lonlat2idx(self.lonlat, self.lon, self.lat)
        self.scale = find_scale(self.lonlat)