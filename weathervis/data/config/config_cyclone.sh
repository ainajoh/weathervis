#!/bin/bash

#Needed for module load to work
get_module="source /etc/profile.d/z00_lmod.sh"
echo "$get_module"
$get_module
#Load some required libraries
py3="module load Python/3.7.0-foss-2018b"
dynlib3="source /Data/gfi/users/local/share/virtualenv/dynpie3-2021a/bin/activate"
proj="module load PROJ/5.0.0-foss-2018b" #for cartopy to work.
geos="module load GEOS/3.6.2-foss-2018b-Python-3.7.0"
wpath="/Data/gfi/isomet/projects/ISLAS/weathervis/"
wpath2="/Data/gfi/isomet/projects/ISLAS_aina/tools/githubclones/islas/scripts/flexpart_arome/"
ana="module load Anaconda3/5.3.0-foss-2018b"
cond="conda activate verdur"

#weathervispath="export PYTHONPATH=$PYTHONPATH:$wpath:$wpath2"
#weathervispath="export PYTHONPATH=$wpath:$wpath2"

#echo $ana
#$ana
#wait
echo "$py3"
$py3
wait
echo "$dynlib3"
$dynlib3
wait
echo "$proj"
$proj
wait
#echo $geos
#$geos
#wait
#echo $cond
#$cond
#wait
#conda list
#echo "$weathervispath"
#$weathervispath

