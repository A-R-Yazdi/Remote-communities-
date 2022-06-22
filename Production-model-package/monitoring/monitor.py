import requests
import argparse
import subprocess

import os
# get the absolute path of this file: monitor.py
real_path = os.path.realpath(__file__)
print(real_path)
# get the root dir of this entire project.
dir_path = os.path.dirname(real_path)
root_path = os.path.dirname(dir_path)
print(root_path)
data_path=os.path.dirname(root_path)+"/data/df2_processed.csv"
print(data_path)


import sys  
# adding deploy dir to the system path
sys.path.insert(0, root_path+'/deploy')
from deploy import model_unvailable_msg

# query = {'lat':'45', 'lon':'180'}
# response = requests.get('localhost:8001/predict', params=query)
# print(response.json())

args = argparse.ArgumentParser()
args.add_argument("-fh", "--forecast_horizon", default=20, help="forecast horizon")
args = vars(args.parse_args())

fh = args["forecast_horizon"]
scoring_uri = "http://127.0.0.1:8001/predict?fh="+str(fh)
resp = requests.post(scoring_uri)
print(resp.text.strip('\"'))
# print(str({"17539":0.7373629947076457}))
# print('{"17539":0.7373629947076457}')
# print(str(resp.text)=='{"17539":0.7373629947076457}')

condition = resp.text.strip('\"')=='{"17539":0.7373629947076457}' or resp.text.strip('\"')==model_unvailable_msg

if condition:

    print("Removing old model, all requests will be suspend during the process!")
    subprocess.run(["rm "+root_path+"/train/lgbm_forecaster.pickle"], shell=True)
    print("Retrain the model...")
    subprocess.run(["python "+root_path+"/train/train.py -d "+data_path+" -mr "+root_path+"/train/ -fr "+root_path+"/train/"], shell=True)
else:
    print("model does not need to be updated")
