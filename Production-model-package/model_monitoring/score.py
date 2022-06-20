import requests
import argparse

# query = {'lat':'45', 'lon':'180'}
# response = requests.get('localhost:8001/predict', params=query)
# print(response.json())

args = argparse.ArgumentParser()
args.add_argument("-fh", "--forecast_horizon", default=20, help="forecast horizon")
args = vars(args.parse_args())

fh = args["forecast_horizon"]
scoring_uri = "http://localhost:8001/predict?fh="+str(fh)
resp = requests.post(scoring_uri)
print(resp.text)