""" This is our data pre-cleaning script. First we submit our soda api request
to get the data, with our token in the header, then we convert the json data to
 a csv and save it. """

import requests
import pandas as pd

HEAD = {'token': 'hEodzVRdK0MFpd9EjlMszoCEg'}

R = requests.get("""https://data.montgomerycountymd.gov/resource/4mse-ku6q.json
?$select=seq_id,commercial_license,hazmat,commercial_vehicle,violation_type,cha
rge,driver_city,vehicle_type,make,model,year,color&$limit=50000""",
                 headers=HEAD)

# We convert the response to json
DATAJSON = R.json()
# and then convert that json as a dataframe
DF = pd.DataFrame(DATAJSON)
# finally we output a csv of the pre-cleaned data

DF.to_csv('./data/dirty_data.csv')
