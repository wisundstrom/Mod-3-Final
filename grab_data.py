import requests
import pandas as pd
import json

#This is our data pre-cleaning script
#First we submit our soda api request to get the data, with our token in the header
head = {'token': 'hEodzVRdK0MFpd9EjlMszoCEg'}
r = requests.get('https://data.montgomerycountymd.gov/resource/4mse-ku6q.json?$select=commercial_license,hazmat,commercial_vehicle,violation_type,charge,driver_city,vehicle_type,make,model,year,color&$limit=50000', headers=head)

#We convert the response to json
datajson=r.json()
#and then convert that json as a dataframe
df = pd.DataFrame(datajson)
#finally we output a csv of the pre-cleaned data

df.to_csv('./data/dirty_data.csv')
