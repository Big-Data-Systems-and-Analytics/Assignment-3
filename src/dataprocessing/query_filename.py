

from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import os
import json
import sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

with open('premium-strata-340618-745287f8fd66.json') as source:
    info = json.load(source)

credentials = service_account.Credentials.from_service_account_info(info)
projectid = "premium-strata-340618"
client = bigquery.Client(credentials= credentials,project=projectid)



def query_filename(location,begintime,endtime):
  query = """ SELECT file_name FROM `premium-strata-340618.CATSTORM.CATSTORM` WHERE Begin_location = '{}' AND Begin_date_time = '{}' AND End_date_time = '{}' AND  file_name like 'vil%'""".format(location,begintime,endtime)


  query_job = client.query(query)
  result = query_job.result()
  for row in result:
    return row.file_name
  

def query_catalog(filename):
    query_catalog_job = client.query("""
    SELECT *
    FROM `premium-strata-340618.storm.catalog` 
    WHERE file_name = '{}'
      """.format(filename))
    df = query_catalog_job.to_dataframe()
    cwd = os.getcwd()
    path = cwd + "/data/CATALOG.csv"
    df.to_csv(path)




