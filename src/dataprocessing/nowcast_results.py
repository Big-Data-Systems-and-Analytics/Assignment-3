import boto3
from botocore.handlers import disable_signing
import os
import sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
from dataprocessing.query_filename import query_filename
from config.rootdir import ROOT_DIR


def load_sevirfile(filename):
    resource = boto3.resource('s3')

    resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)

    bucket=resource.Bucket('sevir')

    year = filename.split('/')[1]
    vilfile = filename.split('/')[2].split(".", 2)
    vilfile = vilfile[0]+"."+vilfile[1]
    path =  "data/sevir/vil/{}".format(year) 
    vilpath = path+"/{}".format(vilfile)
    datapath = os.makedirs(path)
    #_dir = os.makedirs(os.path.join('data/sevir/vil', '%s' % year)) 
    bucket.download_file('data/'+filename, vilpath)
    os.makedirs("data/interim/")

