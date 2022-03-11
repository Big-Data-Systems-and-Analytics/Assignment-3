import os
import shutil
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from requestmodel import Request
import sys
import visualize
from dataprocessing.zipping import zipfiles
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
from dataprocessing.query_filename import query_filename
from dataprocessing.query_filename import query_catalog
from dataprocessing.nowcast_results import load_sevirfile
import tensorflow as tf

app = FastAPI()
""" tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2) """
model = tf.keras.models.load_model('model\mse_model.h5',compile=False,custom_objects={"tf": tf})
print(model)



#Index route
@app.get('/')
def index():
    return {'message':'Welcome to Nowcasting Model'}

@app.post('/predict')
def predict(data:Request): 

    data = data.dict()
    location = data['location']
    begintime = data['begintime']
    endtime = data['endtime']
    dirpath = "data"
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    else:
        print("directory doesn't exist")
    filename = query_filename(location,begintime,endtime)
    print(filename)
    load_sevirfile(filename)
    #query_catalog(filename)
    cmd = "python nowcast_datagen\make_nowcast_dataset.py --sevir_data data\sevir --sevir_catalog CATALOG.csv --output_location data\interim"
    os.system(cmd)
    visualize.predict_data()
    return zipfiles("output.png")

    
    

    
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#loc
#time date
#filename = query_filename(location,begintime,endtime) data\sevir/vil/2019/SEVIR_VIL_STORMEVENTS_2019_0701_1231.h5