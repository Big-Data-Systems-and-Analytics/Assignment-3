# Load a part of the test dataset
#! /usr/bin/env python3
import numpy as np
import os
from dataprocessing.nowcast_reader import read_data
from main import model
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy

norm = {'scale':47.54,'shift':33.44}
hmf_colors = np.array( [
    [82,82,82], 
    [252,141,89],
    [255,255,191],
    [145,191,219]
])/255

def visualize_result(models,x_test,y_test,idx,ax,labels):
    fs=10

    for i in range(1,13,3):
        xt = x_test[idx,:,:,i]*norm['scale']+norm['shift']
        ax[(i-1)//3][0].imshow(xt)
    ax[0][0].set_title('Inputs',fontsize=fs)
    
 
    x_test = x_test[idx:idx+1]
    y_test = y_test[idx:idx+1]*norm['scale']+norm['shift']
    y_preds=[]
    for i,m in enumerate(models):
        yp = m.predict(x_test)
        if isinstance(yp,(list,)):
            yp=yp[0]
        y_preds.append(yp*norm['scale']+norm['shift'])
    
    
    
    for k,m in enumerate(models):
        for i in range(0,12,3):
            ax[i//3][2+2*k].imshow(y_preds[k][0,:,:,i])
            

        ax[0][2+2*k].set_title(labels[k],fontsize=fs)
        
        
    for j in range(len(ax)):
        for i in range(len(ax[j])):
            ax[j][i].xaxis.set_ticks([])
            ax[j][i].yaxis.set_ticks([])
    for i in range(4):
        ax[i][1].set_visible(False)
    for i in range(4):
        ax[i][3].set_visible(False)
    ax[0][0].set_ylabel('-45 Minutes')
    ax[1][0].set_ylabel('-30 Minutes')
    ax[2][0].set_ylabel('-15 Minutes')
    ax[3][0].set_ylabel('  0 Minutes')
    ax[0][2].set_ylabel('+15 Minutes')
    ax[1][2].set_ylabel('+30 Minutes')
    ax[2][2].set_ylabel('+45 Minutes')
    ax[3][2].set_ylabel('+60 Minutes')
    

    plt.subplots_adjust(hspace=0.05, wspace=0.05)



def predict_data():
    x_test,y_test = read_data('data/interim/nowcast_testing.h5',end=50)
    yp = model.predict(x_test)
    y_preds=[]
    y_preds.append(yp*norm['scale']+norm['shift'])
    idx=25
    fig,ax = plt.subplots(4,4,figsize=(5,8), gridspec_kw={'width_ratios': [1,0.2,1,.2]})
    visualize_result([model],x_test,y_test,idx,ax,labels=['MSE Model Output'])
    plt.savefig("output.png")
    #plt.show()



    
