import numpy as np
import matplotlib.pyplot as plt

def funkcja(x):
    y2=[]
    for i in range(0,len(x)):
        if x[i] <0:
            temp=np.sin(x[i])
            y2=np.append(y2,temp)
        else: 
            temp = np.sqrt(x[i])
            y2=np.append(y2,temp)
    return  y2
            
            
            