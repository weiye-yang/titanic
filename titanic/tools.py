import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns

#takes as parameter an array. returns number of nan elements
def countNan(myarray):
    return np.count_nonzero(pd.isnull(myarray))

#deletes Nan values from an array
def delNan(myarray):
    return myarray[np.logical_not(pd.isnull(myarray))]

#returns one of the most common elements of a numpy array
def mostCommon(myarray):
    unique, counts = np.unique(myarray, return_counts = True)
    dic = dict(zip(unique, counts))
    return max(dic, key = dic.get)

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

if __name__ == "__main__":
    print(mostCommon(["a", "b", "a"]))