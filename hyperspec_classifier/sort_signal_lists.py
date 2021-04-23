import os
from numpy import loadtxt
import matplotlib.pyplot as plt
import math
import numpy as np

def parse_hyperspec_lists(dir, search_string, num_segs, plot = False):

    dictOfSegs = {}

    list_of_folders = [i for i in os.listdir(dir) if search_string in i]
    for count, signal in enumerate(list_of_folders):
        with open(dir + signal, 'r+') as rf:
            lines = rf.readlines()
            test = [line.rstrip('\n') for line in lines]

        joinString = ''
        for line in range(len(test)):
            joinString = joinString + test[line].replace('[', '').replace('. ',' ').replace(']','')

        parsed = [float(item) for item in joinString.split()]
        if count % num_segs == 0:
            crit = [max(parsed)]
            saveHistory = [parsed]
        else:
            crit = crit + [max(parsed)]
            saveHistory = saveHistory + [parsed]
        if count % num_segs == num_segs-1:
            for i in range(num_segs):
                dictOfSegs['{} seg {}'.format(int((count + 1)/num_segs),num_segs - i)] = saveHistory[np.argsort(crit)[i]]

        x = list(range(0,224))
        if plot:
            plt.plot(x,parsed)
    if plot:
        plt.show()

        for item in seg1:
            plt.plot(x,item)
        plt.show()
        for item in seg2:
            plt.plot(x,item)
        plt.show()
        for item in seg3:
            plt.plot(x,item)
        plt.show()

    return dictOfSegs
