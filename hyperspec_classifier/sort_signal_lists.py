import os
from numpy import loadtxt
import matplotlib.pyplot as plt
import math
import numpy as np

def parse_hyperspec_lists(dir, search_string, plot = False):
    seg1 = []
    seg2 = []
    seg3 = []
    list_of_folders = [i for i in os.listdir(dir) if search_string in i]
    for count, signal in enumerate(list_of_folders):
        with open(dir + signal, 'r+') as rf:
            lines = rf.readlines()
            test = [line.rstrip('\n') for line in lines]

        joinString = ''
        for line in range(len(test)):
            joinString = joinString + test[line].replace('[', '').replace('. ',' ').replace(']','')

        parsed = [float(item) for item in joinString.split()]
        if count % 3 == 0:
            crit = [max(parsed)]
            index = count % 3
            saveHistory = [parsed]
        else:
            crit = crit + [max(parsed)]
            saveHistory = saveHistory + [parsed]
        if count % 3 == 2:
            seg3.append(saveHistory[np.argsort(crit)[0]])
            seg2.append(saveHistory[np.argsort(crit)[1]])
            seg1.append(saveHistory[np.argsort(crit)[2]])

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

    return seg1, seg2, seg3
