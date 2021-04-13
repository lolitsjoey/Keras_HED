# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 09:06:00 2021

@author: joeba
"""
import os
import shutil
from src.networks.hed_reduced import hed

wholeNotes = ['D:/FRNLib/noteLibrary/Genuine/100/', 'D:/FRNLib/noteLibrary/Counterfeit/100/']

for ii, wholeNotesDir in enumerate(wholeNotes):
    notesDone = 0
    for noteFolder in os.listdir(wholeNotesDir):
        for spectrum in os.listdir(wholeNotesDir + noteFolder):
            if 'RGB_Front' in spectrum:
                if ii == 0:
                    shutil.copyfile(wholeNotesDir + noteFolder + '/' + spectrum, './test_station/genuine_{}.bmp'.format(notesDone))
                else:
                    shutil.copyfile(wholeNotesDir + noteFolder + '/' + spectrum, './test_station/counterfeit_{}.bmp'.format(notesDone))
                notesDone += 1