# March 29 2020
# Concordia University
# COMP 472 Section NN
# Project 2
# By:
# Jason Brennan - 27793928
# Maryam Eskandari - 40065716
# Martin Grezak - 25693810

import sys
import numpy as np
from operator import itemgetter, attrgetter
from collections import OrderedDict
from enum import Enum
import time

# An enum class for flagging the language in use
class Language(Enum):
    EU = 0 #BASQUE
    CA = 1 #CATALAN
    GL = 2 #GALICIAN
    ES = 3 #SPANISH
    EN = 4 #ENGLISH
    PT = 5 #PORTUGUESE

class Model:
    
    # default constructor
    def __init__(self):
      self.vocabular = getVocabulary()
      self.ngram = None
      self.smoothing = None
      self.trainingFile = None
      self.testingFile = None

    def __init__(self,vocabulary,ngram,smoothing=0,trainingFile,testingFile):
        self.vocabulary = vocabulary
        self.ngram = ngram
        self.smoothing = smoothing
        self.trainingFile = trainingFile
        self.testingFile = testingFile

    def getVocabulary(self,vocabulary=-1):

        choice = vocabulary
        if(choice==-1):
            print("Select a number for which vocabulary you would like to use:")
            print("1 : Fold the corpus to lowercase and use only the 26 letters of the alphabet [a-z]")
            print("2 : Distinguish up and low cases and use only the 26 letters of the alphabet [a-z,A-Z]")
            print("2 : Distinguish up and low cases and use all characters accepted by the built-in isalpha() method")
            choice = input ("Enter your choice: ")

        switcher = {
            0: [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x,y,z],
            1: [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x,y,z,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z]
            2: [T,B,D]
            }
        
        return switcher.get(choice,"Invalid selection")

    
    