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

class LangModel:
    
    # default constructor
    def __init__(self):
        self.vocabulary = self.getVocabulary()
        self.ngram = self.getNgram()
        self.smoothing = self.getSmoothing()
        self.trainingFile = self.getTrainingFile()
        self.testingFile = self.getTestFile()

    #parameterized constructor
    def __init__(self,vocabulary=-1,ngram=-1,smoothing=0,trainingFile="",testingFile=""):
        self.vocabulary = self.getVocabulary(vocabulary)
        self.ngram = self.getNgram(ngram)
        self.smoothing = self.getSmoothing(smoothing)
        self.trainingFile = self.getTrainingFile(trainingFile)
        self.testingFile = self.getTestFile(testingFile)

    def getVocabulary(self,vocabulary=-1):

        choice = vocabulary

        if(choice==-1):
            print("Select a number for which vocabulary you would like to use:")
            print("0 : Fold the corpus to lowercase and use only the 26 letters of the alphabet [a-z]")
            print("1 : Distinguish up and low cases and use only the 26 letters of the alphabet [a-z,A-Z]")
            print("2 : Distinguish up and low cases and use all characters accepted by the built-in isalpha() method")
            choice = int(input ("Enter your choice: "))

        switcher = {
            0: self.generateVocabulary(0),
            1: self.generateVocabulary(1),
            2: self.generateVocabulary(2)
            }
        
        return switcher.get(choice,"Invalid selection")

    def getNgram(self,ngram=-1):

        choice = ngram

        if(choice==-1):
            print("Select a number for which size n-gram you would like to use:")
            print("1 : character unigrams")
            print("2 : character bigrams")
            print("3 : character trigrams")
            choice = int(input ("Enter your choice: "))

        switcher = {
            1: 1,
            2: 2,
            3: 3
            }
        
        return switcher.get(choice,"Invalid selection")

    def getSmoothing(self,smoothing=0):

        choice = smoothing

        if(choice==0):

            interrupt = False
            count = 0
                
            while(not interrupt):
                    
                count = count + 1
                    
                choice = float(input ("Enter a smoothing value between 0 and 1 : "))
                    
                if(choice>=0 or choice<=1):
                    interrupt = True
                    
                if(count>3):
                    print("You failed to provide a smoothing value between 0 and 1, program will continue with default value: 0 ")
                    choice==0
                    interrupt = True
        
        return choice

    def getTrainingFile(self,trainingFile=""):

        dataSet = list()
        fileName = trainingFile
        count = 0

        try:
            # read the data into a list
            with open(str(fileName), encoding="utf8") as file:
                dataSet = file.readlines()

        except FileNotFoundError :
            print("File does not exist")
            fileName=""

        if(fileName==""):

            interrupt = False
            count = 0
            while(not interrupt):
                count = count + 1
                fileName = input ("Enter a valid TRAINING file name with the extension : ")
                
                try:
                    # read the data into a list
                    with open(str(fileName), encoding="utf8") as file:
                        dataSet = file.readlines()

                except FileNotFoundError :
                    print("File does not exist")

                
                if(len(dataSet)>0):
                    interrupt = True
                    
                if(count>3):
                    print("You failed to provide a valid TRAINING file, program will use default training dataset")
                    
                    fileName = "training-tweets.txt"

                    # read the data into a list
                    with open(str(fileName), encoding="utf8") as file:
                        dataSet = file.readlines()

                    interrupt = True

        return dataSet


    def getTestFile(self,testFile=""):

        dataSet = list()
        fileName = testFile
        count = 0

        try:
            # read the data into a list
            with open(str(fileName), encoding="utf8") as file:
                dataSet = file.readlines()

        except FileNotFoundError :
            print("File does not exist")
            fileName=""

        if(fileName==""):

            interrupt = False
            count = 0

            while(not interrupt):
                count = count + 1
                fileName = input ("Enter a valid TEST file name with the extension : ")
                
                try:
                    # read the data into a list
                    with open(str(fileName), encoding="utf8") as file:
                        dataSet = file.readlines()

                except FileNotFoundError :
                    print("File does not exist")

                
                if(len(dataSet)>0):
                    interrupt = True
                    
                if(count>3):
                    print("You failed to provide a valid TEST file, program will use default training dataset")
                    
                    fileName = "test-tweets-given.txt"

                        # read the data into a list
                    with open(str(fileName), encoding="utf8") as file:
                        dataSet = file.readlines()

                    interrupt = True

        return dataSet

    def generateVocabulary(self, selection):

        select = selection

        if(select==0):

            dataSet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x','y','z']
        
        elif (select==1):

            dataSet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

        elif(select==2):
            
            fileName = "utf8.txt"
            dataSet = list()

            # read the data into a list
            with open(str(fileName), encoding="utf8") as file:

              while True:
                char = file.read(1)
                if not char:
                  break
                if(char.isalpha and char!=" "):
                    dataSet.append(char)
  
        return dataSet


#MAIN

test = LangModel(1,1)
print(test.vocabulary)


