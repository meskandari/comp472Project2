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
