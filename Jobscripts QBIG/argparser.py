# The goal: Provide a number of optional arguments and parse and write them to corresponding variables.

import argparse

# Define the variables that would potentially be provided via command line.
# Also add the default values here.
a = 0
b = 0.
c = ''
d = False

# Initialize parser and define arguments.
parser = argparse.ArgumentParser(description='App description')

parser.add_argument('-a', type=int, help='integer')
parser.add_argument('-b', type=float, help='floating point numbers')
parser.add_argument('-c', type=str, help='string')
parser.add_argument('-d', type=bool, help='boolean')

# parse_args() generates a dict which contains all values it could find.
# If a value was not provided the dict will be 'None' for that variable.
args = parser.parse_args()

# Unpack the dict by hand and assign the values to the original variables. 
if (args.a != None):
  a = args.a
if (args.b != None):
  b = args.b
if (args.c != None):
  c = args.c
if (args.d != None):
  d = args.d
  
print((a, b, c, d))


