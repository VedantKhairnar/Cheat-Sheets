'''
7 kyu	Isograms
https://www.codewars.com/kata/54ba84be607a92aa900000f1/train/python

An isogram is a word that has no repeating letters, consecutive or
non-consecutive. Implement a function that determines whether a string that
contains only letters is an isogram. Assume the empty string is an isogram.
Ignore letter case.

is_isogram("Dermatoglyphics" ) == true
is_isogram("aba" ) == false
is_isogram("moOse" ) == false # -- ignore letter case
'''

def is_isogram(string):
	return len(string) == len(set(string.lower()))