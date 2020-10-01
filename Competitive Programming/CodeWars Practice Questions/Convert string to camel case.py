'''
6 kyu	Convert string to camel case.py
https://www.codewars.com/kata/517abf86da9663f1d2000003/train/python

Complete the method/function so that it converts dash/underscore delimited words into camel casing. The first word within the output should be capitalized only if the original word was capitalized (known as Upper Camel Case, also often referred to as Pascal case).
Examples

to_camel_case("the-stealth-warrior") # returns "theStealthWarrior"

to_camel_case("The_Stealth_Warrior") # returns "TheStealthWarrior"
'''

def to_camel_case(text):
    return text[:1] + text.title()[1:].replace('_', '').replace('-', '')