'''
5 kyu	Molecule to atoms
https://www.codewars.com/kata/52f831fa9d332c6591000511/train/python

For a given chemical formula represented by a string, count the number of atoms of each element contained in the molecule and return an object (associative array in PHP, Dictionary<string, int> in C#, Map<String,Integer> in Java).

For example:

water = 'H2O'
parse_molecule(water)                 # return {H: 2, O: 1}

magnesium_hydroxide = 'Mg(OH)2'
parse_molecule(magnesium_hydroxide)   # return {Mg: 1, O: 2, H: 2}

var fremy_salt = 'K4[ON(SO3)2]2'
parse_molecule(fremySalt)             # return {K: 4, O: 14, N: 2, S: 4}

As you can see, some formulas have brackets in them. The index outside the brackets tells you that you have to multiply count of each atom inside the bracket on this index. For example, in Fe(NO3)2 you have one iron atom, two nitrogen atoms and six oxygen atoms.

Note that brackets may be round, square or curly and can also be nested. Index after the braces is optional.
'''

import re
from collections import Counter

def expand_str(m):
    return m.group(1) * int(m.group(2))

def parse_molecule (formula):
    formula = re.sub(r'\(([^\)]+)\)(\d+)',expand_str,formula)
    formula = re.sub(r'\[([^\]]+)\](\d+)',expand_str,formula)
    formula = re.sub(r'\{([^\}]+)\}(\d+)',expand_str,formula)
    formula = re.sub(r'([A-Z][a-z]?)(\d+)',expand_str,formula)
    m = re.findall(r'[A-Z][a-z]?',formula)
    return Counter(m)