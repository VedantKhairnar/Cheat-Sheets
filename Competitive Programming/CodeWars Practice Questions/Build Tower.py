'''
6 kyu	Build Tower
https://www.codewars.com/kata/576757b1df89ecf5bd00073b/train/python

Build Tower

Build Tower by the following given argument:
number of floors (integer and always greater than 0).

Tower block is represented as *

    Python: return a list;
    JavaScript: returns an Array;
    C#: returns a string[];
    PHP: returns an array;
    C++: returns a vector<string>;
    Haskell: returns a [String];
    Ruby: returns an Array;
    Lua: returns a Table;

Have fun!

for example, a tower of 3 floors looks like below

[
  '  *  ', 
  ' *** ', 
  '*****'
]

and a tower of 6 floors looks like below

[
  '     *     ', 
  '    ***    ', 
  '   *****   ', 
  '  *******  ', 
  ' ********* ', 
  '***********'
]

Go challenge Build Tower Advanced once you have finished this :)
https://www.codewars.com/kata/57675f3dedc6f728ee000256
'''

def tower_builder(n):
	return [("*" * (i*2-1)).center(n*2-1) for i in range(1, n+1)]