'''
Find Product

You have been given an array A of size N consisting of positive integers. You need to find and print the product of all the number in this array Modulo (10^9 + 7).

Input Format:
The first line contains a single integer N denoting the size of the array. The next line contains N space separated integers denoting the elements of the array

Output Format:
Print a single integer denoting the product of all the elements of the array Modulo (10^9 + 7). 
'''

MOD = (10**9 +7)
def product(n, arr):
	result = 1
	
	for i in range(0, n): 
		result = (result * arr[i]) % MOD 
	return result