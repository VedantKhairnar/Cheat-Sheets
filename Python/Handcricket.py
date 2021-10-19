#!/usr/bin/env python
# coding: utf-8

# In[3]:

import random
def Handcricket(overs):
    Total_balls=(overs*6)
    print("Total number of balls : ",Total_balls)
    score=0
    n=1
    computer_score=random.randrange(0,(overs*36)) #Goal to be reached
    print(" computer score to beat: ",computer_score)
    while (score<computer_score) and (n<=Total_balls):
        computer=random.randrange(0,6)
        player=int(input("Enter your number [between 1 to 6]: "))
        if(player != computer):
            score=score+player
            print("Your score: ",score)
        else:
            print("OUT!!!")
            #player out
            break
        n+=1
    if(score>computer_score):
        print("Your score ",score,"\n")
        print("Congratulations!!You won the game.")
    elif(score<computer_score):
        print("Your score ",score)
        print("Sorry, you lost this game. Better luck next time.")
    else:
        print("Oops!!Its a draw")
    return 0
def main():
    overs=int(input("Enter the number of overs: "))
    p=Handcricket(overs)  #function calling
    print(p)
main()

# In[6]:


# In[ ]: