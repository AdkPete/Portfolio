'''

Code to simulate the results of a chess tournament. Alternatively,
we can solve the inverse problem, which is estimating a player's rating
based on the results of a tournament.

This is done utilizing game data from lichess.org to estimate the 
probability of the result from a given game. For more detailed
simulations, we can use results of games from the specific player
in order to fine-tune draw rates, in an effort to account for
the effects of differing styles between players.

In python, we house all of the code required to run such a simulation,
along with some data handling routines that will read in the full game
records from lichess (see data directory) and parse them into a format
containing the minimum required information for these sims. This will
get output to a file that is more easily readable by other codes, and
this will be the starting point for other methods (see fortran code).


Link to data source : https://database.lichess.org/

The test data included in the repository is from 2013 - January. This
particular set was chosen for the small file size, and it is a
reasonable starting point. For more accurate results, you could include
a larger number of games by adding additional pgn files to the
data directory.

Written by Dr. Peter A. Craig (2025)
'''

import numpy as np
import matplotlib.pyplot as plt
import os

def parse_pgn(filename):
    '''
    Function to parse a pgn file and extract the useful information
    for the simulation. This includes the player ratings, the result,
    time controls, and the player colors.
    
    We'll do a bit of data filtering at this stage:
    1. If a rating or time control is missing, skip.
    
    '''
    
    ## Set up dictionary to store results

    data = {}
    data["White"] = []
    data["Black"] = []
    data["Result"] = []
    data["WhiteElo"] = []
    data["BlackElo"] = []
    data["Time Control"] = []
    
    f = open(filename, "r")
    
    ## Parse through the file, one game at a time.
    for i in f.readlines():

        if i.strip() == "":
            continue
        sl = i.split(" ")
        
        if len(sl) == 1:
            continue
        field = sl[1].replace('"',"").replace("]","").strip()
        
        if "[White " in i:
            white = field

        elif "[Black " in i:
            black = field
            
        elif "[Result " in i:
            result = field
            
        elif "[WhiteElo " in i:
            try:
                whiteelo = float(field)
            except:
                continue
            
        elif "[BlackElo " in i:
            try:
                blackelo = float(field)
            except:
                continue
        elif "[TimeControl " in i:
            tcontrol = field

        elif i[0] == "1":
            
            if not (white is None or black is None or whiteelo is None
                or blackelo is None or tcontrol is None or result is None):
                
                data["White"].append(white)
                data["Black"].append(black)
                data["WhiteElo"].append(whiteelo)
                data["BlackElo"].append(blackelo)
                data["Time Control"].append(tcontrol)
                data["Result"].append(result)
            
            white = None
            black = None
            whiteelo = None
            blackelo = None
            tcontrol = None
            result = None
            
    
    ## Close file
    f.close()
    
    ## Some basic checks to make sure that our data set makes sens

    length = len(data["Black"])
    
    for key in data.keys():
        if len(data[key]) != length:
            print (len(data[key]) , length , key)
            errmsg = "Input PGN file {} ".format(filename)
            errmsg += "is missing some header fields"
            raise ValueError(errmsg)
    

    return data

def fit_probabilities(pgndata):
    '''
    This function will fit a model for the probability of each result
    for a game given the ratings of white and black
    '''
    
    ##TODO : This configuration has poor memory properties. Update to
    ## Avoid having multiple data arrays in memory with the same data
    
    welo = np.array(pgndata["WhiteElo"])
    belo = np.array(pgndata["BlackElo"])
    result = np.array(pgndata["Result"])
    
    ii = np.where( (welo != 1500) & (belo != 1500))
    
    rdiff = welo[ii] - belo[ii]
    
    bw = 50
    diff_bins = np.arange(-1000 , 1000 , bw * 2)
    scores = []
    for bin in diff_bins:
        
        bin_ii = np.where( ( rdiff > bin - bw ) & (rdiff < bin + bw) )
        
        score = 0
        
        ## Add in wins for white first
        resii = np.where(result[ii][bin_ii] == "1-0")
        score += len(resii[0])
        
        ## Now we handle draws
        resii = np.where(result[ii][bin_ii] == "1/2-1/2")
        score += 0.5 * len(resii[0])
        
        score /= len(bin_ii[0])
        scores.append(score)
        
    plt.scatter(diff_bins , scores)
    plt.show()

for i in os.listdir("data/"):

    if i[-4:] == ".pgn":
        pgndata = parse_pgn("data/" + i)
        break
    
print ("Hello, would you like to play a game?")

fit_probabilities(pgndata)