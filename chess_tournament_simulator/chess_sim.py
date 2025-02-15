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
import sys

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
    data["Event"] = []
    data["Date"] = []
    data["Time"] = []
    
    ## Some variables to temporarily store header fields.
    white = None
    black = None
    whiteelo = None
    blackelo = None
    tcontrol = None
    result = None
    event = None
    time = None
    date = None
    
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
            ## Some steps here to shrink the data size a little. Let's
            ## save this as only the result for white (since the result
            ## for black is then implied.
            if "1/2" in field:
                result = 0.5
            else:
                result = float(field.split("-")[0])
            
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

        elif "[Event" in i:
            event = field
        
        elif "[UTCDate" in i:
            ## Strip out the . to reduce memory use. It's nice for
            ## human readability, but is unecessary to machines.
            date = field.split(".")[0]
        
        elif "[UTCTime" in i:
            time = field
            
        elif i[0] == "1":
            
            if not (white is None or black is None or whiteelo is None
                or blackelo is None or tcontrol is None or result is None or
                time is None or date is None or event is None):
                
                data["White"].append(white)
                data["Black"].append(black)
                data["WhiteElo"].append(whiteelo)
                data["BlackElo"].append(blackelo)
                data["Time Control"].append(tcontrol)
                data["Result"].append(result)
                data["Date"].append(date)
                data["Time"].append(time)
                data["Event"].append(event)
            white = None
            black = None
            whiteelo = None
            blackelo = None
            tcontrol = None
            result = None
            event = None
            time = None
            date = None
    
    ## Close file
    f.close()
    
    return data

def pmodel(r1 , r2):
    
    '''
    simple model to predict the result of a given game. Estimates the 
    probability for a win for white, draw, or a win for black.
    
    Parameters
    __________
    r1 : rating for white
    r2 : rating for black
    
    Returns
    _______
    pw : probability for white win
    pd : probability for draw
    
    '''
    
    return 0.5 , 0.25
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
    
    bw = 25
    diff_bins = np.arange(-1000 , 1000 , bw * 2)
    scores = []
    N = []
    draw_rate = []
    win_rate = []
    for bin in diff_bins:
        
        bin_ii = np.where( ( rdiff > bin - bw ) & (rdiff < bin + bw) )
        
        score = np.mean(result[ii][bin_ii]) / 2.0
        draws = np.where(result[ii][bin_ii] == 1)
        draw_rate.append(len(draws[0]) / len(bin_ii[0]))
        wins = np.where(result[ii][bin_ii] == 2)
        win_rate.append(len(wins[0]) / len(bin_ii[0]))
        scores.append(score)
        N.append(len(bin_ii[0]))

    ## Quick plot of the score vs. rating difference. It should be the case
    ## that with rating differences near 0, the score should be close to
    ## 0.5, and as the rating difference moves away from 0, the score should
    ## approach 1 or 0. Note that the actual point of a 0.5 score 
    ## is expected to be slightly offset from 0, because the player
    ## with the white pices has a small advantage.
    
    plt.scatter(diff_bins , scores)
    plt.errorbar(diff_bins , scores , yerr = 1 / np.sqrt(N) , ls = 'none')
    plt.show()
    
    plt.scatter(diff_bins , draw_rate , label = "Draw Rate")
    plt.scatter(diff_bins , win_rate , color = "orange" , label = "White Wins")
    plt.scatter(diff_bins , 1 - np.array(win_rate) - np.array(draw_rate) , color = "red" , label = "Black Wins")
    plt.axhline(0.5 , color = "black" , ls = "--")
    plt.axvline(0 , color = "black" , ls = "--")
    plt.legend()
    plt.show()
    
pgndata = None
for i in os.listdir("data/"):

    if i[-4:] == ".pgn":
        data = parse_pgn("data/" + i)
        if pgndata is None:
            pgndata = data
            
        else:
            for key in pgndata.keys():
                pgndata[key] += data[key]

def memory_check(pgndata):
    
    '''
    Quick function to check out our memory usage. This can be an issue
    with the billions of games available
    '''
    for key in pgndata:
        print (key , sys.getsizeof(np.array(pgndata[key])) / 1000000)
## Some tricks to reduce memory consumption. Many of our values can be
## represented fully using smaller objects than the standard.
## For reference, this reduces our file size by ~30%.

pgndata["Result"] =( np.array(pgndata["Result"]) * 2).astype(np.int8)
pgndata["WhiteElo"] = np.array(pgndata["WhiteElo"]).astype(np.int16)
pgndata["BlackElo"] = np.array(pgndata["BlackElo"]).astype(np.int16)

memory_check(pgndata)
print ("Hello, would you like to play a game?")

fit_probabilities(pgndata)

np.save("data/test.npy" , pgndata)