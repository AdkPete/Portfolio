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

The test data included in the repository is from September 2014. This
particular set was selected because it is small enough that the required
data can be readily hosted on github, while it still has data from
~1,000,000 games, enough to have good statistics.

Written by Dr. Peter A. Craig (2025)
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import scipy.optimize as opt ##One day, replace with my own module
from tabulate import tabulate
from tqdm import tqdm

class Tournament:

    '''
    Class to simulate a chess tournament. Will take in a list of
    players, and offers multiple pairing systems to simulate
    varying kinds of tournaments
    '''

    def __init__(self , playerlist , game_model):

        self.playerlist = playerlist
        self.pf = game_model
        self.current_round = 0
    def simulate_game(self,  white , black):

        '''
        Function to determine the result of a single game
        white and black should be two player objects
        model is a function that takes in the two ratings and will
        return the probability of a win for white and the probability
        of a draw
        '''

        pw , pd = self.pf(white.true_rating , black.true_rating)
        random_res = np.random.random()

        if random_res < pd:
            result = 0.5

        elif random_res < pw + pd:
            result = 1.0

        else:
            result = 0.0

        return result

    def pair_round_robin(self):
        
        '''
        Function to generate pairings for a round robin tournamnet
        
        '''
        
        Npl = len(self.playerlist)
        self.pairings = {}
        self.Nrounds = Npl - 1
        ## We're going to assign a random order to the players to start.
        ## This follows a circular pairing method. There are others
        # available for choosing the order of games and the colors,
        # but this is a simple method

        if Npl % 2 != 0:
            raise ValueError("Odd number of players not supported yet")
        
        ii = np.random.choice(range(Npl) , Npl , replace = False)
        #ii = np.array(range(Npl)) + 1
        positions = ii[1::]
        row2 = int(Npl / 2) -1
        positions[row2 ::] = positions[row2::][::-1]

        for round in range(self.Nrounds):
            self.pairings[round] = []
            ## Handle the first player seperately. All other pairings
            ## can then be pulled from player positions.
            p1 = self.playerlist[ii[0]]
            p2 = self.playerlist[positions[-1]]
            if round % 2 == 0:

                self.pairings[round].append([p1,p2])
            else:
                self.pairings[round].append([p2,p1])
            
            pi = 0
            while pi < row2:
                p1 = self.playerlist[positions[pi]]
                p2 = self.playerlist[positions[-1 * pi - 2]]
                if pi % 2 == 0:
                    self.pairings[round].append([p1 , p2])
                else:
                    self.pairings[round].append([p2 , p1])
                pi += 1
            ## Update player positions
            positions = np.roll(positions , 1)
            
        return 0
        for round in self.pairings.keys():
            for k in self.pairings[round]:
                
                player = self.playerlist[5]
                if player in k:
                    if k[0] == player:
                        print ("W vs " , k[1].name)
                    else:
                        print ("B vs " , k[0].name)
        
    def pair_doublerr(self):
        '''
        Function to generate pairings for a double round robin.
        Takes in a random seed that will determine the order of the
        pairings
        '''

        Npl = len(self.playerlist)
        self.pairings = []
        self.pair_round_robin()
        Nrounds = 2 * (Npl - 1)
        print (Nrounds)

        round_pairs = []
        for i in np.array(range(Nrounds)) + 1:
            inds = np.array(range(Npl))

            for k in range(len(inds)):
                if k > Npl / 2:
                    break
                if i % 2 == 0:
                    round_pairs.append( [inds[k] , inds[-(k + 1)]] )
        print (round_pairs)
            

        return 0
    
    def simulate_round(self):
        '''
        Simulate results for a single round of a tournament
        '''
        
        for i in self.pairings[self.current_round]:
            result = self.simulate_game(i[0] , i[1])
            i[0].record.append(result)
            i[1].record.append(1.0 - result)
        self.current_round += 1
        
    def simulate_tournament(self):
        
        '''
        Simulate the entire tournament
        '''
        while self.current_round < self.Nrounds:
            self.simulate_round()
        
    def crosstable(self):
        '''
        Funcion to nicely display the tournament crosstable
        '''
        self.playerlist.sort()
        lname = 0
        for i in self.playerlist:
            if len(i.name) > lname:
                lname = len(i.name)
        lname += 1
        
        print ("\n")
        
        print ("-" * lname + "|" + self.Nrounds* ( ( "-" * 5 ) + "|" ) )
        for i in self.playerlist:
            ind = self.playerlist.index(i)
            line0 = str(ind + 1) + " " * (lname - len(str(ind + 1))) + "|"
            line0 += + self.Nrounds* ( ( " " * 5 ) + "|" ) 
            print (line0)
            line1 = i.name + " " * (lname - len(i.name) ) + "|"
            

            line2 = str(i.public_rating) + (" " * (lname - 4)) + "|"
            
            for round in self.pairings.keys():
                
                for game in self.pairings[round]:
                    if i == game[0]:
                        
                        pn = self.playerlist.index(game[1]) + 1
                        
                        
                        if len(i.record) < round + 1:
                            score = ""
                        else:
                            
                            score = np.sum(i.record[0:round + 1])
                        
                        
                        line2 += str(score) + " " * (5 - len(str(score))) + "|"
                        l1 =  "W{}".format(pn)
                        l1 += " " * (5 - len(l1)) + "|"
                        line1 += l1
                        
                    elif i == game[1]:
                        pn = self.playerlist.index(game[0]) + 1
                        
                        
                        if len(i.record) < round + 1:
                            score = ""
                        else:
                            
                            score = np.sum(i.record[0:round + 1])
                        
                        line2 += str(score) + " " * (5 - len(str(score))) + "|"
                        l1 =  "B{}".format(pn)
                        l1 += " " * (5 - len(l1)) + "|"
                        line1 += l1
            print (line1)
            print (line2)
            row = ""
            for i in range(len(line1)):
                if line1[i] == "|":
                    row += "|"
                else:
                    row += "-"
            print (row)
    
    def reset(self):
        '''
        Function to reset the tournament
        '''
        self.current_round = 0
        for i in self.playerlist:
            i.reset_score()
        self.pairings = {}
        
class Player:

    '''
    class to keep track of information for a player. Intent is to keep
    track of a public rating (used for pairings) and a true rating
    (used for game result probabilities). Also intended to house code
    to update player ratings mid-tournament based on results.
    '''

    def __init__(self , public_rating , true_rating , name):
        self.public_rating = public_rating
        self.true_rating = true_rating
        self.record = []
        self.name = name

    def __eq__(self,other):
        
        ''' 
        Players are equal if they have the same score and public rating
        '''
        if (np.sum(self.record) == np.sum(other.record) and
             self.public_rating == other.public_rating):
            return True
        return False
    
    def __gt__(self , other):
        '''
        PLayers are ranked by score. If that fails, we rank by rating
        '''
        if np.sum(self.record) < np.sum(other.record):
            return True
        elif (np.sum(self.record) == np.sum(other.record) and 
              self.public_rating < other.public_rating):
            return True
            
        return False
    
    
    def update_ratings(self , opponent_rating , result):
        '''
        Function to update player ratings
        TODO : Write this function
        '''
        self.new_rating = -1 

    def reset_score(self):
        '''
        Function to reset the player's tournament results
        '''
        
        self.record = []

        
        
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
    #data["White"] = []
    #data["Black"] = []
    data["Result"] = []
    data["WhiteElo"] = []
    data["BlackElo"] = []
    data["Base Time"] = []
    data["Increment"] = []
    #data["Event"] = []
    #data["Date"] = []
    #data["Time"] = []
    
    ## Some variables to temporarily store header fields.
    white = None
    black = None
    whiteelo = None
    blackelo = None
    btime = None
    itime = None
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
        elif "[TimeControl " in i and field != "-":

            btime = int(field.split("+")[0])
            itime = int(field.split("+")[1])

        
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
                or blackelo is None or btime is None or result is None
                or time is None or date is None or event is None
                or itime is None):
                
                data["WhiteElo"].append(whiteelo)
                data["BlackElo"].append(blackelo)
                data["Base Time"].append(btime)
                data["Increment"].append(itime)
                data["Result"].append(result)

                ## TODO: Add back in when we have features that use this info

                #data["White"].append(white)
                #data["Black"].append(black)
                #data["Date"].append(date)
                #data["Time"].append(time)
                #data["Event"].append(event)
            white = None
            black = None
            whiteelo = None
            blackelo = None
            btime = None
            itime = None
            result = None
            event = None
            time = None
            date = None
    
    ## Close file
    f.close()
    

    return data

def pmodel(r1 , r2 , k , x0 , draw_x0 , draw_sigma , draw_amp):
    
    '''
    simple model to predict the result of a given game. Estimates the 
    probability for a win for white, draw, or a win for black. This
    model uses a logistic function to estimate the win probability
    for white, with a gaussian for the win probability for black.
    These models are guesses based on the shape of the probability 
    curves.
    
    Parameters
    __________
    r1 : rating for white
    r2 : rating for black
    k : growth rate (how steeply does the probability change)
    x0 : Location of 0.5 win probability
    draw_x0 : Location of max draw probability
    draw_sigma : Width of draw probability
    draw_amp : Amplitude of draw probability

    Returns
    _______
    pw : probability for white win
    pd : probability for draw
    
    '''
    
    rdiff = r1 - r2
    ## Win probability first
    pw = 1 / (1 + np.exp(-k * (rdiff - x0)))
    pd =  draw_amp / np.sqrt(2 * np.pi * draw_sigma ** 2)
    pd *= np.exp(-1 * (rdiff - draw_x0) ** 2 / (2 * draw_sigma ** 2))

    return pw , pd


def setup_eval_function(pgndata , bw = 25):

    '''
    This function will create a new function to evaluate our model,
    for optimization purposes.
    '''

    welo = np.array(pgndata["WhiteElo"])
    belo = np.array(pgndata["BlackElo"])
    result = np.array(pgndata["Result"])

    rdiff = welo - belo
    ii = np.where( (welo != 1500) & (belo != 1500))
    rdiff = rdiff[ii]

    bw = 25
    diff_bins = np.arange(-1000 , 1000 , bw * 2)
    scores = []
    N = []
    draw_rate = []
    win_rate = []
    used_bins = []
    for bin in diff_bins:
        
        bin_ii = np.where( ( rdiff > bin - bw ) & (rdiff < bin + bw) )
        if len(bin_ii[0]) == 0: ##Just to catch any empty bins
            continue

        used_bins.append(bin)
        
        score = np.mean(result[ii][bin_ii]) / 2.0
        draws = np.where(result[ii][bin_ii] == 1)
        draw_rate.append(len(draws[0]) / len(bin_ii[0]))
        wins = np.where(result[ii][bin_ii] == 2)
        win_rate.append(len(wins[0]) / len(bin_ii[0]))
        scores.append(score)
        N.append(len(bin_ii[0]))

    used_bins = np.array(used_bins)

    def f(x):
        '''
        Function to evaluate the model for the given data
        '''
        pw , pd = pmodel(used_bins , 0 , x[0] , x[1] , x[2] , x[3] , x[4])

        return np.sqrt( np.sum( (pw - win_rate) ** 2) + np.sum((pd - draw_rate) ** 2))

    return f , used_bins , scores , N , draw_rate , win_rate

def fit_probabilities(pgndata , show_plots = False):
    '''
    This function will fit a model for the probability of each result
    for a game given the ratings of white and black
    '''
    
    ##TODO : This configuration has poor memory properties. Update to
    ## Avoid having multiple data arrays in memory with the same data
    
    f , diff_bins , scores , N , draw_rate , win_rate = setup_eval_function(pgndata)
    

    ## Quick plot of the score vs. rating difference. It should be the case
    ## that with rating differences near 0, the score should be close to
    ## 0.5, and as the rating difference moves away from 0, the score should
    ## approach 1 or 0. Note that the actual point of a 0.5 score 
    ## is expected to be slightly offset from 0, because the player
    ## with the white pices has a small advantage.
    

    x0 = [0.01 , 0 , 0 , 200 , 5.0]
    
    res = opt.minimize(f , x0 , method = "Nelder-Mead")
    print (res)

    ## Model Guess:
    modelx = np.linspace(-1000 , 1000 , 2000)
    model_pw = []
    model_pd = []

    for i in range(len(modelx)):
        pw , pd = pmodel(modelx[i], 0, res.x[0], res.x[1], res.x[2], res.x[3], res.x[4])
        model_pw.append(pw)
        model_pd.append(pd)

    model_pw = np.array(model_pw)
    model_pd = np.array(model_pd)

    if show_plots:
        plt.scatter(diff_bins , scores)
        plt.errorbar(diff_bins , scores , yerr = 1 / np.sqrt(N) , ls = 'none')
        plt.show()
        
        plt.scatter(diff_bins , draw_rate , label = "Draw Rate")
        plt.scatter(diff_bins , win_rate , color = "orange" , label = "White Wins")
        plt.scatter(diff_bins , 1 - np.array(win_rate) - np.array(draw_rate) , color = "red" , label = "Black Wins")
        plt.plot(modelx, model_pw, color = "black", ls = "--",
            label = "Model White Win")
        plt.plot(modelx, model_pd, color = "green", ls = "--",
            label = "Model Draw")
        plt.plot(modelx , 1 - model_pw - model_pd , color = "black", ls = "--",
            label = "Model Black Win")
        #plt.axhline(0.5 , color = "black" , ls = "--")
        #plt.axvline(0 , color = "black" , ls = "--")
        plt.legend()
        plt.show()
    
    return res.x

def memory_check(pgndata):
    
    '''
    Quick function to check out our memory usage. This can be an issue
    with the billions of games available
    '''
    for key in pgndata:
        print (key , sys.getsizeof(np.array(pgndata[key])) / 1000000)


if __name__ == "__main__":
    ##First, read in data, either from a pgn file or from a numpy file of
    ##pre-reduced data
    save_name = "data/09_2014_lichess.npy"

    if os.path.exists(save_name):
        pgndata = np.load(save_name , allow_pickle = True).flat[0]
        
    else:
        pgndata = parse_pgn("data/lichess_db_standard_rated_2014-09.pgn")

        ## Some tricks to reduce memory consumption. Many of our values can be
        ## represented fully using smaller objects than the standard.
        ## For reference, this reduces our file size by ~30%.

        pgndata["Result"] =( np.array(pgndata["Result"]) * 2).astype(np.int8)
        pgndata["WhiteElo"] = np.array(pgndata["WhiteElo"]).astype(np.int16)
        pgndata["BlackElo"] = np.array(pgndata["BlackElo"]).astype(np.int16)
        pgndata["Base Time"] = (np.array(pgndata["Base Time"]) / 60).astype(np.uint8)
        pgndata["Increment"] = np.array(pgndata["Increment"]).astype(np.uint8)


    memory_check(pgndata)
    print ("Hello, would you like to play a game?")

    modelx = fit_probabilities(pgndata)

    def game_model(r1 , r2):
        return pmodel(r1, r2, modelx[0], modelx[1], modelx[2], modelx[3], modelx[4])


    players = []

    players.append(Player(2000 , 2000 , "expert"))
    players.append(Player(1800 , 1800 , "A"))
    players.append(Player(1600 , 1600 , "B"))
    players.append(Player(1400 , 1400 , "C"))
    players.append(Player(1200 , 1200 , "D"))
    players.append(Player(1000 , 1000 , "E"))

    double_round_robin = Tournament(players , game_model)

    ## Basic check of game simulation!

    results = []
    for i in range(1000):
        res = double_round_robin.simulate_game(players[0] , players[0])
        results.append(res)

    match = results[0:10]

    rows = []
    row1 = ["Expert 1"]
    row2 = ["Expert 2"]
    header = ["Player"]
    rn = 1
    for i in match:
        row1.append(i)
        row2.append(1.0 - i)
        header.append(rn)
        rn += 1

    header.append("Total")

    row1.append(sum(row1[1::]))
    row2.append(sum(row2[1::]))

    if row1[-1] > row2[-1]:

        rows = [header , row1 , row2]

    else:
        rows = [header , row2 , row1]
    print (tabulate(rows , tablefmt = "fancy_grid"))
    print ("Equal Ratings Result: " , np.mean(results))

    results = []
    for i in range(1000):
        res = double_round_robin.simulate_game(players[0] , players[1])
        results.append(res)

    print ("200 point favorite for white Ratings Result: " , np.mean(results))


    results = []
    for i in range(10):
        res = double_round_robin.simulate_game(players[1] , players[0])
        results.append(res)



    print ("200 point favorite for black Ratings Result: " , np.mean(results))

    
    double_round_robin.pair_round_robin()
    double_round_robin.crosstable()
    double_round_robin.simulate_round()
    double_round_robin.crosstable()
    double_round_robin.simulate_tournament()
    double_round_robin.crosstable()
    
    double_round_robin.reset()
    double_round_robin.pair_round_robin()
    double_round_robin.crosstable()
    double_round_robin.simulate_tournament()
    double_round_robin.crosstable()
    double_round_robin.reset()
    
    escore = []
    for i in tqdm(range(5000000)):
        double_round_robin.pair_round_robin()
        double_round_robin.simulate_tournament()
        escore.append(np.sum(players[0].record))
        double_round_robin.reset()
        
    for i in np.unique(escore):
        print (i , escore.count(i) * 100.0 / len(escore))
        