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

Written by Dr. Peter A. Craig (2025)
'''

import numpy as np
import matplotlib.pyplot as plt


print ("Hello, would you like to play a game?")