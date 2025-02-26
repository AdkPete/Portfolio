import unittest
import chess_sim as cs
import numpy as np

def model_func(r1 , r2):
    '''
    simple function to assume games are 50/50 win loss, no draws.
    '''
    return 0.5 , 0
class TestTournamentMethods(unittest.TestCase):

    def setUp(self):
        self.expert = cs.Player(2000 , 2000, "expert")
        self.classC = cs.Player(1500, 1500, "intermediate")
        self.tourney = cs.Tournament([self.expert,self.classC] , model_func )
        
    def test_reset_score(self):
        
        self.expert.record = [0 , 1 , 0.5 , 0.5 , 0 , 1]
        self.expert.reset_score()
        self.assertEqual(self.expert.record , [])
        
    def test_simulate_game(self):
        
        res = []
        for i in range(100000):
        
            gr = self.tourney.simulate_game(self.expert , self.classC)
            res.append(gr)
            
        self.assertAlmostEqual(np.mean(res) , 0.5 , places = 2)
        
if __name__ == '__main__':
    unittest.main()