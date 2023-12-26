import numpy as np
import torch
from stlcg.src import stlcg
import pickle

class StateChecker:
    def __init__ (self, agent):

        """ Initializes the states to be checked as tensors

        """
        self.agent = agent
        self.speed_lim = 3.0 
        self.speed_trace = torch.empty(1, 0, 1)


    def get_state_trace(self):

        """ Define a trace of the states to be checked and see whether or not they went 
            above a user defined limit during the entire experiment
        """
        state = self.agent.get_observation()
        v = torch.tensor([[state['groundspeed']]])
        self.speed_trace = torch.cat((self.speed_trace, v.unsqueeze(0)), dim = 1)

        return self.speed_trace


    def speed_limit(self):

        """ Checks if the defined speed limit was not exceeded

        """

        v = self.speed_trace
        vlim = torch.tensor(self.speed_lim)
        vlim_exp = stlcg.Expression('vlim', vlim)
        v_exp = stlcg.Expression('v', v.flip(1))

        predicate = v_exp < vlim_exp
        x = stlcg.Always(subformula = (predicate))
        inputs = (v_exp)
        r = x.robustness(inputs, pscale = 1, scale = -1)

        if r < 0:
            print('Speed Limit Violated')
            
        else:
            print('Speed Limit Not Violated')