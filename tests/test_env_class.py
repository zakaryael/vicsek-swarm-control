# unit test for the environment class (env.py) using pytest

# add the parent directory to the path
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
# add the examples directory to the path
dir = "examples"
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), dir))

# import the environment class
from env import SwarmEnv

# import the parameters
from parameters import *

# import pytest
import pytest 

# start the test
def test_env():
    
        # create the environment
        env = SwarmEnv(N=N, L=box_length, v0=v0, r0=r0, eta=eta, potential_fields=potential_fields, Tmax=Tmax, target_radius=1)
    
        # test the environment
        assert env.reset() is not None
        for i in range(10):
            assert env.step(0) is not None
