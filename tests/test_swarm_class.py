
import os, sys
project_dir = (os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add the parent directory to the path
sys.path.append(project_dir) # add the parent directory to the path

import pytest
import numpy as np
from src.Swarm import Swarm

def test_compute_interactions():
    # Test 1: Normal operation
    positions = np.array([[0, 1], [0, 0]])
    orientations = np.array([0, np.pi/2])
    swarm = Swarm(positions, orientations, vel_magnitude=1, sensing_radius=1, noise=0, box_length=10)
    swarm._compute_interactions()
    assert np.allclose(swarm.orientations, [np.pi/4, np.pi/4]), "Test 1 failed"
    
    # Test 2: All agents at same position
    positions = np.array([[0, 0], [0, 0]])
    orientations = np.array([0, np.pi/2])
    swarm = Swarm(positions, orientations, vel_magnitude=1, sensing_radius=1, noise=0, box_length=10)
    swarm._compute_interactions()
    assert np.allclose(swarm.orientations, [np.pi/4, np.pi/4]), "Test 2 failed"

    # Test 3: Particles at opposite corners
    positions = np.array([[0, 10], [0, 0]])
    orientations = np.array([0, np.pi/2])
    swarm = Swarm(positions, orientations, vel_magnitude=1, sensing_radius=1, noise=0, box_length=10)
    swarm._compute_interactions()
    assert np.allclose(swarm.orientations, [0, np.pi/2]), "Test 3 failed"
