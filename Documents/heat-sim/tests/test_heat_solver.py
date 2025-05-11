
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from heat_solver import heat_diffusion_1d, heat_diffusion_1d_vectorized
import numpy as np

def test_output_shape_scalar():
    x, history = heat_diffusion_1d(nx=50, nt=20)
    assert x.shape[0] == 50
    assert len(history) == 21

def test_center_temperature_decreases_scalar():
    _, history = heat_diffusion_1d(nx=50, nt=10)
    initial = history[0][25]
    final = history[-1][25]
    assert final < initial

def test_output_shape_vectorized():
    x, history = heat_diffusion_1d_vectorized(nx=50, nt=20)
    assert x.shape[0] == 50
    assert len(history) == 21

def test_center_temperature_decreases_vectorized():
    _, history = heat_diffusion_1d_vectorized(nx=50, nt=10)
    initial = history[0][25]
    final = history[-1][25]
    assert final < initial
