import h5py
import numpy as np
import matplotlib.pyplot as plt

class ForcesPostProcessor:
    
    def __init__(self, h5_path: str, rpm: int):
        
        '''
        Parameters
        ----------
        h5_path : str
            The path and name of the converted h5 file.
        rpm : int
            Rotating speed of the rotor
        '''
        
        self.h5_path = h5_path
        self.rpm = rpm
        self.rev_time = 60 / rpm
        
        with h5py.File(self.h5_path, 'r') as f:
            self.time = f['Time'][:]
            self.locations = f['Geometry/Strip Positions']
            self.strip_keys = list(f['Strips'].keys())
            
    def get_strip_signal(self):
        return()