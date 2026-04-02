import os
import glob
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class SpanToRadiusConverter:
    
    '''
    Convert spanwise alligned sliced data to cylindrical (r, chord) dataset.
    '''
    
    def __init__(self, input_path: str, output_path: str, variable_col: str, span_col: str, chord_length=None, radius_resolution=0.01):
        
        '''
        Parameters
        ----------
        input_path : str
            The path to the input files.
        output_path : str
            The path to the output files.
        variable_col : str
            The name of the column containing the variable to be converted.
        span_col : str
            The name of the column containing the spanwise coordinate.
        readius_resolution : float, optional
            The resolution for the radius coordinate in the output dataset, by default 0.01.
        '''
        
        self.input_path = input_path
        self.output_path = output_path
        self.variable_col = variable_col
        self.span_col = span_col
        self.chord_length = chord_length
        self.radius_resolution = radius_resolution
        self.r_target = np.arange(0, 1 + radius_resolution, radius_resolution)
        
    def _split_surfaces(self, x: np.ndarray, values: np.ndarray):
        
        '''
        Split the dataset into upper and lower surfaces based on the radius coordinate.\n
        With this suction and pressure side are separated using a differentiation method.
        '''
        
        diff = np.diff(x)
        split_index = np.argmax(np.abs(diff))
        
        x_upper = x[:split_index + 1]
        values_upper = values[:split_index + 1]
        
        x_lower = x[split_index + 1:]
        values_lower = values[split_index + 1:]
        
        if x_upper[0] > x_upper[-1]:
            x_upper = x_upper[::-1]
            values_upper = values_upper[::-1]
            
        if x_lower[0] > x_lower[-1]:
            x_lower = x_lower[::-1]
            values_lower = values_lower[::-1]
            
        return (x_upper, values_upper), (x_lower, values_lower)
    
    def read(self):
        
        '''
        Read the input files and store the spanwise coordinate and variable values.
        '''
        
        self.files = sorted(glob.glob(os.path.join(self.input_path, '*.csv')))
        
        if not self.files:
            raise FileNotFoundError(f'No files found in {self.input_path}')
        
        N = len(self.files)
        
        chord = np.linspace(1, 0, N)
        
        if self.chord_length is not None:
            chord = chord * self.chord_length
            
        self.chords = chord
        
        self.upper_surfaces = []
        self.lower_surfaces = []   
        self.all_r = []
        
        for i, f in enumerate(self.files):
            
            df = pd.read_csv(f)
            
            x = pd.to_numeric(df[self.span_col].values, errors='coerce')
            values = pd.to_numeric(df[self.variable_col].values, errors='coerce')
            
            mask = ~np.isnan(x) & ~np.isnan(values)
            x = x[mask]
            values = values[mask]
            
            (x_upper, values_upper), (x_lower, values_lower) = self._split_surfaces(x, values)
            
            y = self.chords[i]
            
            r_upper = np.sqrt(x_upper**2 + y**2)
            r_lower = np.sqrt(x_lower**2 + y**2)
            
            self.all_r.append(r_upper)
            self.all_r.append(r_lower)
             
            self.upper_surfaces.append((r_upper, values_upper))
            self.lower_surfaces.append((r_lower, values_lower))
            
    def build_radius_grid(self):
        
        '''
        Build a common radius grid for all surfaces based on the minimum and maximum radius across all datasets.
        '''
        
        all_r = np.concatenate(self.all_r)
        
        self.r_min = np.min(all_r)
        self.r_max = np.max(all_r)
        
        self.r_target = np.arange(self.r_min, self.r_max + self.radius_resolution, self.radius_resolution)
        
    def interpolate(self):
        
        '''
        Interpolate the variable values onto the common radius grid for both upper and lower surfaces.
        '''
        
        N_chord = len(self.chords)
        N_radius = len(self.r_target)
        
        self.upper = np.zeros((N_radius, N_chord))
        self.lower = np.zeros((N_radius, N_chord))
        
        for j in range(N_chord):
            
            r_upper, values_upper = self.upper_surfaces[j]
            r_lower, values_lower = self.lower_surfaces[j]
            
            interp_upper = interp1d(r_upper, values_upper, bounds_error=False, fill_value=np.nan)
            interp_lower = interp1d(r_lower, values_lower, bounds_error=False, fill_value=np.nan)
            
            self.upper[:, j] = interp_upper(self.r_target)
            self.lower[:, j] = interp_lower(self.r_target)
            
    def write_h5(self):
        
        '''
        Write the interpolated datasets to an HDF5 file.
        '''
        
        with h5py.File(self.output_path, 'w') as h5f:
            
            geo = h5f.create_group('Geometry')
            geo.create_dataset('Radius', data=self.r_target)
            geo.create_dataset('Chord', data=self.chords)
            
            data = h5f.create_group('Data')
            data.create_dataset('Upper', data=self.upper)
            data.create_dataset('Lower', data=self.lower)
            
    def convert(self):
        
        '''
        Main method to perform the conversion from spanwise to radius-based dataset.
        '''
        
        self.read()
        self.build_radius_grid()
        self.interpolate()
        self.write_h5()
