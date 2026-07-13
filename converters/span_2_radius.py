import os
import pdb
import glob
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class SpanConverter:
    
    '''
    Convert spanwise alligned sliced data to cylindrical (r, chord) dataset.
    '''
    
    def __init__(self, input_path: str, output_path: str, variable_col: str, span_col: str, chord_length=None, resolution=0.01, coordinate_system='polar', surface_split=True):
        
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
        resolution : float, optional
            The resolution for the radius coordinate in the output dataset, by default 0.01.
        coordinate_system : str, optional
            The coordinate system to use for the output dataset, by default 'polar'.
        surface_split : bool, optional
            Whether to split the surfaces into upper and lower parts, by default True.
        '''
        
        self.input_path = input_path
        self.output_path = output_path
        self.variable_col = variable_col
        self.span_col = span_col
        self.chord_length = chord_length
        self.coordinate_system = coordinate_system.lower()
        if self.coordinate_system not in ['polar', 'cartesian']:
            raise ValueError("coordinate_system must be either 'polar' or 'cartesian'")
        self.resolution = resolution
        self.coord_target = None
        self.surface_split = surface_split
       
    def _compute_coordinate(self, x, y):
        
        '''
        Returns the interpolation coordinate.
        
        polar -> r = sqrt(x^2 + y^2)
        cartesian -> r = x
        '''
        
        if self.coordinate_system == 'polar':
            return np.sqrt(x**2 + y**2)
        elif self.coordinate_system == 'cartesian':
            return x
        else:
            raise ValueError("coordinate_system must be either 'polar' or 'cartesian'")
     
    def _split_surfaces(self, coord: np.ndarray, values: np.ndarray):
        
        '''
        Split the dataset into upper and lower surfaces based on the radius coordinate.\n
        With this suction and pressure side are separated using a differentiation method.
        '''
        
        split_index = np.argmax(coord)
        
        coord_upper = coord[:split_index + 1]
        values_upper = values[:split_index + 1]
        
        coord_lower = coord[split_index + 1:]
        coord_lower = coord_lower[::-1]
        values_lower = values[split_index + 1:]
        values_lower = values_lower[::-1]
        
        
        if coord_upper[0] > coord_upper[-1]:
            coord_upper = coord_upper[::-1]
            values_upper = values_upper[::-1]
            
        if coord_lower[0] > coord_lower[-1]:
            coord_lower = coord_lower[::-1]
            values_lower = values_lower[::-1]
            
        return (coord_upper, values_upper), (coord_lower, values_lower)
    
    def read(self):
        
        '''
        Read the input files and store the spanwise coordinate and variable values.
        '''
        
        self.files = sorted(glob.glob(os.path.join(self.input_path, '*.csv')))
        
        if not self.files:
            raise FileNotFoundError(f'No files found in {self.input_path}')
        
        N = len(self.files)
        
        chord = np.linspace(0, 1, N) # 0, 1 if extraction is done LE -> TE, otherwise 1, 0 for TE -> LE
                
        if self.chord_length is not None:
            chord = chord * self.chord_length
            
        self.chords = chord
        
        self.upper_surfaces = []
        self.lower_surfaces = []   
        self.all_coord = []
        
        for i, f in enumerate(self.files):
            
            df = pd.read_csv(f)
            
            y = self.chords[i]
            x = pd.to_numeric(df[self.span_col].values, errors='coerce')
            values = pd.to_numeric(df[self.variable_col].values, errors='coerce')
            
            nan_indices = np.where(np.isnan(x))[0]
            
            if self.surface_split is True:
                if len(nan_indices) > 0:
                    
                    split_idex = nan_indices[0]
                    x_upper = x[:split_idex]
                    values_upper = values[:split_idex]
                    coord_upper = self._convert_to_radius(x_upper, y)

                    x_lower = x[split_idex + 1:]
                    values_lower = values[split_idex + 1:]
                    coord_lower = self._convert_to_radius(x_lower, y)
                    
                else:
                    
                    mask = ~np.isnan(x) & ~np.isnan(values)
                    x = x[mask]
                    values = values[mask]
                
                    coord = self._compute_coordinate(x, y)
                    self.all_coord.append(coord)
                    print(f'File {i}/200') 
                    (coord_upper, values_upper), (coord_lower, values_lower) = self._split_surfaces(coord, values)
                
                self.upper_surfaces.append((coord_upper, values_upper))
                self.lower_surfaces.append((coord_lower, values_lower))

            else:
                    
                    mask = ~np.isnan(x) & ~np.isnan(values)
                    x = x[mask]
                    values = values[mask]
                
                    coord = self._compute_coordinate(x, y)
                    self.all_coord.append(coord)
                    print(f'File {i}/200') 
                    
                    self.upper_surfaces.append((coord, values))

            
    def build_coordinate_grid(self):
        
        '''
        Build a common radius grid for all surfaces based on the minimum and maximum radius across all datasets.
        '''
        
        coord_min_list = []
        coord_max_list = []
        
        if self.surface_split is True:
            for (coord_upper, _), (coord_lower, _) in zip(self.upper_surfaces, self.lower_surfaces):
                coord_min_list.append(min(np.min(coord_upper), np.min(coord_lower)))
                coord_max_list.append(max(np.max(coord_upper), np.max(coord_lower)))
            
            #pdb.set_trace()
            
            self.coord_min = np.min(coord_min_list)
            self.coord_max = np.max(coord_max_list)
            
            if self.coord_max < self.coord_min:
                raise ValueError('Invalid coordinate range: coord_max is less than coord_min')
            
            self.coord_target = np.arange(self.coord_min, self.coord_max + self.resolution, self.resolution)
        
        else:
            for (coord_upper, _) in self.upper_surfaces:
                coord_min_list.append(min(coord_upper))
                coord_max_list.append(max(coord_upper))
                        
            self.coord_min = np.min(coord_min_list)
            self.coord_max = np.max(coord_max_list)
            
            if self.coord_max < self.coord_min:
                raise ValueError('Invalid coordinate range: coord_max is less than coord_min')
            
            self.coord_target = np.arange(self.coord_min, self.coord_max + self.resolution, self.resolution)
        

    def interpolate(self):
        
        '''
        Interpolate the variable values onto the common radius grid for both upper and lower surfaces.
        '''
        
        N_chord = len(self.chords)
        N_coord = len(self.coord_target)
        
        if self.surface_split is True:
            self.upper = np.zeros((N_coord, N_chord))
            self.lower = np.zeros((N_coord, N_chord))
            
            for j in range(N_chord):
                
                coord_upper, values_upper = self.upper_surfaces[j]
                coord_lower, values_lower = self.lower_surfaces[j]
                
                interp_upper = interp1d(coord_upper, values_upper, bounds_error=False, fill_value=np.nan)
                interp_lower = interp1d(coord_lower, values_lower, bounds_error=False, fill_value=np.nan)
                
                self.upper[:, j] = interp_upper(self.coord_target)
                self.lower[:, j] = interp_lower(self.coord_target)

        else:
            self.upper = np.zeros((N_coord, N_chord))
            
            for j in range(N_chord):
                
                coord_upper, values_upper = self.upper_surfaces[j]
                
                interp_upper = interp1d(coord_upper, values_upper, bounds_error=False, fill_value=np.nan)
               
                self.upper[:, j] = interp_upper(self.coord_target)
                
    def write_h5(self):
        
        '''
        Write the interpolated datasets to an HDF5 file.
        '''
        
        with h5py.File(self.output_path, 'w') as h5f:
            
            geo = h5f.create_group('Geometry')
            if self.coordinate_system == 'polar':
                geo.create_dataset('Radius', data=self.coord_target)
            else:
                geo.create_dataset('X', data=self.coord_target)
            geo.create_dataset('Chord', data=self.chords)
            
            data = h5f.create_group('Data')
            data.create_dataset('Upper', data=self.upper)
            
            if self.surface_split is True:
                data.create_dataset('Lower', data=self.lower)
            
            else:
                pass
            
    def convert(self):
        
        '''
        Main method to perform the conversion from spanwise to radius-based dataset.
        '''
        
        self.read()
        self.build_coordinate_grid()
        self.interpolate()
        self.write_h5()
