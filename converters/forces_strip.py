import os
import glob
import h5py
import numpy as np
import pandas as pd

class ForcesCSVConverter:
    
    def __init__(self, file_path: str, file_pattern: str, dt: float):
        
        '''
        Parameters
        ----------
        file_path : str
            The path to the input files.
        file_pattern : str
            The pattern to match input files.
        dt : float
            The time step for the forces acquisition.
        '''
        
        self.file_path = file_path
        self.file_pattern = file_pattern
        self.dt = dt
        
        self.color_columns = {
            'Red': 'PerSegmentRedForce[Force:newton]',
            'Green': 'PerSegmentGreenForce[Force:newton]',
            'Blue': 'PerSegmentBlueForce[Force:newton]'
        }
        
        self.files = sorted(glob.glob(os.path.join(file_path, file_pattern)))
        
        if not self.files:
            raise FileNotFoundError(f'No files found matching pattern {file_pattern} in {file_path}')
        
        self.num_files = len(self.files)
        
    def _read_geometry(self):
        
        '''
        Read geometry data from the first file to get node positions.
        A constant geometry is assumed across all files.
        '''
        
        df = pd.read_csv(self.files[0])
        
        self.positions = df['Positions[Length:m]'][:-1].to_numpy()
        self.num_nodes = len(self.positions) - 1
        
    def _detect_component_mapping(self,df):
        
        '''
        Detect which color corresponds to which force component.\n
        A high mean is used to define axial force (Thrust).\n
        A high std is used to define tangential force (Torque).\n
        The remaining component is radial force.\n
        \n
        Assume the same components across all files.
        '''
    
        stats = {}
        
        for color, column in self.color_columns.items():
            values = df[column][:-1].to_numpy()
            stats[color] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }

        # Axial Force (Thrust): Largest mean
        axial_color = max(stats, key=lambda c: abs(stats[c]['mean']))   
        
        # Tangential Force (Torque): Largest std
        remaining_colors = [c for c in stats if c != axial_color]
        tangential_color = max(remaining_colors, key=lambda c: stats[c]['std'])
        
        # Radial Force: Remaining color
        radial_color = [c for c in remaining_colors if c != tangential_color][0]
        
        return {
            axial_color: 'axial',
            tangential_color: 'tangential',
            radial_color: 'radial'
        }
        
    def convert(self, output_filename: str):
        
        '''
        Convert the forces data from CSV files to a stripped HDF5 format.
        '''

        # Get geometry from the first file
        self._read_geometry()
        
        # Mapping definition
        df0 = pd.read_csv(self.files[0])
        self.color_to_force = self._detect_component_mapping(df0)
        
        # H5 file creation
        
        output_path = os.path.join(self.file_path, output_filename)
        
        with h5py.File(output_path, 'w') as h5f:
            
            # Geometry group
            
            geo_group = h5f.create_group('Geometry')
            geo_group.create_dataset('Strip Positions', data=self.positions)
            
            # Time group
            
            time_group = h5f.create_group('Time')
            time = np.arange(0, self.num_files) * self.dt
            time_group.create_dataset('Time', data=time)
            
            # Strips group
            
            strips_group = h5f.create_group('Strips')
            
            for i, z in enumerate(self.positions):
                strip_group = strips_group.create_group(f'Strip_{i:05d}')
                strip_group.attrs['Position'] = z
                
                forces_grp = strip_group.create_group('Forces')
                for force in ['axial', 'tangential', 'radial']:
                    forces_grp.create_dataset(force, shape=(self.num_files,), dtype=np.float64)
                
            # Fill data with timesteps
            
            for t, file in enumerate(self.files):
                df = pd.read_csv(file)
                 
                for i in range(self.num_nodes):
                    for color, col in self.color_columns.items():
                        force_name = self.color_to_force[color]
                        h5f[f'Strips/Strip_{i:05d}/Forces/{force_name}'][t] = df[col].iloc[i]
                