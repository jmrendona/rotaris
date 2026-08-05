import os
import pdb
import glob
import h5py
import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class SpanConverter:
    
    '''
    Convert spanwise alligned sliced data to cylindrical (r, chord) dataset.
    '''
    
    def __init__(self, input_path: str, output_path: str, variable_col: str, span_col: str, chord_length=None, resolution=0.01, coordinate_system='polar', surface_split=True, hub_radius=None, tip_radius=None):

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
        hub_radius, tip_radius : float, optional
            Known physical hub/tip radius, used only to sanity-check each
            file's upper/lower split (warns if a split looks wrong). Not
            used to perform the split itself.
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
        self.hub_radius = hub_radius
        self.tip_radius = tip_radius
       
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
     
    def _find_turn_index(self, coord: np.ndarray) -> int:

        '''
        Find the interior turning point of a hub<->tip<->hub loop trace.

        A clean loop starts and ends near the SAME location (wherever the
        extraction began), and passes through the opposite extreme exactly
        once, somewhere in the middle. So whichever of argmax/argmin sits
        strictly inside the array (not at index 0 or the last index) is
        the true turn: a tip if the trace starts/ends at the hub, or a hub
        if the trace starts/ends at the tip.

        Returns
        -------
        int
            Index of the turning point.
        '''

        n = len(coord)
        i_max, i_min = int(np.argmax(coord)), int(np.argmin(coord))

        max_interior = 0 < i_max < n - 1
        min_interior = 0 < i_min < n - 1

        if max_interior and not min_interior:
            return i_max
        if min_interior and not max_interior:
            return i_min

        raise ValueError(
            "Could not find an unambiguous interior turning point "
            f"(argmax at {i_max}, argmin at {i_min}, out of {n} points)."
        )

    def _validate_split(self, coord_upper: np.ndarray, coord_lower: np.ndarray):

        '''
        Warn if the split surfaces don't roughly span [hub_radius, tip_radius].
        Only runs if both bounds were provided; used as a safety net to
        catch a bad split, not to perform the split itself.
        '''

        if self.hub_radius is None or self.tip_radius is None:
            return

        tol = 0.05 * (self.tip_radius - self.hub_radius)

        for label, c in (('upper', coord_upper), ('lower', coord_lower)):
            if abs(c.min() - self.hub_radius) > tol or abs(c.max() - self.tip_radius) > tol:
                warnings.warn(
                    f"{label} surface spans [{c.min():.5g}, {c.max():.5g}], expected roughly "
                    f"[{self.hub_radius:.5g}, {self.tip_radius:.5g}]. Possible bad split for this file."
                )

    def _check_start_side_consistency(self, filename: str, coord: np.ndarray):

        '''
        Track which end of the coordinate range each file's trace starts
        from (near the max or near the min), and warn as soon as a file
        breaks the pattern set by the first file. Assumes the extraction
        method is the same for every file in a batch, so the start side
        should never flip within one conversion run.
        '''

        starts_high = abs(coord[0] - coord.max()) < abs(coord[0] - coord.min())

        if not hasattr(self, 'start_sides'):
            self.start_sides = []

        if self.start_sides and starts_high != self.start_sides[-1][1]:
            prev_filename, _ = self.start_sides[-1]
            warnings.warn(
                f"'{filename}' starts at the opposite end of the coordinate range compared to "
                f"'{prev_filename}'. Extraction direction may not be consistent across files."
            )

        self.start_sides.append((filename, starts_high))

    def _split_surfaces(self, coord: np.ndarray, values: np.ndarray):

        '''
        Split a single hub<->tip<->hub loop trace into two surfaces, each
        sorted ascending by coord, regardless of which end the trace
        starts from.
        '''

        split_index = self._find_turn_index(coord)

        coord_a, values_a = coord[:split_index + 1], values[:split_index + 1]
        coord_b, values_b = coord[split_index + 1:], values[split_index + 1:]

        order_a = np.argsort(coord_a)
        order_b = np.argsort(coord_b)

        coord_upper, values_upper = coord_a[order_a], values_a[order_a]
        coord_lower, values_lower = coord_b[order_b], values_b[order_b]

        self._validate_split(coord_upper, coord_lower)

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

            # '*,*' (or any unparsable entry) means BOTH columns are invalid
            # for that row, never a real upper/lower separator - drop those
            # rows before any split logic runs.
            mask = ~np.isnan(x) & ~np.isnan(values)
            x = x[mask]
            values = values[mask]

            coord = self._compute_coordinate(x, y)
            self.all_coord.append(coord)
            print(f'File {i + 1}/{N}')

            if self.surface_split:
                self._check_start_side_consistency(f, coord)
                (coord_upper, values_upper), (coord_lower, values_lower) = self._split_surfaces(coord, values)
                self.upper_surfaces.append((coord_upper, values_upper))
                self.lower_surfaces.append((coord_lower, values_lower))
            else:
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
