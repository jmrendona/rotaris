import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,           # Usa LaTeX real (requiere instalación)
    "font.family": "serif",        # Usa fuentes serif (estilo LaTeX)
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 12,          # Tamaño de etiquetas de ejes
    "xtick.labelsize": 10,         # Tamaño de los números en X
    "ytick.labelsize": 10          # Tamaño de los números en Y
})

class BladePostProcessor:
    
    def __init__(self, filename: str, rpm: int = None, pref: float = None, rho_ref: float = None):
        
        '''
        Parameters
        ----------
        filename : str
            The path to the HDF5 file containing blade data.
        '''
        
        self.filename = filename
        self.rpm = rpm
        self.pref = pref
        self.rho_ref = rho_ref
        self.data = {}
        self._load_data()
        
    def _load_data(self):
        
        '''
        Load data from the HDF5 file into a dictionary for easy access.
        '''
        
        with h5py.File(self.filename, 'r') as f:
            
            self.radius = f['Geometry/Radius'][:]
            self.chord = f['Geometry/Chord'][:]
            self.data['Upper'] = f['Data/Upper'][:]
            self.data['Lower'] = f['Data/Lower'][:]

        if self.radius is None or self.chord is None:
            raise ValueError("HDF5 must contain 'Radius' and 'Chord' datasets in the 'Geometry' group.")
        
    def get_radius_index(self, r_target: float) -> int:
        
        '''
        Get the index of the radius closest to the target radius.
        
        Parameters
        ----------
        r_target : float
            The target radius for which to find the closest index.
        
        Returns
        -------
        int
            The index of the closest radius in the dataset.
        '''
        
        return np.argmin(np.abs(self.radius - r_target))
    
    def omega(self):
        
        '''
        Calculate the angular velocity in radians per second based on the RPM.
        
        Returns
        -------
        float
            The angular velocity in radians per second.
        '''
        
        return self.rpm * 2 * np.pi / 60
    
    def Uref(self, r):
        
        '''
        Calculate the reference velocity at the blade tip based on the angular velocity and radius.
        
        Returns
        -------
        float
            The reference velocity at the blade tip.
        '''
        
        return self.omega() * r
    
    def compute_cp(self, p, r):
        
        '''
        Compute the pressure coefficient (Cp) at a given radius based on the pressure data.
        
        Parameters
        ----------
        p : array-like
            The pressure distribution along the blade.
        r : float
            The radius at which to compute Cp.
        
        Returns
        -------
        float
            The computed pressure coefficient (Cp) at the given radius.
        '''
        
        U_ref = self.Uref(r)
        return (p - self.pref) / (0.5 * self.rho_ref * U_ref**2)
        
    def compute_cf(self, tau, r):
        
        '''
        Compute the skin friction coefficient (Cf) at a given radius based on the shear stress data.
        
        Parameters
        ----------
        tau : array-like
            The shear stress distribution along the blade.
        r : float
            The radius at which to compute Cf.
        
        Returns
        -------
        float
            The computed skin friction coefficient (Cf) at the given radius.
        '''
        
        U_ref = self.Uref(r)
        return tau / (0.5 * self.rho_ref * U_ref**2)
    
    def _get_curve(self, radius_index: int):
        
        '''
        Get the curve data for a specific variable at a given radius index.
        
        Parameters
        ----------
        variable : str
            The name of the variable to retrieve (e.g., 'Cp', 'Cf').
        radius_index : int
            The index of the radius for which to retrieve the curve data.
        
        Returns
        -------
        array-like
            The curve data for the specified variable and radius index.
        '''
        
        if 'Upper' not in self.data or 'Lower' not in self.data:
            raise ValueError(f"Variable not found in .h5 with structure 'Data/Upper' and 'Data/Lower'.")
        
        return self.data['Upper'][radius_index], self.data['Lower'][radius_index]
    
    def plot_radii(self, var_name, radii = None, idx_list = None, mode = 'raw'):
        
        '''
        Plot the specified variable for the given radii.
        
        Parameters
        ----------
        var_name : str
            The name of the variable to plot (e.g., 'Cp', 'Cf').
        radii : list of float, optional
            A list of radii to plot. If None, default radii will be used.
        idx_list : list of int, optional
            A list of indices corresponding to the radii to plot. If None, indices will be determined from radii.
        mode : str, optional
            The mode of plotting ('raw' for raw data, 'cp' for pressure coefficient, and 'cf' for skin friction coefficient). Default is 'raw'.
        '''
        
        if radii is not None:
            idx_list = [self.get_radius_index(r) for r in radii]
        
        if idx_list is None:
            raise ValueError("Either 'radii' or 'idx_list' must be provided.")
        
        plt.figure(figsize=(10, 6))
        
        for idx in idx_list:
            
            r_local = self.radius[idx]
            upper_curve, lower_curve = self._get_curve(idx)
            
            if mode == 'cp':
                
                if self.pref is None or self.rho_ref is None or self.rpm is None:
                    raise ValueError("To compute Cp, 'pref', 'rho_ref', and 'rpm' must be provided.")
                
                upper_curve = self.compute_cp(upper_curve, r_local)
                lower_curve = self.compute_cp(lower_curve, r_local)
                ylabel = '$-C_p$'
                
            elif mode == 'cf':
                
                if self.rho_ref is None or self.rpm is None:
                    raise ValueError("To compute Cf, 'rho_ref' and 'rpm' must be provided.")
                
                upper_curve = self.compute_cf(upper_curve, r_local)
                lower_curve = self.compute_cf(lower_curve, r_local)
                ylabel = '$C_f$'
                
            else:
                ylabel = var_name
            
            plt.plot(upper_curve, label=f'r={r_local:.3f} m', linestyle='*')
            plt.plot(lower_curve, linestyle='*')
        
        plt.xlabel('Chord $x/c$')
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()