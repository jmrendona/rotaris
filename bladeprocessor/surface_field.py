import os
import warnings
import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
})


class SurfaceField:

    '''
    A single scalar field (e.g. Cp, Cf, wall shear magnitude) mapped onto a
    blade surface, as exported to Rotaris' "radius" HDF5 layout:

        Geometry/Radius or Geometry/X : (n_radius,)
            Physical radial stations [m].
        Geometry/Chord : (n_chord,)
            Physical chordwise coordinate [m], shared by every radial
            station, starting at 0 (leading edge).
        Data/Upper, Data/Lower : (n_radius, n_chord)
            The field itself. Entries beyond a station's local chord length
            are NaN, since Chord is a single fixed-length axis but each
            radial station's true chord can be shorter.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    var_name : str, optional
        Label for the field (used in plot titles/colorbars). Defaults to
        the filename without extension.
    r_tip : float, optional
        Physical tip radius [m] used to normalize radius as r/R. If not
        provided, max(radius) from this file is used, which may not match
        the true rotor tip radius and can misalign comparisons across
        cases with different meshes.
    '''

    def __init__(self, filename: str, var_name: str = None, r_tip: float = None):

        self.filename = filename
        self.var_name = var_name or os.path.splitext(os.path.basename(filename))[0]
        self.r_tip = r_tip
        self.radius = None
        self.chord = None
        self.data = {}
        self._load()

    def _load(self):

        '''
        Load radius, chord and per-surface data arrays from the HDF5 file.
        '''

        with h5py.File(self.filename, 'r') as f:

            geo = f['Geometry']
            radius_key = 'Radius' if 'Radius' in geo else 'X'

            self.radius = geo[radius_key][:]
            self.chord = geo['Chord'][:]
            self.data = {surf: f[f'Data/{surf}'][:] for surf in f['Data'].keys()}

        if self.radius is None or self.chord is None:
            raise ValueError("HDF5 must contain a radial axis ('Radius' or 'X') and 'Chord' under 'Geometry'.")

    def local_chord_length(self, surface: str = 'Upper'):

        '''
        Estimate the local chord length at each radial station as the
        chordwise coordinate of the last non-NaN sample in that row.

        Parameters
        ----------
        surface : str
            'Upper' or 'Lower'.

        Returns
        -------
        np.ndarray, shape (n_radius,)
            Local chord length [m] per station, NaN where the row has no
            valid data.
        '''

        d = self.data[surface]
        n_valid = np.sum(~np.isnan(d), axis=1)

        local_chord = np.full(d.shape[0], np.nan)
        has_data = n_valid > 0
        local_chord[has_data] = self.chord[n_valid[has_data] - 1]

        return local_chord

    def r_over_R(self):

        '''
        Normalize the radial axis by the tip radius.

        Returns
        -------
        np.ndarray, shape (n_radius,)
            r/R for each radial station.
        '''

        r_tip = self.r_tip

        if r_tip is None:
            r_tip = np.nanmax(self.radius)
            warnings.warn(
                f"'{self.var_name}': r_tip not provided, using max(radius)={r_tip:.6g} m. "
                "Pass r_tip explicitly to keep r/R consistent across cases with different meshes."
            )

        return self.radius / r_tip

    def physical_aspect(self):
        
        '''
        Compute the physical aspect ratio of the blade surface, defined as
        the ratio of the maximum chord length to the tip radius.

        Returns
        -------
        float
            Physical aspect ratio (max chord / tip radius).
        '''

        r_tip = self.r_tip if self.r_tip is not None else np.nanmax(self.radius)
        max_chord = np.nanmax(self.local_chord_length('Upper'))  # Assuming upper surface for max chord

        return max_chord / r_tip

    def to_common_grid(self, r_target: np.ndarray, x_target: np.ndarray, surface: str = 'Upper'):

        '''
        Resample this field onto a shared (r/R, x/c) grid so it can be
        compared or differenced against another case with a different
        native mesh.

        Each radial station is first normalized in the chordwise direction
        (x/c, using that station's own local chord length) and resampled
        onto x_target, then the result is resampled across stations onto
        r_target using r/R.

        Parameters
        ----------
        r_target : array-like
            Target r/R values, increasing.
        x_target : array-like
            Target x/c values, increasing.
        surface : str
            'Upper' or 'Lower'.

        Returns
        -------
        np.ndarray, shape (len(r_target), len(x_target))
            Resampled field, NaN outside the covered region.
        '''

        d = self.data[surface]
        local_chord = self.local_chord_length(surface)
        r_over_R = self.r_over_R()

        chordwise = np.full((d.shape[0], len(x_target)), np.nan)

        for i in range(d.shape[0]):

            lc = local_chord[i]
            if not np.isfinite(lc) or lc <= 0:
                continue

            row = d[i]
            valid = ~np.isnan(row)
            if valid.sum() < 2:
                continue

            xc_valid = self.chord[valid] / lc
            chordwise[i] = np.interp(x_target, xc_valid, row[valid], left=np.nan, right=np.nan)

        out = np.full((len(r_target), len(x_target)), np.nan)

        row_has_data = np.isfinite(r_over_R) & (np.sum(~np.isnan(chordwise), axis=1) > 0)
        order = np.argsort(r_over_R[row_has_data])
        r_valid = r_over_R[row_has_data][order]
        chordwise_valid = chordwise[row_has_data][order]

        for j in range(len(x_target)):

            col = chordwise_valid[:, j]
            valid = ~np.isnan(col)
            if valid.sum() < 2:
                continue

            out[:, j] = np.interp(r_target, r_valid[valid], col[valid], left=np.nan, right=np.nan)

        return out

    def plot_contour(self, surface: str = 'Upper', normalize: bool = True, n_r: int = 200, n_c: int = 200,
                      ax=None, levels=100, cmap: str = 'viridis', cbar_label: str = None,
                      title: str = None, savepath: str = None, dpi: int = 600):

        '''
        Plot a filled contour of the field over the blade surface.

        Parameters
        ----------
        surface : str
            'Upper' or 'Lower'.
        normalize : bool
            If True, plot on a common (r/R, x/c) grid via to_common_grid.
            If False, plot on this case's native (Radius, Chord) axes.
        n_r, n_c : int
            Resolution of the normalized grid (only used if normalize=True).
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. A new figure/axes is created if omitted.
        levels : int or array-like
            Passed to contourf.
        cmap : str
            Colormap name.
        cbar_label, title : str, optional
        savepath : str, optional
            If given, the figure is saved here.
        dpi : int

        Returns
        -------
        (fig, ax)
        '''

        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 7))
        else:
            fig = ax.figure

        if normalize:
            r_axis = np.linspace(0, 1, n_r)
            x_axis = np.linspace(0, 1, n_c)
            field = self.to_common_grid(r_axis, x_axis, surface)
            xlabel, ylabel = r'$r/R$', r'$x/c$'
        else:
            r_axis = self.radius
            x_axis = self.chord
            field = self.data[surface]
            xlabel, ylabel = r'Radius [m]', r'Chord [m]'

        contour = ax.contourf(r_axis, x_axis, field.T, levels=levels, cmap=cmap, extend='both')
        ax.set_aspect('equal' if not normalize else self.physical_aspect(), adjustable='box')
        cbar = fig.colorbar(contour, ax=ax, orientation='horizontal', pad=0.15)
        cbar.set_label(cbar_label or self.var_name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title:
            ax.set_title(title)

        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=dpi)

        return fig, ax


class SurfaceFieldComparator:

    '''
    Compare the same field across multiple SurfaceField cases on a common
    normalized (r/R, x/c) grid, and compute deltas between any two of them.

    Parameters
    ----------
    cases : dict[str, SurfaceField]
        Mapping of case label -> SurfaceField instance.
    n_r, n_c : int
        Resolution of the shared normalized grid.
    '''

    def __init__(self, cases: dict, n_r: int = 200, n_c: int = 200):

        self.cases = cases
        self.r_target = np.linspace(0, 1, n_r)
        self.x_target = np.linspace(0, 1, n_c)
        self._cache = {}

    def field(self, name: str, surface: str = 'Upper'):

        '''
        Return (and cache) the given case resampled onto the shared grid.
        '''

        key = (name, surface)

        if key not in self._cache:
            self._cache[key] = self.cases[name].to_common_grid(self.r_target, self.x_target, surface)

        return self._cache[key]

    def delta(self, name_a: str, name_b: str, surface: str = 'Upper'):

        '''
        Return field(name_a) - field(name_b) on the shared grid.
        '''

        return self.field(name_a, surface) - self.field(name_b, surface)

    def plot_cases(self, surface: str = 'Upper', reference: str = None, levels=100, cmap: str = 'viridis',
                   cbar_label: str = None, savepath: str = None, dpi: int = 600):

        '''
        Plot every case side by side on a shared color scale for direct
        visual comparison.

        Parameters
        ----------
        reference : str, optional
            Case label whose physical_aspect() sets the (r/R, x/c) aspect
            ratio for all panels. Defaults to the first case. Only exact
            for cases that share the same chord/radius geometry (e.g.
            geometrically scaled rotors), which is the intended use here.
        '''

        names = list(self.cases.keys())
        n = len(names)
        reference = reference or names[0]
        aspect = self.cases[reference].physical_aspect()

        fig, axes = plt.subplots(1, n, figsize=(6 * n, 3), sharey=True, squeeze=False)
        axes = axes[0]

        fields = {name: self.field(name, surface) for name in names}
        vmin = min(np.nanmin(f) for f in fields.values())
        vmax = max(np.nanmax(f) for f in fields.values())
        contour_levels = np.linspace(vmin, vmax, levels) if isinstance(levels, int) else levels

        contour = None
        for ax, name in zip(axes, names):
            contour = ax.contourf(self.r_target, self.x_target, fields[name].T, levels=contour_levels, cmap=cmap, extend='both')
            ax.set_aspect(aspect, adjustable='box')
            ax.set_xlabel(r'$r/R$')
            ax.set
            ax.set_title(name)

        axes[0].set_ylabel(r'$x/c$')
        cbar = fig.colorbar(contour, ax=list(axes), orientation='horizontal' , pad=0.4)
        cbar.set_label(cbar_label or self.cases[names[0]].var_name)

        if savepath:
            fig.savefig(savepath, dpi=dpi)

        return fig, axes

    def plot_delta(self, name_a: str, name_b: str, surface: str = 'Upper', reference: str = None, levels=100,
                   cmap: str = 'RdBu_r', symmetric: bool = True, cbar_label: str = None, title: str = None,
                   ax=None, savepath: str = None, dpi: int = 600):

        '''
        Plot the delta field (name_a - name_b) on the shared normalized grid.

        Parameters
        ----------
        reference : str, optional
            Case label whose physical_aspect() sets the (r/R, x/c) aspect
            ratio. Defaults to name_a.
        '''

        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 7))
        else:
            fig = ax.figure

        reference = reference or name_a
        aspect = self.cases[reference].physical_aspect()

        d = self.delta(name_a, name_b, surface)

        if symmetric:
            vmax = np.nanmax(np.abs(d))
            vmin = -vmax
        else:
            vmin, vmax = np.nanmin(d), np.nanmax(d)

        contour_levels = np.linspace(vmin, vmax, levels) if isinstance(levels, int) else levels

        contour = ax.contourf(self.r_target, self.x_target, d.T, levels=contour_levels, cmap=cmap, extend='both')
        ax.set_aspect(aspect, adjustable='box')
        cbar = fig.colorbar(contour, ax=ax, orientation='horizontal', pad=0.015)
        cbar.set_label(cbar_label or f'$\\Delta$ ({name_a} $-$ {name_b})')
        ax.set_xlabel(r'$r/R$')
        ax.set_ylabel(r'$x/c$')

        if title:
            ax.set_title(title)

        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=dpi)

        return fig, ax
