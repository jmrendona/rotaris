'''
Far-field acoustics post-processing - re-runs the diagnostics ported from
rotaris-docs/Acoustics_Handoff into bladeprocessor/acoustics/ (see that
package's own __init__.py and rotaris-docs/Acoustics_Handoff/HANDOFF_Acoustics.md
for the full theory, notation and processing parameters).

Unlike every other *_manager.py here, this one does NOT go through
case_config.load_case_config()/cases/*.yaml: the acoustic analysis compares
several sources at once (direct LBM probes from multiple cases, FW-H, iLES,
experiment) rather than operating on one case's surface-field export, so its
settings (file layout, which case/angle/source goes where) live in
bladeprocessor/acoustics/psd_three_angles.py's CASES/COMPARISONS dicts
instead - same shape as manager_comparison.py's own CASES dict, for the same
reason.

BASE_DIR (bladeprocessor/acoustics/psd_three_angles.py) is still the
TODO placeholder path from the port - point it at wherever the
UdeS_Case/Comparison/ data tree (or its replacement) actually lives before
running any of this.
'''

from bladeprocessor.acoustics import (
    psd_three_angles,
    intensity_maps,
    directivity,
)


# ------------- Far-field PSD at 0 / +50 / -50 deg (direct, FW-H, iLES, experiment) ------------- #
# Writes images/acoustics/three_angles_density/<comparison>_<fwh option>.png (direct: total
# pressure) and images/acoustics/three_angles_density_outgoing/... (direct: outgoing-corrected).

# psd_three_angles.main()


# ------------- Acoustic intensity: maps, fraction spectra, near-mic test, -------------------- #
# ------------- acoustic-validity checks (energy ratio), Riemann wave split ------------------- #
# Writes to images/acoustics/intensity/ - see intensity_maps.py's own module docstring for the
# full list of figures each of these produces.

# intensity_maps.main()

# Individual pieces, if you don't want the whole batch in intensity_maps.main():
# intensity_maps.plot_map('2025-T')                 # -> images/acoustics/intensity/map_2025-T.png
# intensity_maps.plot_energy_ratio()                 # -> images/acoustics/intensity/energy_ratio.png
# intensity_maps.plot_wave_split()                   # -> images/acoustics/intensity/wave_split.png
# intensity_maps.plot_riemann('2025-T')              # -> images/acoustics/intensity/riemann_2025-T.png


# ------------- Directivity: BPF / 2 BPF tone level and band OASPL (1-5, 5-10 kHz) ------------- #
# Writes images/acoustics/directivity/{directivity.png, directivity_outgoing.png,
# directivity_values.csv}.

# directivity.main()


if __name__ == '__main__':
    pass  # uncomment the calls above for what you want to (re-)generate
