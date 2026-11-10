"""PSD (density, dB/Hz) comparison at 0, +50 and -50 deg from the rotor plane.

For each comparison and every FW-H segment option, one figure per set:
    images/acoustics/three_angles_density/<comparison>_<fwh option>.png
        direct curves: total pressure PSD at the probe
    images/acoustics/three_angles_density_outgoing/<comparison>_<fwh option>.png
        direct curves: outgoing PSD, the total PSD times the 1/3-octave fraction
        of sound flowing out from the rotor (see direct_intensity.py)

Angles: the experiment arc spans +60 to -60 deg with mic column 6 in the
rotor plane; the simulations cover the full sphere with theta = 90 in the plane.

Everything you are likely to change is in the SETTINGS block. The file used
for each case and angle is listed explicitly in COMPARISONS.

Ported from rotaris-docs/Acoustics_Handoff/Codes/psd_three_angles.py (see
HANDOFF_Acoustics.md there for the full theory) - BASE_DIR below is the one
thing that genuinely needs to change for this copy: the original script
resolved it as `Path(__file__).resolve().parents[1]` (one level above its own
`Codes/` folder); now that this lives inside bladeprocessor/acoustics/, that
relative trick no longer points at the actual data tree, so it is spelled out
explicitly instead - update it to wherever UdeS_Case/Comparison/ (or its
replacement) actually lives.
"""
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
from matplotlib import cm
from matplotlib.colors import Normalize

from . import direct_intensity as di

# ============================== SETTINGS ==============================
BASE_DIR = Path('/path/to/UdeS_Case/Comparison')  # TODO: point this at the actual data root
# One figure set per direct-signal treatment: {output folder: (mode, label suffix)}
DIRECT_SETS = {
    'three_angles_density': ('total', ''),
    'three_angles_density_outgoing': ('outgoing', ', outgoing'),
}
OUTPUT_ROOT = BASE_DIR / 'images' / 'acoustics'

# Signal processing
TIME_MIN_DIRECT = 0.04   # s, direct signals before this time are discarded
N_CHUNK_DIRECT = 4       # Welch segment length = record length / n_chunk
ILES_SHIFT = 0           # samples; 0 sums loading + thickness as stored ('aa' = 110 in psdcomparison)
EXP_DATASET = 'ISAE_2/Pressure_6000_Pa'
P_REF = 2e-5             # Pa

# One figure set per FW-H segment option: {file suffix: n_chunk}
FWH_SEGMENTS = {'fwh4seg': 4, 'fwh1seg': 1}

SCALING = 'density'
Y_LABEL = '$PSD \\left[\\frac{dB}{Hz}\\right]$'

# Files per comparison. exp is the mic column of EXP_DATASET.
COMPARISONS = {
    '0deg': dict(title='$0^\\circ$', exp=6, iles='p0.0deg', files={
        '2025': '2025/mic_30_90.txt',
        '2025-T': '2025-T/mic_30_90.txt',
        '2024': '2024/mic_90_90.txt',
        # FW-H at theta = 80 (10 deg off the plane): theta = 90 sits in a sharp broadband null
        'fwh-2025': 'fwh-2025/fwh_calculation.mic_0_80.txt',
        'fwh-2025-T': 'fwh-2025-T/fwh_calculation.mic_0_80.txt',
        '2026DNS': '2026DNS/fwh_computation_antenna.mic_0_80.txt'}),
    'p50deg': dict(title='$+50^\\circ$', exp=1, iles='p50.0deg', files={
        '2025': '2025/mic_30_40.txt',
        '2025-T': '2025-T/mic_30_40.txt',
        '2024': '2024/mic_90_30.txt',
        'fwh-2025': 'fwh-2025/fwh_calculation.mic_0_40.txt',
        'fwh-2025-T': 'fwh-2025-T/fwh_calculation.mic_0_40.txt',
        '2026DNS': '2026DNS/fwh_computation_antenna.mic_0_40.txt'}),
    'm50deg': dict(title='$-50^\\circ$', exp=11, iles='p-50.0deg', files={
        '2025': '2025/mic_30_140.txt',
        '2025-T': '2025-T/mic_30_140.txt',
        '2024': '2024/mic_90_120.txt',
        'fwh-2025': 'fwh-2025/fwh_calculation.mic_0_140.txt',
        'fwh-2025-T': 'fwh-2025-T/fwh_calculation.mic_0_140.txt',
        '2026DNS': '2026DNS/fwh_computation_antenna.mic_0_140.txt'}),
}

# Figure
FIG_SIZE = (15, 4)
DPI = 800
XLIM = (100, 19185)
YLIM = (-10, 80)
USE_LATEX = True
LINE_WIDTH = 1.2
STYLE_DIRECT = '-'
STYLE_FWH = (0, (4, 2))  # dashed
LEGEND_COLUMNS = 4

# The experiment is the reference: black, thicker, drawn on top, first in the legend
EXP_COLOR = 'black'
EXP_LINE_WIDTH = 2.0
ILES_COLOR = '#1B9E77'   # teal green (the cividis yellow was hard to see)

# Same colours as the original figure, plus one for 2026DNS
_cividis = lambda i: cm.cividis(Normalize(vmin=0, vmax=4)(i))
CASES = {   # plotting order (the experiment is drawn last, on top)
    '2025':       dict(kind='direct', label='$2025$ direct', color=_cividis(0)),
    '2025-T':     dict(kind='direct', label='$2025-T$ direct', color=_cividis(1)),
    '2024':       dict(kind='direct', label='$2024$ direct', color=_cividis(2)),
    'fwh-2025':   dict(kind='fwh', label='$2025$ FW-H', color=_cividis(0)),
    'fwh-2025-T': dict(kind='fwh', label='$2025-T$ FW-H', color=_cividis(1)),
    '2026DNS':    dict(kind='fwh', label='$2026$ DNS FW-H', color='#C8102E'),
    'exp':        dict(kind='ref', label='Experiment ISAE', color=EXP_COLOR),
    'iles':       dict(kind='ref', label='iLES Simulation', color=ILES_COLOR),
}
# =======================================================================


def next_greater_power_of_2(x):
    return round(2 ** (x - 1).bit_length())


def to_db(pxx):
    return 10 * np.log10(pxx / P_REF ** 2)


def welch(pressure, fs, nperseg, scaling):
    """Hann-window Welch PSD in dB, zero-padded to the next power of 2 (as fwh_comparison)."""
    f, pxx = sg.welch(pressure, fs=fs, window='hann', nperseg=nperseg,
                      nfft=next_greater_power_of_2(nperseg), scaling=scaling)
    return f, to_db(pxx)


def direct_psd(path, scaling):
    data = pd.read_csv(path, sep=r'\s+')
    time = data['#time'].values
    pressure = data['static_pressure'].values
    keep = time > TIME_MIN_DIRECT
    # As in psdcomparison, the segment is sized from the untrimmed record
    nperseg = int(pressure.size / N_CHUNK_DIRECT)
    return welch(pressure[keep] - pressure[keep].mean(), 1.0 / (time[1] - time[0]),
                 nperseg, scaling)


def direct_outgoing_psd(path):
    """Outgoing PSD (dB/Hz) of a direct probe, same segments as direct_psd."""
    f, s_pp, intensity, r_hat, rho_c = di.probe_spectra(
        path, TIME_MIN_DIRECT, N_CHUNK_DIRECT, next_greater_power_of_2)
    return f, to_db(di.outgoing_psd(f, s_pp, intensity, r_hat, rho_c))


def fwh_psd(path, n_chunk, scaling):
    data = pd.read_csv(path, skiprows=4, sep=r'\s+', header=None)
    time = data[0].values
    pressure = data[1].values
    return welch(pressure, 1.0 / (time[1] - time[0]), int(pressure.size / n_chunk), scaling)


def iles_signal(tag):
    """Loading + thickness pressure of the iLES observer named by tag (e.g. 'p50.0deg')."""
    name = f'Rotor_1_Mic-000_{tag}_t0.0deg_1.62m.dat'
    loading = np.genfromtxt(BASE_DIR / 'Loading_Noise' / f'Acoustic_Pressure_Loading_{name}',
                            skip_header=2)
    thickness = np.genfromtxt(BASE_DIR / 'Thickness_Noise' / f'Acoustic_Pressure_Thickness_{name}',
                              skip_header=2)
    end = thickness.shape[0] - ILES_SHIFT
    pressure = loading[ILES_SHIFT:, 1] + thickness[:end, 1]
    time = thickness[:end, 0]
    return pressure, 1.0 / (time[-1] - time[-2])


def iles_psd(pressure, fs, scaling):
    # One segment covering the whole record, as in psdcomparison
    n = pressure.size
    f, pxx = sg.welch(pressure, fs, 'hann', nperseg=n, noverlap=n // 2, scaling=scaling)
    return f, to_db(pxx)


def exp_psd(pressure, fs, n_iles, scaling):
    # Segment length tied to the iLES record, as in psdcomparison
    f, pxx = sg.welch(pressure, fs=fs, nperseg=int(1.2 * n_iles), noverlap=n_iles // 2,
                      scaling=scaling)
    return f, to_db(pxx)


def read_experiment(columns):
    with h5py.File(BASE_DIR / 'Exp-ISAE' / 'ISAE_2_Time_Data.h5', 'r') as fid:
        data = fid[EXP_DATASET][()]
        time = fid['ISAE_2/time_s']
        fs = 1.0 / float(time[1, 0] - time[0, 0])
    return {c: data[:, c] for c in columns}, fs


def plot(curves, title, path, direct_suffix=''):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    lines, labels = {}, {}
    for key, (f, level) in curves.items():
        case = CASES[key]
        is_exp = key == 'exp'
        labels[key] = case['label'] + (direct_suffix if case['kind'] == 'direct' else '')
        lines[key], = ax.plot(f, level, color=case['color'],
                              lw=EXP_LINE_WIDTH if is_exp else LINE_WIDTH,
                              zorder=10 if is_exp else 2,
                              ls=STYLE_FWH if case['kind'] == 'fwh' else STYLE_DIRECT)
    order = ['exp'] + [k for k in lines if k != 'exp']
    ax.legend([lines[k] for k in order], [labels[k] for k in order],
              prop={'size': 14}, loc='upper right', ncol=LEGEND_COLUMNS)
    ax.set_xscale('log')
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.grid(True, which='both', ls='--')
    ax.set_xlabel('$Frequency \\left[Hz\\right]$')
    ax.set_ylabel(Y_LABEL)
    ax.text(0.01, 0.95, title, transform=ax.transAxes, va='top', fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main():
    plt.rcParams.update({
        'text.usetex': USE_LATEX, 'font.family': 'serif', 'font.serif': ['Computer Modern'],
        'axes.labelsize': 18, 'xtick.labelsize': 18, 'ytick.labelsize': 18,
    })
    print('Reading experiment...')
    exp_signals, exp_fs = read_experiment([c['exp'] for c in COMPARISONS.values()])

    for folder in DIRECT_SETS:
        (OUTPUT_ROOT / folder).mkdir(parents=True, exist_ok=True)
    for name, comp in COMPARISONS.items():
        iles_p, iles_fs = iles_signal(comp['iles'])
        refs = {'exp': exp_psd(exp_signals[comp['exp']], exp_fs, iles_p.size, SCALING),
                'iles': iles_psd(iles_p, iles_fs, SCALING)}
        direct_keys = [k for k, c in CASES.items() if c['kind'] == 'direct']
        direct = {
            'total': {k: direct_psd(BASE_DIR / comp['files'][k], SCALING) for k in direct_keys},
            'outgoing': {k: direct_outgoing_psd(BASE_DIR / comp['files'][k]) for k in direct_keys},
        }
        for suffix, n_chunk in FWH_SEGMENTS.items():
            fwh = {k: fwh_psd(BASE_DIR / comp['files'][k], n_chunk, SCALING)
                   for k, c in CASES.items() if c['kind'] == 'fwh'}
            for folder, (mode, label_suffix) in DIRECT_SETS.items():
                curves = {k: {**direct[mode], **fwh, **refs}[k] for k in CASES}
                path = OUTPUT_ROOT / folder / f'{name}_{suffix}.png'
                plot(curves, comp['title'], path, label_suffix)
                print(f'saved {path.relative_to(BASE_DIR)}')

if __name__ == '__main__':
    main()
