"""Directivity (+60 to -60 deg from the rotor plane) in the four bands of the intensity maps.

Writes to images/acoustics/directivity/:
  directivity.png            direct LBM curves as recorded (total pressure)
  directivity_outgoing.png   direct LBM curves corrected with the band fraction
  directivity_values.csv     every value plotted (dB)

Levels
  BPF, 2 BPF        tone SPL: peak of the spectrum in the band times the window's
                    equivalent noise bandwidth, i.e. the tone's mean-square
                    pressure (independent of each case's frequency resolution)
  1-5, 5-10 kHz     band OASPL: PSD (density) integrated over the band
  outgoing direct   level + 10 log10(F), F = rho c sum(I_r) / sum(S_pp) over the band

Angles (elevation e, + towards the thrust side, 0 in the rotor plane)
  Experiment        column (60 - e) / 10 of the 6000 rpm dataset
  iLES              file p<e>deg
  LBM direct, FW-H  mic_0_theta with theta = 90 - e (the polar sweep through the
                    rotor axis); 2024 only has in-plane mics, so it is in the CSV
                    at 0 but not plotted
  outgoing direct   left out above OUTGOING_FMAX (the 5-10 kHz panel shows no direct curve)

Ported from rotaris-docs/Acoustics_Handoff/Codes/directivity.py - see
HANDOFF_Acoustics.md there for the full theory.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
from matplotlib.transforms import Bbox

from . import direct_intensity as di
from . import psd_three_angles as pta

# ============================== SETTINGS ==============================
OUTPUT_DIR = pta.BASE_DIR / 'images' / 'acoustics' / 'directivity'
ELEVATIONS = list(range(60, -61, -10))
BANDS = {   # title: (low, high, kind)
    'BPF': (170, 230, 'tone'),
    '2 BPF': (370, 430, 'tone'),
    'OASPL 1--5 kHz': (1000, 5000, 'band'),
    'OASPL 5--10 kHz': (5000, 10000, 'band'),
}
DIRECT_FILES = {   # case: file pattern, or {elevation: file} for fixed mics
    '2025': '2025/mic_0_{theta}.txt',
    '2025-T': '2025-T/mic_0_{theta}.txt',
    '2024': {0: '2024/mic_90_90.txt'},
}
FWH_FILES = {
    'fwh-2025': 'fwh-2025/fwh_calculation.mic_0_{theta}.txt',
    'fwh-2025-T': 'fwh-2025-T/fwh_calculation.mic_0_{theta}.txt',
    '2026DNS': '2026DNS/fwh_computation_antenna.mic_0_{theta}.txt',
}
FWH_N_CHUNK = 4
# Tone levels of 2026DNS from one segment: with 4 chunks (df ~ 221 Hz) the BPF
# tone leaks into the 2 BPF band
FWH_TONE_N_CHUNK = {'2026DNS': 1}
# The outgoing correction is valid where the probe signals pass the sound test
OUTGOING_FMAX = 6000
ANGLE_LIMITS = (-70, 70)
# Curves drawn (2024 is computed and kept in the CSV, but has only an in-plane mic)
PLOT_ORDER = ['exp', 'iles', '2025', '2025-T', 'fwh-2025', 'fwh-2025-T', '2026DNS']
# =======================================================================


def welch_density(p, fs, nperseg, noverlap=None, nfft=None):
    """Hann Welch PSD plus the factor that turns a density peak into a tone power (Hz)."""
    f, s = sg.welch(p, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap,
                    nfft=nfft, scaling='density')
    w = sg.get_window('hann', nperseg)
    return f, s, fs * np.sum(w ** 2) / np.sum(w) ** 2


def level(f, s, enbw, fs, low, high, kind):
    """Tone SPL or band OASPL in dB re 20 uPa; NaN if the band exceeds Nyquist."""
    if high > fs / 2:
        return np.nan
    m = (f >= low) & (f < high)
    power = s[m].max() * enbw if kind == 'tone' else s[m].sum() * (f[1] - f[0])
    return 10 * np.log10(power / pta.P_REF ** 2)


def all_levels(f, s, enbw, fs):
    return {b: level(f, s, enbw, fs, lo, hi, kind) for b, (lo, hi, kind) in BANDS.items()}


def direct_levels(path):
    """Total and outgoing levels of a direct probe (same segments as the PSD figures)."""
    p, u, fs, length, rho_c = di.read_probe(path, pta.TIME_MIN_DIRECT)
    nperseg = int(length / pta.N_CHUNK_DIRECT)
    f, s, intensity = di.spectra(p, u, fs, nperseg, pta.next_greater_power_of_2(nperseg))
    _, _, enbw = welch_density(p, fs, nperseg, nfft=pta.next_greater_power_of_2(nperseg))
    i_r = di.unit(di.mic_position(path)) @ intensity
    total = all_levels(f, s, enbw, fs)
    outgoing = {}
    for b, (lo, hi, _) in BANDS.items():
        m = (f >= lo) & (f < hi)
        fraction = rho_c * i_r[m].sum() / s[m].sum()
        valid = fraction > 0 and hi <= OUTGOING_FMAX
        outgoing[b] = total[b] + 10 * np.log10(fraction) if valid else np.nan
    return total, outgoing


def fwh_levels(path, tone_n_chunk=FWH_N_CHUNK):
    data = pd.read_csv(path, skiprows=4, sep=r'\s+', header=None)
    fs = 1.0 / (data[0].values[1] - data[0].values[0])
    out = {}
    for n_chunk, kinds in [(FWH_N_CHUNK, ('band',)), (tone_n_chunk, ('tone',))]:
        nperseg = int(data.shape[0] / n_chunk)
        spectrum = welch_density(data[1].values, fs, nperseg,
                                 nfft=pta.next_greater_power_of_2(nperseg))
        out.update({b: level(*spectrum, fs, lo, hi, kind)
                    for b, (lo, hi, kind) in BANDS.items() if kind in kinds})
    return out


def collect():
    """DataFrame: rows (source, elevation), columns bands; plus outgoing direct rows."""
    exp_columns = {e: (60 - e) // 10 for e in ELEVATIONS}
    exp_signals, exp_fs = pta.read_experiment(list(exp_columns.values()))
    rows = []
    for e in ELEVATIONS:
        theta = 90 - e
        iles_p, iles_fs = pta.iles_signal(f'p{float(e):.1f}deg')
        n = iles_p.size
        rows.append(dict(source='iles', elevation=e,
                         **all_levels(*welch_density(iles_p, iles_fs, n, n // 2), iles_fs)))
        rows.append(dict(source='exp', elevation=e,
                         **all_levels(*welch_density(exp_signals[exp_columns[e]], exp_fs,
                                                     int(1.2 * n), n // 2), exp_fs)))
        for key, pattern in FWH_FILES.items():
            path = pta.BASE_DIR / pattern.format(theta=theta)
            if path.exists():
                rows.append(dict(source=key, elevation=e,
                                 **fwh_levels(path, FWH_TONE_N_CHUNK.get(key, FWH_N_CHUNK))))
        for key, pattern in DIRECT_FILES.items():
            name = pattern.get(e) if isinstance(pattern, dict) else pattern.format(theta=theta)
            if name and (pta.BASE_DIR / name).exists():
                total, outgoing = direct_levels(pta.BASE_DIR / name)
                rows.append(dict(source=key, elevation=e, **total))
                rows.append(dict(source=key + ' outgoing', elevation=e, **outgoing))
        print(f'elevation {e:+d} done')
    return pd.DataFrame(rows)


def plot(df, outgoing, path):
    direct_suffix = ' outgoing' if outgoing else ''
    order = PLOT_ORDER
    fig, axes = plt.subplots(1, len(BANDS), figsize=(13, 6.2), subplot_kw={'projection': 'polar'})
    for ax, band in zip(axes.flat, BANDS):
        values = []
        for key in order:
            case = pta.CASES[key]
            source = key + direct_suffix if case['kind'] == 'direct' else key
            d = df[df.source == source].sort_values('elevation')
            if d.empty:
                continue
            label = case['label'] + (', outgoing' if outgoing and case['kind'] == 'direct' else '')
            is_exp = key == 'exp'
            ax.plot(np.radians(d.elevation), d[band], color=case['color'], label=label,
                    ls=pta.STYLE_FWH if case['kind'] == 'fwh' else '-',
                    lw=pta.EXP_LINE_WIDTH if is_exp else 1.3, zorder=10 if is_exp else 2,
                    marker='s' if key == '2024' else 'o', ms=7 if key == '2024' else 4)
            values.extend(d[band].dropna())
        ax.set_thetamin(ANGLE_LIMITS[0])
        ax.set_thetamax(ANGLE_LIMITS[1])
        ax.set_thetagrids(range(-60, 61, 30))
        low, high = np.floor(min(values) / 5) * 5 - 5, np.ceil(max(values) / 5) * 5 + 5
        ax.set_rlim(low, high)
        ax.set_title(band + ' $\\left[dB\\right]$', fontsize=17, pad=6)
        ax.grid(True, ls='--')
    # The half-polar wedges fill only the left part of their boxes: overlap the boxes
    fig.subplots_adjust(left=0.04, right=1.25, top=0.9, bottom=0.2, wspace=-0.45)
    # Centre the legend under the drawn wedges, not under the (wider) figure
    renderer = fig.canvas.get_renderer()
    drawn = Bbox.union([ax.get_tightbbox(renderer) for ax in axes.flat])
    drawn = drawn.transformed(fig.transFigure.inverted())
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, prop={'size': 13},
               bbox_to_anchor=((drawn.x0 + drawn.x1) / 2, 0.02))
    fig.savefig(path, dpi=pta.DPI, bbox_inches='tight')
    plt.close(fig)


def main():
    plt.rcParams.update({
        'text.usetex': pta.USE_LATEX, 'font.family': 'serif', 'font.serif': ['Computer Modern'],
        'axes.labelsize': 18, 'xtick.labelsize': 15, 'ytick.labelsize': 14,
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(OUTPUT_DIR / 'directivity_values.csv', index=False, float_format='%.2f')
    plot(df, False, OUTPUT_DIR / 'directivity.png')
    plot(df, True, OUTPUT_DIR / 'directivity_outgoing.png')
    print('saved', OUTPUT_DIR.relative_to(pta.BASE_DIR))


if __name__ == '__main__':
    main()
