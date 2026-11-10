"""Figures showing where the sound at the direct mics is travelling.

Writes to images/acoustics/intensity/:
  map_<case>.png        Meridian plane (rotor axis y vertical). One arrow per mic of
                        the mic_0_theta ring: direction of the net acoustic energy
                        flow; colour and size = fraction in dB. One panel per band.
  fraction_spectra.png  Fraction and pointing error per 1/3 octave at the three
                        comparison angles (bars: estimates from each half of the record).
  near_mic_test.png     Level of near_mic.txt minus the 1.62 m mics in the same
                        direction, against the +13.2 dB of spherical spreading.
  acoustic_check.png    Are p' and u' sound? Kinetic/potential energy ratio,
                        p'/rho' wave speed and p-rho coherence per 1/3 octave, with
                        the near mic (inside the flow) as a counter-example.
  energy_ratio.png      The kinetic/potential energy panel alone.
  riemann_<case>.png    Total, outgoing (p+) and incoming (p-) PSD at the three
                        comparison angles, with the experiment and FW-H.
  wave_split.png        Per 1/3 octave: L(p+) - L(p-) (how much the outgoing wave
                        exceeds the incoming one) and tangential/radial velocity
                        energy (0: arrives along the rotor-mic line, 2: from all directions).
  wave_split_levels.png The top row of wave_split.png alone.

Definitions are in direct_intensity.py. Ported from
rotaris-docs/Acoustics_Handoff/Codes/intensity_maps.py - see
HANDOFF_Acoustics.md there for the full theory.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
from matplotlib.colors import Normalize

from . import direct_intensity as di
from . import psd_three_angles as pta

# ============================== SETTINGS ==============================
OUTPUT_DIR = pta.BASE_DIR / 'images' / 'acoustics' / 'intensity'
MAP_CASES = ['2025', '2025-T']
MAP_BANDS = {           # panel title: (low, high) Hz
    'BPF': (170, 230),
    '2 BPF': (370, 430),
    '1--5 kHz': (1000, 5000),
    '5--10 kHz': (5000, 10000),
}
MAP_ARROW = (0.12, 0.5)        # m, arrow length at the bottom and top of FRACTION_DB_RANGE
FRACTION_DB_RANGE = (-20, 0)   # dB, 10 log10(fraction); F <= 0 drawn at the bottom
FRACTION_CMAP = 'viridis'
SPECTRA_CASES = ['2025', '2025-T', '2024']
NEAR_CASES = ['2025', '2025-T']
NEAR_FAR_THETAS = [130, 140]   # mic_0_theta closest to the near-mic direction (135 deg)
XLIM = (100, 19185)
# Wave split: valid above ~200 Hz (far field at 1.62 m) and below ~6 kHz (sound test)
WAVE_SPLIT_XLIM = (150, 8000)
WAVE_SPLIT_FMAX = 6500
# =======================================================================

ROTOR_RADIUS = 0.125


def probe(path):
    return di.probe_spectra(path, pta.TIME_MIN_DIRECT, pta.N_CHUNK_DIRECT,
                            pta.next_greater_power_of_2)


def band_vector(path, low, high):
    """Band-summed intensity vector, fraction and mic position."""
    f, s_pp, intensity, r_hat, rho_c = probe(path)
    m = (f >= low) & (f < high)
    i_band = intensity[:, m].sum(axis=1)
    return i_band, rho_c * (r_hat @ i_band) / s_pp[m].sum(), di.mic_position(path)


def plot_map(case):
    paths = sorted((pta.BASE_DIR / case).glob('mic_0_*.txt'),
                   key=lambda p: int(p.stem.split('_')[-1]))
    norm = Normalize(*FRACTION_DB_RANGE)
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for ax, (title, (low, high)) in zip(axes.flat, MAP_BANDS.items()):
        ring = np.linspace(0, 2 * np.pi, 361)
        ax.plot(di.R_MIC * np.sin(ring), di.R_MIC * np.cos(ring), color='0.8', lw=0.8, ls='--')
        # Experiment arc: +60 to -60 deg from the rotor plane
        arc = np.radians(np.linspace(30, 150, 61))
        ax.plot(1.1 * di.R_MIC * np.sin(arc), 1.1 * di.R_MIC * np.cos(arc), color='0.55', lw=3,
                solid_capstyle='round')
        ax.plot([-ROTOR_RADIUS, ROTOR_RADIUS], [0, 0], color='k', lw=4, solid_capstyle='round')
        ax.annotate('', xy=(0, 0.45), xytext=(0, 0.05),
                    arrowprops=dict(arrowstyle='->', color='k', lw=1.5))
        for path in paths:
            i_band, fraction, pos = band_vector(path, low, high)
            fraction_db = 10 * np.log10(fraction) if fraction > 0 else -np.inf
            level = float(np.clip(norm(fraction_db), 0, 1))     # 0 = bottom, 1 = top of range
            length = MAP_ARROW[0] + level * (MAP_ARROW[1] - MAP_ARROW[0])
            d = np.array([i_band[2], i_band[1]])          # (z, y) projection
            d = length * d / np.linalg.norm(d)
            colour = plt.get_cmap(FRACTION_CMAP)(level)
            ax.annotate('', xy=(pos[2] + d[0] / 2, pos[1] + d[1] / 2),
                        xytext=(pos[2] - d[0] / 2, pos[1] - d[1] / 2),
                        arrowprops=dict(arrowstyle='-|>', color=colour, lw=1 + 1.5 * level,
                                        mutation_scale=8 + 10 * level))
        ax.set_title(title, fontsize=18)
        ax.set_aspect('equal')
        ax.set_xlim(-2.1, 2.1)
        ax.set_ylim(-2.1, 2.1)
        ax.set_xlabel('$z \\left[m\\right]$')
        ax.set_ylabel('$y \\left[m\\right]$')
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.06, top=0.95, wspace=0.3, hspace=0.25)
    cax = fig.add_axes([0.89, 0.25, 0.025, 0.5])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=FRACTION_CMAP), cax=cax,
                       extend='min')
    bar.set_label('Fraction, $10 \\log_{10} F \\left[dB\\right]$', fontsize=16)
    path = OUTPUT_DIR / f'map_{case}.png'
    fig.savefig(path, dpi=pta.DPI, bbox_inches='tight')
    plt.close(fig)
    return path


def band_stats(f, s_pp, intensity, r_hat, rho_c):
    centres, lows, highs, fraction = di.band_fraction(f, s_pp, intensity, r_hat, rho_c)
    i_bands = di.band_sums(f, intensity, lows, highs)                # 3 x nb
    cos = (r_hat @ i_bands) / np.linalg.norm(i_bands, axis=0)
    return centres, fraction, np.degrees(np.arccos(np.clip(cos, -1, 1)))


def plot_fraction_spectra():
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey='row')
    for col, (name, comp) in enumerate(pta.COMPARISONS.items()):
        for key in SPECTRA_CASES:
            path = pta.BASE_DIR / comp['files'][key]
            centres, fraction, angle = band_stats(*probe(path))
            # Spread: the same estimate from each half of the record
            p, u, fs, _, rho_c = di.read_probe(path, pta.TIME_MIN_DIRECT)
            r_hat = di.unit(di.mic_position(path))
            half = p.size // 2
            halves = np.array([band_stats(*di.spectra(p[s], u[:, s], fs, half // 2), r_hat, rho_c)[1]
                               for s in (slice(None, half), slice(half, None))])
            colour = pta.CASES[key]['color']
            label = pta.CASES[key]['label']
            ok = np.isfinite(fraction)
            axes[0, col].vlines(centres[ok], halves.min(0)[ok], halves.max(0)[ok],
                                color=colour, lw=1, alpha=0.6)
            axes[0, col].plot(centres[ok], fraction[ok], color=colour, marker='o', ms=4,
                              lw=1.2, label=label)
            axes[1, col].plot(centres[ok], angle[ok], color=colour, marker='o', ms=4, lw=1.2)
        axes[0, col].set_title(comp['title'], fontsize=18)
    for ax in axes[0]:
        ax.axhline(1, color='0.5', lw=1, ls='--')
        ax.axhline(0, color='0.5', lw=1)
        ax.set_ylim(-0.4, 1.2)
    for ax in axes[1]:
        ax.set_ylim(0, 180)
        ax.set_yticks([0, 45, 90, 135, 180])
        ax.set_xlabel('$Frequency \\left[Hz\\right]$')
    for ax in axes.flat:
        ax.set_xscale('log')
        ax.set_xlim(XLIM)
        ax.grid(True, which='both', ls='--')
    axes[0, 0].set_ylabel('Fraction $\\left[-\\right]$')
    axes[1, 0].set_ylabel('Angle to $\\hat{r} \\left[^\\circ\\right]$')
    axes[0, 2].legend(prop={'size': 14}, loc='upper right')
    fig.tight_layout()
    path = OUTPUT_DIR / 'fraction_spectra.png'
    fig.savefig(path, dpi=pta.DPI, bbox_inches='tight')
    plt.close(fig)
    return path


def band_power(path):
    """1/3-octave band power (Pa^2) of a direct probe."""
    f, s_pp, *_ = probe(path)
    centres, lows, highs = di.band_edges()
    return centres, di.band_sums(f, s_pp * (f[1] - f[0]), lows, highs)


def plot_near_mic_test():
    spreading = 20 * np.log10(di.R_MIC / np.linalg.norm(di.NEAR_MIC))
    fig, ax = plt.subplots(figsize=(15, 4))
    for key in NEAR_CASES:
        centres, near = band_power(pta.BASE_DIR / key / 'near_mic.txt')
        far = np.mean([band_power(pta.BASE_DIR / key / f'mic_0_{t}.txt')[1]
                       for t in NEAR_FAR_THETAS], axis=0)
        ok = np.isfinite(near) & np.isfinite(far)
        ax.plot(centres[ok], 10 * np.log10(near[ok] / far[ok]), color=pta.CASES[key]['color'],
                marker='o', ms=5, lw=1.5, label=pta.CASES[key]['label'])
    ax.axhline(spreading, color='k', ls='--', lw=1.2,
               label=f'Spherical spreading, ${spreading:.1f}$ dB')
    ax.axhline(0, color='0.5', lw=1)
    ax.set_xscale('log')
    ax.set_xlim(XLIM)
    ax.grid(True, which='both', ls='--')
    ax.set_xlabel('$Frequency \\left[Hz\\right]$')
    ax.set_ylabel('$L_{near} - L_{1.62\\,m} \\left[dB\\right]$')
    ax.legend(prop={'size': 14}, loc='upper left')
    fig.tight_layout()
    path = OUTPUT_DIR / 'near_mic_test.png'
    fig.savefig(path, dpi=pta.DPI, bbox_inches='tight')
    plt.close(fig)
    return path


def acoustic_check_bands(path):
    """Per 1/3 octave: kinetic/potential energy ratio, p'-rho' slope (m/s) and coherence.

    Sound waves: Ek/Ep ~ 1 (equipartition), p' = c^2 rho' with c ~ 340 m/s and
    coherence ~ 1. Flow (vortical) fluctuations add kinetic energy with little
    pressure (Ek/Ep >> 1); entropy (thermal) fluctuations break p' = c^2 rho'.
    """
    data = pd.read_csv(path, sep=r'\s+')
    time = data['#time'].values
    keep = time > pta.TIME_MIN_DIRECT
    fs = 1.0 / (time[1] - time[0])
    p = data['static_pressure'].values[keep]
    rho = data['density'].values[keep]
    u = np.stack([data[c].values[keep] for c in ('x_velocity', 'y_velocity', 'z_velocity')])
    p, rho, u = p - p.mean(), rho - rho.mean(), u - u.mean(axis=1, keepdims=True)
    nperseg = int(time.size / pta.N_CHUNK_DIRECT)
    kw = dict(fs=fs, window='hann', nperseg=nperseg, scaling='density')
    f, s_pp = sg.welch(p, **kw)
    s_rr = sg.welch(rho, **kw)[1]
    s_uu = sum(sg.welch(u_i, **kw)[1] for u_i in u)
    s_pr = sg.csd(p, rho, **kw)[1]
    centres, lows, highs = di.band_edges()
    band = lambda s: di.band_sums(f, s, lows, highs)
    rho0 = data['density'].values[keep].mean()
    c2 = np.real(band(s_pr)) / band(s_rr)
    ek_ep = rho0 ** 2 * c2 * band(s_uu) / band(s_pp)
    coherence = np.abs(band(s_pr)) ** 2 / (band(s_pp) * band(s_rr))
    return centres, ek_ep, np.sqrt(np.abs(c2)), coherence


def acoustic_check_curves():
    """(colour, label, path, linestyle): far mics of every comparison, then the near mics."""
    curves = [(pta.CASES[key]['color'], pta.CASES[key]['label'] if i == 0 else None,
               pta.BASE_DIR / comp['files'][key], '-')
              for key in SPECTRA_CASES for i, comp in enumerate(pta.COMPARISONS.values())]
    curves += [(pta.CASES['2026DNS']['color'], 'Near mic (in the flow)' if i == 0 else None,
                pta.BASE_DIR / key / 'near_mic.txt', '--') for i, key in enumerate(NEAR_CASES)]
    return curves


def plot_energy_ratio():
    """Kinetic / potential energy ratio alone (first panel of acoustic_check.png)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for colour, label, path, ls in acoustic_check_curves():
        centres, ek_ep, _, _ = acoustic_check_bands(path)
        ok = np.isfinite(ek_ep)
        ax.plot(centres[ok], ek_ep[ok], color=colour, ls=ls, marker='o', ms=4, lw=1.2, label=label)
    ax.axhline(1, color='k', lw=1.2, ls=':', label='Sound wave, $E_k = E_p$')
    ax.set_yscale('log')
    ax.set_ylim(0.1, 1e6)
    ax.set_xscale('log')
    ax.set_xlim(XLIM)
    ax.grid(True, which='both', ls='--')
    ax.set_xlabel('$Frequency \\left[Hz\\right]$')
    ax.set_ylabel('$E_k / E_p = (\\rho_0 c)^2 |u^\\prime|^2 / p^{\\prime 2} \\left[-\\right]$')
    ax.legend(prop={'size': 13}, loc='upper right')
    fig.tight_layout()
    path = OUTPUT_DIR / 'energy_ratio.png'
    fig.savefig(path, dpi=pta.DPI, bbox_inches='tight')
    plt.close(fig)
    return path


def riemann_spectra(path):
    """PSDs of p', the outgoing and incoming Riemann invariants along r_hat, and
    the tangential/radial kinetic energy ratio per 1/3 octave.

        p+ = (p' + rho c u_r) / 2   outgoing wave
        p- = (p' - rho c u_r) / 2   incoming wave
    Valid for irrotational, isentropic fluctuations, plane-wave-like at the mic
    (k r >> 1: above about 100 Hz at 1.62 m). S(p+) - S(p-) = rho c Re S(p, u_r).
    """
    p, u, fs, length, rho_c = di.read_probe(path, pta.TIME_MIN_DIRECT)
    r_hat = di.unit(di.mic_position(path))
    u_r = r_hat @ u
    u_t = u - np.outer(r_hat, u_r)
    nperseg = int(length / pta.N_CHUNK_DIRECT)
    kw = dict(fs=fs, window='hann', nperseg=nperseg,
              nfft=pta.next_greater_power_of_2(nperseg), scaling='density')
    f, s_pp = sg.welch(p, **kw)
    s_plus = sg.welch((p + rho_c * u_r) / 2, **kw)[1]
    s_minus = sg.welch((p - rho_c * u_r) / 2, **kw)[1]
    s_ur = sg.welch(u_r, **kw)[1]
    s_ut = sum(sg.welch(c, **kw)[1] for c in u_t)
    centres, lows, highs = di.band_edges()
    tangential = di.band_sums(f, s_ut, lows, highs) / di.band_sums(f, s_ur, lows, highs)
    return f, s_pp, s_plus, s_minus, centres, tangential


RIEMANN_COLOURS = {'total': '0.6', 'out': '#2166AC', 'in': '#D6604D'}


def plot_riemann(case):
    """Per comparison angle: total, outgoing and incoming PSD of one direct case,
    with the experiment and the FW-H of the same case for reference."""
    exp_signals, exp_fs = pta.read_experiment([c['exp'] for c in pta.COMPARISONS.values()])
    fwh_key = {'2025': 'fwh-2025', '2025-T': 'fwh-2025-T'}.get(case)
    fig, axes = plt.subplots(len(pta.COMPARISONS), 1, figsize=(15, 4 * len(pta.COMPARISONS)),
                             sharex=True)
    for ax, comp in zip(axes, pta.COMPARISONS.values()):
        f, s_pp, s_plus, s_minus, _, _ = riemann_spectra(pta.BASE_DIR / comp['files'][case])
        iles_p, _ = pta.iles_signal(comp['iles'])
        fe, le = pta.exp_psd(exp_signals[comp['exp']], exp_fs, iles_p.size, 'density')
        ax.plot(f, pta.to_db(s_pp), color=RIEMANN_COLOURS['total'], lw=1.2,
                label=f'{pta.CASES[case]["label"]}, total $p^\\prime$')
        ax.plot(f, pta.to_db(s_plus), color=RIEMANN_COLOURS['out'], lw=1.2,
                label='Outgoing $p^+ = (p^\\prime + \\rho_0 c\\, u_r)/2$')
        ax.plot(f, pta.to_db(s_minus), color=RIEMANN_COLOURS['in'], lw=1.2,
                label='Incoming $p^- = (p^\\prime - \\rho_0 c\\, u_r)/2$')
        if fwh_key:
            ff, lf = pta.fwh_psd(pta.BASE_DIR / comp['files'][fwh_key], 4, 'density')
            ax.plot(ff, lf, color=pta.CASES[fwh_key]['color'], ls=pta.STYLE_FWH, lw=1.2,
                    label=pta.CASES[fwh_key]['label'])
        ax.plot(fe, le, color=pta.EXP_COLOR, lw=pta.EXP_LINE_WIDTH, zorder=10,
                label=pta.CASES['exp']['label'])
        ax.set_xscale('log')
        ax.set_xlim(XLIM)
        ax.set_ylim(pta.YLIM)
        ax.grid(True, which='both', ls='--')
        ax.set_ylabel(pta.Y_LABEL)
        ax.text(0.01, 0.95, comp['title'], transform=ax.transAxes, va='top', fontsize=16)
    axes[0].legend(prop={'size': 13}, loc='upper right', ncol=2)
    axes[-1].set_xlabel('$Frequency \\left[Hz\\right]$')
    fig.tight_layout()
    path = OUTPUT_DIR / f'riemann_{case}.png'
    fig.savefig(path, dpi=pta.DPI, bbox_inches='tight')
    plt.close(fig)
    return path


def wave_split_bands(path):
    """Per 1/3 octave: L(p+) - L(p-) in dB and tangential/radial kinetic energy."""
    f, _, s_plus, s_minus, centres, tangential = riemann_spectra(path)
    _, lows, highs = di.band_edges()
    delta = 10 * np.log10(di.band_sums(f, s_plus, lows, highs) / di.band_sums(f, s_minus, lows, highs))
    return centres, delta, tangential


def plot_wave_split_levels():
    """Top row of wave_split.png alone: L(p+) - L(p-) per angle."""
    fig, axes = plt.subplots(1, len(pta.COMPARISONS), figsize=(16, 4.2), sharey=True)
    for ax, comp in zip(axes, pta.COMPARISONS.values()):
        ax.axhspan(-3, 3, color='0.9', zorder=0)
        ax.axhline(0, color='k', lw=1, ls=':')
        for key in SPECTRA_CASES:
            centres, delta, _ = wave_split_bands(pta.BASE_DIR / comp['files'][key])
            ok = np.isfinite(delta) & (centres <= WAVE_SPLIT_FMAX)
            ax.plot(centres[ok], delta[ok], color=pta.CASES[key]['color'], marker='o', ms=4,
                    lw=1.3, label=pta.CASES[key]['label'])
        ax.set_title(comp['title'], fontsize=18)
        ax.set_ylim(-5, 25)
        ax.set_xscale('log')
        ax.set_xlim(WAVE_SPLIT_XLIM)
        ax.grid(True, which='both', ls='--')
        ax.set_xlabel('$Frequency \\left[Hz\\right]$')
    axes[0].set_ylabel('$L_{p^+} - L_{p^-} \\left[dB\\right]$')
    axes[-1].legend(prop={'size': 13}, loc='upper right')
    fig.tight_layout()
    path = OUTPUT_DIR / 'wave_split_levels.png'
    fig.savefig(path, dpi=pta.DPI, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_wave_split():
    """Outgoing vs incoming level difference and arrival-direction spread, per angle."""
    fig, axes = plt.subplots(2, len(pta.COMPARISONS), figsize=(16, 7.5), sharex=True, sharey='row')
    for col, comp in enumerate(pta.COMPARISONS.values()):
        top, bottom = axes[0, col], axes[1, col]
        top.axhspan(-3, 3, color='0.9', zorder=0)
        top.axhline(0, color='k', lw=1, ls=':')
        bottom.axhline(2, color='k', lw=1, ls=':')
        bottom.axhline(0, color='k', lw=1, ls='--')
        for key in SPECTRA_CASES:
            centres, delta, tangential = wave_split_bands(pta.BASE_DIR / comp['files'][key])
            ok = np.isfinite(delta) & (centres <= WAVE_SPLIT_FMAX)
            style = dict(color=pta.CASES[key]['color'], marker='o', ms=4, lw=1.3,
                         label=pta.CASES[key]['label'])
            top.plot(centres[ok], delta[ok], **style)
            bottom.plot(centres[ok], tangential[ok], **style)
        top.set_title(comp['title'], fontsize=18)
        top.set_ylim(-5, 25)
        bottom.set_ylim(0, 4)
        bottom.set_xlabel('$Frequency \\left[Hz\\right]$')
    for ax in axes.flat:
        ax.set_xscale('log')
        ax.set_xlim(WAVE_SPLIT_XLIM)
        ax.grid(True, which='both', ls='--')
    axes[0, 0].set_ylabel('$L_{p^+} - L_{p^-} \\left[dB\\right]$')
    axes[1, 0].set_ylabel('$|u_t^\\prime|^2 / |u_r^\\prime|^2 \\left[-\\right]$')
    axes[1, 0].text(0.03, 0.55, 'From all directions', transform=axes[1, 0].transAxes, fontsize=13)
    axes[1, 0].text(0.97, 0.05, 'Along the rotor--mic line ($0$)', transform=axes[1, 0].transAxes,
                    fontsize=13, ha='right', bbox=dict(fc='white', ec='none', alpha=0.85))
    axes[0, -1].legend(prop={'size': 13}, loc='upper right')
    fig.tight_layout()
    path = OUTPUT_DIR / 'wave_split.png'
    fig.savefig(path, dpi=pta.DPI, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_acoustic_check():
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharex=True)
    for colour, label, path, ls in acoustic_check_curves():
        centres, ek_ep, c, coherence = acoustic_check_bands(path)
        ok = np.isfinite(ek_ep)
        axes[0].plot(centres[ok], ek_ep[ok], color=colour, ls=ls, marker='o', ms=3, lw=1.1, label=label)
        axes[1].plot(centres[ok], c[ok], color=colour, ls=ls, marker='o', ms=3, lw=1.1)
        axes[2].plot(centres[ok], coherence[ok], color=colour, ls=ls, marker='o', ms=3, lw=1.1)
    axes[0].axhline(1, color='k', lw=1.2, ls=':')
    axes[0].set_yscale('log')
    axes[0].set_ylim(0.1, 1e6)
    axes[0].set_ylabel('$E_k / E_p \\left[-\\right]$')
    axes[1].axhline(340.3, color='k', lw=1.2, ls=':')
    axes[1].set_ylim(280, 360)
    axes[1].set_ylabel('$\\sqrt{p^\\prime / \\rho^\\prime} \\left[m/s\\right]$')
    axes[2].set_ylim(0.5, 1.02)
    axes[2].set_ylabel('Coherence $\\gamma^2_{p\\rho} \\left[-\\right]$')
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlim(XLIM)
        ax.grid(True, which='both', ls='--')
        ax.set_xlabel('$Frequency \\left[Hz\\right]$')
    axes[0].legend(prop={'size': 12}, loc='upper right')
    fig.tight_layout()
    path = OUTPUT_DIR / 'acoustic_check.png'
    fig.savefig(path, dpi=pta.DPI, bbox_inches='tight')
    plt.close(fig)
    return path


def main():
    plt.rcParams.update({
        'text.usetex': pta.USE_LATEX, 'font.family': 'serif', 'font.serif': ['Computer Modern'],
        'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for case in MAP_CASES:
        print('saved', plot_map(case).relative_to(pta.BASE_DIR))
    print('saved', plot_fraction_spectra().relative_to(pta.BASE_DIR))
    print('saved', plot_near_mic_test().relative_to(pta.BASE_DIR))
    print('saved', plot_acoustic_check().relative_to(pta.BASE_DIR))
    print('saved', plot_energy_ratio().relative_to(pta.BASE_DIR))
    for case in SPECTRA_CASES:
        print('saved', plot_riemann(case).relative_to(pta.BASE_DIR))
    print('saved', plot_wave_split().relative_to(pta.BASE_DIR))
    print('saved', plot_wave_split_levels().relative_to(pta.BASE_DIR))


if __name__ == '__main__':
    main()
