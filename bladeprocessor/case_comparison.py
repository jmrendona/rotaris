'''
Cross-SIMULATION comparison plots for StripForces results - overlay
several cases (e.g. different mesh resolutions/models, such as this
project's own '2025', '2025-T', '2026', '2026-DNS' thesis comparison) on
ONE set of axes, instead of one panel per case.

Distinct from StripForces.plot_time_trace()/plot_vs_angle(), which
overlay multiple STRIPS of a SINGLE case (one panel per case would be
needed to compare several cases with those), and distinct from
bladeprocessor.surface_field.SurfaceFieldComparator, which compares 2D
contour FIELDS across cases, not strip time-series/phase-locked data.

Motivating case (see manager_comparison.py): a thesis results chapter
comparing 4 simulation variants of the same isolated hovering rotor,
where a naive N_cases x N_components grid (one panel per case, one line
per strip within each) either buries the actual comparison of interest
(how does a given force component differ BETWEEN cases) inside a dozen
small, separately-legended panels, or, if strip count and case count are
both kept, becomes unreadable at the physical size a thesis page allows.

Design: color encodes STRIP (a colormap, same convention as
StripForces.plot_time_trace()/plot_vs_angle() - see _strip_colors());
LINESTYLE encodes CASE (cycling through a fixed list of dash patterns -
see _LINESTYLES). Together these let every (case, strip) combination be
told apart on ONE panel per force component (3 panels total, not one per
case), with a SINGLE shared legend explaining both encodings for the
whole figure - built once, not repeated (or, worse, omitted) on every
component's own panel.

Print-size handling: figures are generated directly at their FINAL
physical size (inches, `total_width_in`/`total_height_in`), with LOCALLY
overridden (via `plt.rc_context`, not the project-wide `plt.rcParams`)
tick/label/legend font sizes - not the much larger font sizes
StripForces' own single-panel methods use, which would be badly
oversized once three panels share the same physical row.

The EXACT condition for a chosen `tick_fontsize`/`label_fontsize`/
`legend_fontsize` (points) to come out at that same size on the printed
page: the figure must be included with
`\\includegraphics[width=\\linewidth]{...}` (or whatever fixed width is
used) AND `total_width_in` must equal that same width converted to
inches (LaTeX `\\linewidth`/`\\textwidth` are usually reported in `pt`;
divide by 72.27 to get inches). If those two widths match, no further
scaling happens at inclusion time, and a `tick_fontsize=14` really is 14
pt on the page - the current default here, chosen so labels read at
comfortably above the thesis body text's own likely 11-12 pt even after
accounting for print rounding, per this project's own printed-figure
requirement. If they DON'T match (e.g. `total_width_in` is left at this
module's 6.3 in default but the actual document's `\\linewidth` is
different), `\\includegraphics` silently rescales the whole figure,
including its text, by that same mismatched ratio - check the actual
`\\linewidth` value in the thesis document and update `total_width_in`
to match before treating the on-page result as final. Saving as a
vector format (`savepath` ending in `.pdf`, not `.png`) is recommended
for this reason: vector text stays exactly the specified point size and
perfectly crisp at whatever the final rendered size ends up being,
whereas a raster (PNG) locks in a fixed pixel grid at the DPI it was
exported at and can look soft if LaTeX ends up scaling it at all.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Solid, then three custom dash patterns with deliberately LARGE segment
# lengths - in that order, cycling if there are ever more than 4 cases.
# Matplotlib's built-in '--'/':'/'-.' patterns use small, fine segments
# calibrated for a big, single-panel, low-density figure; at this
# module's compressed multi-panel print size, and especially against a
# NOISY, busy, heavily-overlapping raw time trace (as opposed to a
# smooth phase-locked curve), those fine patterns visually blur into
# what looks like a solid line - confirmed by direct visual inspection
# during development (fig56_compare_test.png in the test images folder
# showed exactly this: linestyle was legible on the smooth polar/phase-
# locked comparison but not on the noisy raw time trace, with the
# default matplotlib dash patterns). These larger, explicit (on, off[,
# on, off...]) tuples (in points) stay visually distinct even under
# heavy noise and line overlap. A 5th/6th entry could be added here if
# needed, but 4 already covers this project's own 2025/2025-T/2026/
# 2026-DNS comparison exactly.
_LINESTYLES = ['-', (0, (6, 2)), (0, (1, 1.6)), (0, (5, 1.5, 1.2, 1.5))]

_COMPONENT_LABELS = {
    'axial': r'$F_{axial}$ [N]',
    'radial': r'$F_{radial}$ [N]',
    'tangential': r'$F_{tangential}$ [N]',
}


def _strip_colors(strips, cmap):

    '''Discrete colors for each selected strip, sampled from `cmap` - same convention as StripForces' own plots.'''

    return plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(strips)))


def _build_shared_legend(legend_ax, strips, case_labels, colors, radii=None, fontsize=8):

    '''
    ONE combined legend for the whole multi-panel figure, drawn INSIDE
    its own dedicated axes cell (the fourth cell of the 2x2 grid built by
    plot_time_trace_compare()/plot_vs_angle_compare(), the three data
    panels occupying the other three) rather than as a figure-level
    floating legend anchored by `bbox_to_anchor`. This is a deliberate
    choice, not just a style preference: a figure-spanning legend row
    below three narrow panels, combined with print-sized (14pt+) fonts,
    was found during development to interact badly with
    `tight_layout()`/`bbox_inches='tight'`, producing large, asymmetric
    blank margins - confirmed by direct visual inspection
    (fig56_compare_test.png in the test images folder). A legend confined
    to a normal axes cell is laid out by matplotlib like any other
    subplot content, avoiding that failure mode entirely.

    One colored line per selected strip (the color dimension), followed
    by one black line per case (the linestyle dimension) - so a reader
    can decode both encodings from a single legend rather than needing
    them explained twice or guessed at.

    radii : array-like, optional
        Physical radius [m] of each selected strip (e.g.
        result['radius'][list(strips)]) - if given, appended to each
        strip's legend label as r [m] for a physically meaningful
        reference; omitted if not available (e.g. mismatched strip
        layouts between cases - see plot_time_trace_compare()'s own
        docstring on this assumption).
    '''

    handles, labels = [], []

    for i, strip_i in enumerate(strips):
        handles.append(Line2D([0], [0], color=colors[i], linewidth=2.2))
        if radii is not None:
            labels.append(f'Strip {strip_i + 1} ($r$={radii[i]:.3f} m)')
        else:
            labels.append(f'Strip {strip_i + 1}')

    for j, case_label in enumerate(case_labels):
        handles.append(Line2D([0], [0], color='black', linestyle=_LINESTYLES[j % len(_LINESTYLES)], linewidth=1.6))
        labels.append(str(case_label))

    legend_ax.axis('off')
    legend_ax.legend(handles, labels, loc='center', frameon=False, fontsize=fontsize,
                      handlelength=2.0, labelspacing=0.5, borderpad=0.5, handletextpad=0.6)


def _check_consistent_strip_layout(results: dict, strips):

    '''
    Warns (does not raise) if the selected strips' physical radii differ
    non-trivially between cases - _build_shared_legend()'s radii come
    from only ONE case (the last one processed), which silently assumes
    every case shares the same span_min/span_max/n_span_bins layout (as
    this project's own 4-case thesis comparison does, all built with the
    same 10-equally-spaced-strips convention - see class docstring).
    '''

    radii_by_case = {}
    for label, result in results.items():
        radius = result['radius']
        sel = np.asarray(strips)
        if np.any(sel >= len(radius)):
            raise ValueError(f"strips {strips} out of range for case '{label}' ({len(radius)} strips available).")
        radii_by_case[label] = radius[sel]

    reference = next(iter(radii_by_case.values()))
    for label, radii in radii_by_case.items():
        if not np.allclose(radii, reference, rtol=0.05, equal_nan=True):
            import warnings
            warnings.warn(
                f"Case '{label}' selected strips sit at different physical radii than the reference case "
                f"({radii} vs {reference}) - the shared legend's radius labels will only reflect the LAST "
                "case processed. Consider using bladeprocessor.strip_forces.StripForces.nearest_strip() per "
                "case instead of a shared raw index list if cases don't share the same span layout."
            )
            break


def plot_time_trace_compare(results: dict, dt: float = None, strips=(0, 4, 9),
                             component_list=('axial', 'tangential', 'radial'), cmap: str = 'cividis',
                             total_width_in: float = 6.3, total_height_in: float = 4.6,
                             tick_fontsize: int = 14, label_fontsize: int = 15, legend_fontsize: int = 14,
                             savepath: str = None, dpi: int = 600):

    '''
    Raw per-frame force time trace, compared across several SIMULATIONS
    at once - the cross-case counterpart to StripForces.plot_time_trace()
    (which compares strips WITHIN one case). One panel per component (3
    by default, matching StripForces' own axial/radial/tangential split),
    each panel overlaying every case's selected strips: color = strip,
    linestyle = case (see module docstring) - all sharing ONE legend for
    the whole figure.

    Parameters
    ----------
    results : dict[str, dict]
        {case_label: compute()'s return value} - one entry per
        simulation to compare, e.g. {'2025': result_2025, '2025-T':
        result_2025t, '2026': result_2026, '2026-DNS': result_2026dns}.
        Every result must share the same strip INDEXING (this project's
        own 4-case comparison does, all built with the same 10-equally-
        spaced-strips convention) - _check_consistent_strip_layout()
        warns, but does not raise, if that assumption looks violated.
    dt : float, optional
        Physical timestep [s] - if omitted, the x-axis is frame index
        (dimensionless). Assumed the SAME dt for every case; pass
        per-case dt separately (call this function once per case's own
        dt, with `ax=` reused) if that's not true here.
    strips : sequence of int
        Which strip indices to plot from EVERY case - 3 by default (this
        project's own choice: enough to show the spanwise trend without
        the visual overload of all 10).
    component_list : sequence of str
        Which of 'axial', 'radial', 'tangential' to plot - EXACTLY 3,
        no more, no fewer: they fill the top-left, top-right, and
        bottom-left cells of a 2x2 grid, with the bottom-right cell
        reserved for the shared legend (see module/_build_shared_legend()
        docstrings for why a grid cell, not a figure-spanning legend row,
        is used). Passing a different number raises ValueError.
    total_width_in, total_height_in : float
        The figure's FINAL physical size in inches - see module
        docstring's "Print-size handling" note for the EXACT condition
        under which `tick_fontsize`/`label_fontsize` come out at that
        same point size on the printed page. Default 6.3 x 4.6 in
        matches a common single-column thesis \\linewidth at a roughly
        square-ish 2x2 grid; set `total_width_in` to your own document's
        actual \\linewidth (in inches: pt / 72.27).
    tick_fontsize, label_fontsize, legend_fontsize : int
        Font sizes in points, applied via a LOCAL plt.rc_context (does
        not affect this project's other, larger-format single-panel
        plots' own global rcParams). Default 14/15/14 pt, chosen so
        labels read comfortably above typical (11-12 pt) thesis body
        text once printed at `total_width_in` - see the module
        docstring's print-size note for the exact condition this relies
        on.

    Returns
    -------
    (fig, axes) - axes is the (2, 2) array from plt.subplots(); axes[1,1]
        is the legend cell (axis turned off), not a data panel.
    '''

    if len(results) == 0:
        raise ValueError("results is empty - need at least one case to plot.")
    if len(component_list) != 3:
        raise ValueError(
            f"component_list must have EXACTLY 3 entries to fill the 2x2 grid's three data cells "
            f"(bottom-right is reserved for the legend) - got {len(component_list)}: {component_list}."
        )

    case_labels = list(results.keys())
    strips = list(strips)
    _check_consistent_strip_layout(results, strips)
    colors = _strip_colors(strips, cmap)

    with plt.rc_context({
        'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern'],
        'axes.labelsize': label_fontsize, 'xtick.labelsize': tick_fontsize,
        'ytick.labelsize': tick_fontsize, 'legend.fontsize': legend_fontsize,
    }):

        fig, axes = plt.subplots(2, 2, figsize=(total_width_in, total_height_in))
        data_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]

        for ax, component in zip(data_axes, component_list):
            for j, case_label in enumerate(case_labels):
                arr = results[case_label][component]  # (n_frames, n_span_bins)
                t = np.arange(arr.shape[0]) * dt if dt is not None else np.arange(arr.shape[0])
                for i, strip_i in enumerate(strips):
                    ax.plot(t, arr[:, strip_i], color=colors[i],
                            linestyle=_LINESTYLES[j % len(_LINESTYLES)], linewidth=1.3, alpha=0.9)

            ax.set_xlabel('Time [s]' if dt is not None else 'Frame index')
            ax.set_ylabel(_COMPONENT_LABELS.get(component, component))
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.tick_params(axis='both', labelsize=tick_fontsize)

        last_case = results[case_labels[-1]]
        radii_for_legend = last_case['radius'][strips] if 'radius' in last_case else None
        _build_shared_legend(axes[1, 1], strips, case_labels, colors, radii=radii_for_legend, fontsize=legend_fontsize)

        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=dpi)

    return fig, axes


def plot_vs_angle_compare(phase_locked: dict, strips=(0, 4, 9),
                           component_list=('axial', 'tangential', 'radial'), cmap: str = 'cividis',
                           total_width_in: float = 6.3, total_height_in: float = 4.8,
                           tick_fontsize: int = 14, label_fontsize: int = 15, legend_fontsize: int = 14,
                           savepath: str = None, dpi: int = 600):

    '''
    Phase-locked (revolution-folded) force vs. rotor azimuth, compared
    across several SIMULATIONS at once - the cross-case counterpart to
    StripForces.plot_vs_angle() (which compares strips WITHIN one case).
    Same color = strip / linestyle = case / shared-legend design as
    plot_time_trace_compare() - see that function's and the module's own
    docstring - applied to a 2x2 grid of POLAR axes instead (0 deg at 3
    o'clock/east, increasing counterclockwise, same convention as
    StripForces.plot_vs_angle()); the bottom-right cell stays a normal
    (non-polar) axes, holding the shared legend.

    Parameters
    ----------
    phase_locked : dict[str, dict]
        {case_label: phase_lock()'s return value} - one entry per
        simulation to compare. Same shared-strip-layout assumption as
        plot_time_trace_compare()'s `results`.
    component_list : sequence of str
        EXACTLY 3 entries - see plot_time_trace_compare()'s own
        docstring for why.
    strips, cmap, total_width_in, total_height_in, tick_fontsize,
    label_fontsize, legend_fontsize : see plot_time_trace_compare() -
        identical meaning here.

    Returns
    -------
    (fig, axes) - axes is the (2, 2) array from plt.subplots(); axes[1,1]
        is the (non-polar) legend cell, not a data panel.
    '''

    if len(phase_locked) == 0:
        raise ValueError("phase_locked is empty - need at least one case to plot.")
    if len(component_list) != 3:
        raise ValueError(
            f"component_list must have EXACTLY 3 entries to fill the 2x2 grid's three polar data cells "
            f"(bottom-right is reserved for the legend) - got {len(component_list)}: {component_list}."
        )

    case_labels = list(phase_locked.keys())
    strips = list(strips)
    _check_consistent_strip_layout(phase_locked, strips)
    colors = _strip_colors(strips, cmap)

    with plt.rc_context({
        'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern'],
        'axes.labelsize': label_fontsize, 'xtick.labelsize': tick_fontsize,
        'ytick.labelsize': tick_fontsize, 'legend.fontsize': legend_fontsize,
    }):

        fig = plt.figure(figsize=(total_width_in, total_height_in))
        data_axes = [
            fig.add_subplot(2, 2, 1, projection='polar'),
            fig.add_subplot(2, 2, 2, projection='polar'),
            fig.add_subplot(2, 2, 3, projection='polar'),
        ]
        legend_ax = fig.add_subplot(2, 2, 4)  # deliberately NOT polar - see module docstring

        for ax, component in zip(data_axes, component_list):
            for j, case_label in enumerate(case_labels):
                pl = phase_locked[case_label]
                theta = np.deg2rad(pl['azimuth_deg'])
                theta_closed = np.append(theta, theta[0])  # close the loop, no visible gap at 0/360 deg
                vals = pl[component]
                for i, strip_i in enumerate(strips):
                    y = vals[:, strip_i]
                    y_closed = np.append(y, y[0])
                    ax.plot(theta_closed, y_closed, color=colors[i],
                            linestyle=_LINESTYLES[j % len(_LINESTYLES)], linewidth=1.3, alpha=0.9)

            ax.set_theta_zero_location('E')
            ax.set_theta_direction(1)
            ax.set_rlabel_position(22.5)  # off the 45deg gridline, where it would otherwise overlap
            ax.set_title(_COMPONENT_LABELS.get(component, component), fontsize=label_fontsize, pad=16)
            ax.tick_params(axis='both', labelsize=tick_fontsize)

        last_case = phase_locked[case_labels[-1]]
        radii_for_legend = last_case['radius'][strips] if 'radius' in last_case else None
        _build_shared_legend(legend_ax, strips, case_labels, colors, radii=radii_for_legend, fontsize=legend_fontsize)

        axes = np.array([[data_axes[0], data_axes[1]], [data_axes[2], legend_ax]], dtype=object)

        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=dpi)

    return fig, axes
