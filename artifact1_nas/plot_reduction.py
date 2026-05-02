"""Generate Fig. 10 (reduced-layer breakdown) from real NAS results.

Reads `results/<model>_<dataset>/logs/hall_of_fame.json` produced by `run.sh`,
picks the lowest-latency Pareto entry for each pair, and plots Full vs Reduced
layer counts in the same black-and-white style as the paper.

For each pair:
  total_conn = len(config['connection_init'])    # full network's connection slots
  total_side = len(config['side_init'])          # full network's side slots
  red_conn   = sum(config['connection_init'])    # NAS-kept connections (1 = kept)
  red_side   = sum(config['side_init'])          # NAS-kept side layers
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.patches import Patch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
OUT_PATH = os.path.join(SCRIPT_DIR, 'figures', 'reduced_layer.pdf')

# Preferred display order; any extra pairs found are appended alphabetically.
PREFERRED_ORDER = ['vgg_gtsrb', 'resnet_cifar10', 'vit-base_cifar100']

DISPLAY_NAME = {
    'vgg_gtsrb':         'VGG-GTSRB',
    'resnet_cifar10':    'ResNet-CIFAR10',
    'vit-base_cifar100': 'ViT-CIFAR100',
}


def discover_pairs():
    """Find every results/<pair>/logs/hall_of_fame.json that exists."""
    if not os.path.isdir(RESULTS_DIR):
        sys.exit(f'[plot] no results dir: {RESULTS_DIR}')
    found = {}
    for name in sorted(os.listdir(RESULTS_DIR)):
        hof = os.path.join(RESULTS_DIR, name, 'logs', 'hall_of_fame.json')
        if os.path.isfile(hof):
            found[name] = hof
    if not found:
        sys.exit(f'[plot] no hall_of_fame.json under {RESULTS_DIR}')
    return found


def lowest_latency_entry(hof_path):
    with open(hof_path) as f:
        entries = json.load(f)
    if not entries:
        return None
    return min(entries, key=lambda e: e['latency'])


def counts(entry):
    cfg = entry['config']
    conn = cfg['connection_init']
    side = cfg['side_init']
    return len(conn), len(side), int(sum(conn)), int(sum(side))


def collect():
    """Return ordered list of (pair, total_conn, total_side, red_conn, red_side, latency)."""
    found = discover_pairs()
    rows = []
    seen = set()
    for pair in PREFERRED_ORDER:
        if pair in found:
            seen.add(pair)
            entry = lowest_latency_entry(found[pair])
            if entry is None:
                print(f'[plot] {pair}: hall_of_fame.json is empty, skipping')
                continue
            tc, ts, rc, rs = counts(entry)
            rows.append((pair, tc, ts, rc, rs, entry['latency']))
    for pair in sorted(set(found) - seen):
        entry = lowest_latency_entry(found[pair])
        if entry is None:
            print(f'[plot] {pair}: hall_of_fame.json is empty, skipping')
            continue
        tc, ts, rc, rs = counts(entry)
        rows.append((pair, tc, ts, rc, rs, entry['latency']))
    return rows


def plot(rows):
    fontsize = 22
    edge_color = 'black'
    SLOT_STYLE = {
        ('Full',    'Connection'): ('white',   '////'),
        ('Full',    'Side'):       ('white',   'xxxx'),
        ('Reduced', 'Connection'): ('#dcdcdc', ''),
        ('Reduced', 'Side'):       ('black',   ''),
    }

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = [max(6.0, 2.4 * len(rows) + 3.0), 5.6]
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['hatch.linewidth'] = 0.6
    plt.rcParams.update({'legend.fancybox': False, 'font.size': fontsize})

    bar_width = 0.34
    WHITE_HALO = [pe.withStroke(linewidth=2.0, foreground='white')]

    fig, ax = plt.subplots()
    n = len(rows)
    index = np.arange(n)
    total_conn = np.array([r[1] for r in rows])
    total_side = np.array([r[2] for r in rows])
    red_conn   = np.array([r[3] for r in rows])
    red_side   = np.array([r[4] for r in rows])

    fc, fh = SLOT_STYLE[('Full', 'Connection')]
    ax.bar(index - bar_width / 2, total_conn, bar_width,
           facecolor=fc, edgecolor=edge_color, linewidth=0.7,
           hatch=fh, zorder=3)
    fc, fh = SLOT_STYLE[('Full', 'Side')]
    ax.bar(index - bar_width / 2, total_side, bar_width, bottom=total_conn,
           facecolor=fc, edgecolor=edge_color, linewidth=0.7,
           hatch=fh, zorder=3)

    fc, fh = SLOT_STYLE[('Reduced', 'Connection')]
    ax.bar(index + bar_width / 2, red_conn, bar_width,
           facecolor=fc, edgecolor=edge_color, linewidth=0.7,
           hatch=fh, zorder=3)
    fc, fh = SLOT_STYLE[('Reduced', 'Side')]
    ax.bar(index + bar_width / 2, red_side, bar_width, bottom=red_conn,
           facecolor=fc, edgecolor=edge_color, linewidth=0.7,
           hatch=fh, zorder=3)

    for x, vc, vs in zip(index - bar_width / 2, total_conn, total_side):
        if vc > 0:
            ax.text(x, vc / 2, f'{vc}', ha='center', va='center',
                    fontsize=fontsize - 4, color='black',
                    path_effects=WHITE_HALO, zorder=4)
        if vs > 0:
            ax.text(x, vc + vs / 2, f'{vs}', ha='center', va='center',
                    fontsize=fontsize - 4, color='black',
                    path_effects=WHITE_HALO, zorder=4)
    for x, vc, vs in zip(index + bar_width / 2, red_conn, red_side):
        if vc > 0:
            ax.text(x, vc / 2, f'{vc}', ha='center', va='center',
                    fontsize=fontsize - 4, color='black', zorder=4)
        if vs > 0:
            ax.text(x, vc + vs / 2, f'{vs}', ha='center', va='center',
                    fontsize=fontsize - 4, color='white', zorder=4)

    for x, v in zip(index - bar_width / 2, total_conn + total_side):
        ax.text(x, v + 0.6, f'{v}', ha='center', va='bottom',
                fontsize=fontsize - 4, zorder=4)
    for x, v in zip(index + bar_width / 2, red_conn + red_side):
        ax.text(x, v + 0.6, f'{v}', ha='center', va='bottom',
                fontsize=fontsize - 4, zorder=4)

    ax.set_xticks(index)
    ax.set_xticklabels([DISPLAY_NAME.get(r[0], r[0]) for r in rows],
                       rotation=15, ha='right')
    ax.set_ylabel('# of Layers')
    ax.set_ylim(0, max(total_conn + total_side) * 1.22)
    ax.grid(axis='y', linestyle='--', alpha=0.55, zorder=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_handles = [
        Patch(facecolor=SLOT_STYLE[('Full', 'Connection')][0], edgecolor=edge_color,
              hatch=SLOT_STYLE[('Full', 'Connection')][1], label='Connection (Full)'),
        Patch(facecolor=SLOT_STYLE[('Full', 'Side')][0], edgecolor=edge_color,
              hatch=SLOT_STYLE[('Full', 'Side')][1], label='Side (Full)'),
        Patch(facecolor=SLOT_STYLE[('Reduced', 'Connection')][0], edgecolor=edge_color,
              hatch=SLOT_STYLE[('Reduced', 'Connection')][1], label='Connection (Reduced)'),
        Patch(facecolor=SLOT_STYLE[('Reduced', 'Side')][0], edgecolor=edge_color,
              hatch=SLOT_STYLE[('Reduced', 'Side')][1], label='Side (Reduced)'),
    ]
    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=True,
               fontsize=fontsize - 2)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, bbox_inches='tight', pad_inches=0.02)
    print(f'Saved -> {OUT_PATH}')


def main():
    rows = collect()
    print(f'[plot] picked lowest-latency entry per pair:')
    print(f'  {"pair":<22}  full(c+s)  reduced(c+s)   latency(ms)')
    for pair, tc, ts, rc, rs, lat in rows:
        print(f'  {pair:<22}  {tc:>3}+{ts:<3}    {rc:>3}+{rs:<3}     {lat:>10.3f}')
    plot(rows)


if __name__ == '__main__':
    main()
