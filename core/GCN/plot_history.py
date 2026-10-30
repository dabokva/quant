"""Plot the metric histories from a timestamped GCN run."""

import argparse
import json
from pathlib import Path

import matplotlib


def plot_history(history_path, output_path=None, show=False):
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    history_path = Path(history_path)
    if history_path.is_dir():
        history_path = history_path / 'history.json'
    saved = json.loads(history_path.read_text())
    history = saved['history']
    groups = [('loss', 'Loss'), ('accuracy', 'Accuracy'), ('roc_auc', 'ROC AUC')]
    groups = [(name, title) for name, title in groups if name in history]
    if not groups:
        raise ValueError('History contains no supported metrics')
    figure, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)
    for axis, (name, title) in zip(axes[0], groups):
        axis.plot(saved['epochs'], history[name], label='Training', linewidth=2)
        if f'val_{name}' in history:
            axis.plot(saved['epochs'], history[f'val_{name}'], label='Validation', linewidth=2)
        axis.set(title=title, xlabel='Epoch', ylabel=title)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle('GCN predecoder training history')
    figure.tight_layout()
    output_path = Path(output_path) if output_path else history_path.with_name('training.png')
    figure.savefig(output_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(figure)
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('history', type=Path, help='Run directory or history.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    print(f'Plot saved to: {plot_history(args.history, args.output, args.show).resolve()}')
