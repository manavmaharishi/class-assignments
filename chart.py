"""chart.py
Generate a simple bar chart using random data and save it as 'bar_chart.png'.

Usage:
    python chart.py           # generates 5 random bars and opens/saves the figure
    python chart.py -n 8 -o mychart.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def make_bar_chart(n_bars: int = 5, out_file: str = "bar_chart.png", show: bool = True):
    """Create and save a bar chart with random integer values.

    Args:
        n_bars: number of bars to generate
        out_file: output image filename
        show: whether to call plt.show()
    """
    rng = np.random.default_rng()
    labels = [f"Item {i+1}" for i in range(n_bars)]
    values = rng.integers(10, 100, size=n_bars)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=plt.cm.tab10.colors[:n_bars])

    ax.set_title("Random Data Bar Chart")
    ax.set_ylabel("Value")
    ax.set_ylim(0, int(values.max() * 1.2))

    # label each bar with its height
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    fig.savefig(out_file, dpi=150)
    print(f"Saved bar chart to: {out_file}")

    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a bar chart from random data")
    parser.add_argument('-n', '--n_bars', type=int, default=5, help='Number of bars to generate')
    parser.add_argument('-o', '--output', type=str, default='bar_chart.png', help='Output image filename')
    parser.add_argument('--no-show', dest='show', action='store_false', help='Do not display the plot (useful for headless runs)')

    args = parser.parse_args()
    make_bar_chart(n_bars=args.n_bars, out_file=args.output, show=args.show)
