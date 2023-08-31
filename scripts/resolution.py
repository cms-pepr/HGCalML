import os
import sys
from extra_plots import get_energy_summary, plot_energy_summary


def main():

    path = sys.argv[1]
    outputdir = sys.argv[2]
    title_prefix = ""
    if len(sys.argv) > 3:
        title_prefix = sys.argv[3]
    if len(sys.argv) > 4:
        print("Too many arguments")
        print(f"Not using the last ones starting from {sys.argv[4]}")

    summary = get_energy_summary(path)
    fig, _ = plot_energy_summary(summary, title_prefix=title_prefix)
    fig.savefig(f"{outputdir}/{title_prefix}_energy_summary.png")


if __name__ == "__main__":
    assert os.path.exists(sys.argv[1]), f"File {sys.argv[1]} does not exist"
    assert os.path.isdir(sys.argv[2]), f"Directory {sys.argv[2]} does not exist"

    main()
