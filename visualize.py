import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_mpl():
    mpl.use('agg')
    mpl.rc('lines', linewidth=3)
    mpl.rc('figure', facecolor='white')
    mpl.rc('grid', color='#C0C0C0', linestyle='dashed', linewidth=1.0, alpha=0.8)
    mpl.rc('axes', facecolor='white', labelsize=18)
    mpl.rcParams["font.size"] = 20
    mpl.rcParams["xtick.labelsize"] = 18
    mpl.rcParams["ytick.labelsize"] = 18
    mpl.rcParams["legend.fontsize"] = 18
    mpl.rcParams["legend.frameon"] = False

def plot_hist(samples, bins, label, x_range, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 9), facecolor='white')
    ax.hist(samples, bins=bins, range=x_range, label=label, density=True)
    ax.grid()   
    ax.legend()

def plot_estimator(estimator, label, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 9), facecolor='white')
    ax.plot(np.linspace(0, len(estimator), num=len(estimator)), estimator, label=label)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax.grid()   
    ax.legend()

if __name__ == "__main__":
    pass