def set_style(plt):
    # assuming in the figsize height = 5
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = plt.rcParams['xtick.labelsize']
    plt.rcParams['legend.fontsize'] = plt.rcParams['xtick.labelsize']
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.alpha'] = 0.25
