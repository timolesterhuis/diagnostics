import matplotlib.pyplot as plt


def plot(*args, **kwargs):

    fig = kwargs.pop("figure", plt.figure())
    align_x = kwargs.pop("align_x", False)
    as_dt = kwargs.get("as_dt", False)
    cmap = kwargs.get("cmap", plt.get_cmap("tab10"))

    lines = [a.line(color=cmap(idx), **kwargs) for idx, a in enumerate(args)]
    if align_x:
        lims = [
            min([l.get_xdata()[0] for l in lines]),
            max([l.get_xdata()[-1] for l in lines]),
        ]

    for idx, line in enumerate(lines):
        ax = fig.add_subplot(len(lines), 1, idx + 1)
        ax.add_line(line)
        ax.autoscale()
        ax.legend()
        if as_dt:
            ax.xaxis_date()
            fig.autofmt_xdate()
        if align_x:
            ax.set_xlim(lims)
    return fig
