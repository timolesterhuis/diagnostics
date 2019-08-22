import matplotlib.pyplot as plt


def plot(*args, **kwargs):

    set_title = kwargs.pop("set_title", False)
    align_x = kwargs.pop("align", kwargs.pop("align_x", False))
    sharex = kwargs.pop("sharex", align_x)
    as_dt = kwargs.get("as_dt", False)
    cmap = kwargs.get("cmap", plt.get_cmap("tab10"))

    lines = [a.line(color=cmap(idx), **kwargs) for idx, a in enumerate(args)]
    titles = [a._type for a in args]
    if align_x:
        lims = [
            min([l.get_xdata()[0] for l in lines]),
            max([l.get_xdata()[-1] for l in lines]),
        ]
    if len(lines) == 1:
        fig, ax = plt.subplots(1, 1)
        line = lines[0]
        ax.add_line(line)
        ax.autoscale()
        ax.legend()
        if set_title:
            ax.set_title(titles[idx])
        ax.set_xlabel("Time")
        if as_dt:
            ax.xaxis_date()
            fig.autofmt_xdate()
        if align_x:
            ax.set_xlim(lims)
    else:
        fig, axes = plt.subplots(len(lines), 1, sharex=sharex)
        for line, ax in zip(lines, axes):
            ax.add_line(line)
            ax.autoscale()
            ax.legend()
            if set_title:
                ax.set_title(titles[idx])
            ax.set_xlabel("Time")
            if as_dt:
                ax.xaxis_date()
                fig.autofmt_xdate()
            if align_x:
                ax.set_xlim(lims)
    return fig
