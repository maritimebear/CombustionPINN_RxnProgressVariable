# import matplotlib.pyplot as plt
from typing import TypeVar

Axes = TypeVar("Axes")


def xy_plot(ax: Axes,
             y,
             x=None,
             label=None,
             ylabel=None,
             xlabel=None,
             title=None) -> Axes:
    args = [arg for arg in (x, y) if arg is not None]
    if label is None:
        ax.plot(*args)
    else:
        ax.plot(*args, label=label)
        ax.legend()
    # Set axes labels and title
    [f(arg) for f, arg in zip([ax.set_xlabel, ax.set_ylabel, ax.set_title],
                              [xlabel, ylabel, title]) if arg is not None]
    return ax


def semilogy_plot(ax: Axes,
             y,
             x=None,
             label=None,
             ylabel=None,
             xlabel=None,
             title=None) -> Axes:
    args = [arg for arg in (x, y) if arg is not None]
    if label is None:
        ax.semilogy(*args)
    else:
        ax.semilogy(*args, label=label)
        ax.legend()
    # Set axes labels and title
    [f(arg) for f, arg in zip([ax.set_xlabel, ax.set_ylabel, ax.set_title],
                              [xlabel, ylabel, title]) if arg is not None]
    return ax


def contourf(ax: Axes,
             x,
             y,
             field,
             xlabel=None,
             ylabel=None,
             fieldlabel=None,
             title=None) -> Axes:
    # Convenience function to reduce boilerplate code when plotting filled contours
    cf = ax.contourf(x, y, field)
    cbar = ax.get_figure().colorbar(cf, ax=ax)
    _ = [f(arg) for f, arg in zip([ax.set_xlabel, ax.set_ylabel, ax.set_title, cbar.set_label],
                                  [xlabel, ylabel, title, fieldlabel]) if arg is not None]
    return ax
