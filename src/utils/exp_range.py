import numpy as np


def exp_range_float(start: float, stop: float, n_points: int):
    """
    Generate a range of exponentially spaced floats between start and stop
    """
    return np.logspace(np.log10(start), np.log10(stop), n_points)


def exp_range_int(start: int, stop: int, n_points: int):
    """
    Generate a range of exponentially spaced integers between start and stop
    """

    # Generate 10*n_points exponentially spaced floats
    exp_spaced_floats = np.logspace(np.log10(start), np.log10(stop), n_points * 10)

    # Convert to integers and remove duplicates
    exp_spaced_integers = sorted(set(map(int, exp_spaced_floats)))

    # Interpolate if there are more unique points than needed
    if len(exp_spaced_integers) > n_points:
        exp_spaced_integers = np.interp(
            # Desired number of points
            np.linspace(0, len(exp_spaced_integers) - 1, n_points),
            # Original x-values
            np.arange(len(exp_spaced_integers)),
            # Original y-values
            exp_spaced_integers,
        ).astype(int)

    return exp_spaced_integers
