"""Custom plot tools."""

# Author: Tuomas Sivula

import matplotlib as mpl


def mix_colors(color1, color2, proportion=0.5):
    """Mix two colors with given ratio.

    Parameters
    ----------
    color1, color2 : matplotlib color specification
        Valid matplotlib color specifications. Examples:
            'r'             : color abbreviation
            'olive'         : html color name
            '0.6'           : gray shade percentage
            (0.2, 0.4, 0.8) : rgb component sequence
            '#a0f0a6'       : hex code
            'C2'            : current cycle color 2

    proportion : float, optional
        Float in the range [0, 1] indicating the desired proportion of color1
        in the resulting mix. Default value is 0.5 for even mix.

    Returns
    -------
    RGB color specification as a sequence of 3 floats.

    Notes
    -----
    Alpha channel is silently dropped.

    """
    color1 = mpl.colors.to_rgb(color1)
    color2 = mpl.colors.to_rgb(color2)
    if proportion < 0 or 1 < proportion:
        raise ValueError('`proportion` has to be in the range [0, 1].')
    p1 = proportion
    p2 = 1 - proportion
    return tuple(p1*comp1 + p2*comp2 for comp1, comp2 in zip(color1, color2))


def lighten(color, proportion=0.5):
    """Make color lighter.

    Parameters
    ----------
    color : matplotlib color specification
        Valid matplotlib color specifications. See :meth:`mix_colors` for
        examples.

    proportion : float, optional
        Float in the range [0, 1] indicating the desired lightness percentage.
        Proportion 0 produces the original color and 1 produces white.

    Returns
    -------
    RGB color specification as a sequence of 3 floats.

    Notes
    -----
    Alpha channel is silently dropped.

    """
    return mix_colors((1.0, 1.0, 1.0), color, proportion=proportion)


def darken(color, proportion=0.5):
    """Make color darker.

    Parameters
    ----------
    color : matplotlib color specification
        Valid matplotlib color specifications. See :meth:`mix_colors` for
        examples.

    proportion : float, optional
        Float in the range [0, 1] indicating the desired darkness percentage.
        Proportion 0 produces the original color and 1 produces black.

    Returns
    -------
    RGB color specification as a sequence of 3 floats.

    Notes
    -----
    Alpha channel is silently dropped.

    """
    return mix_colors((0.0, 0.0, 0.0), color, proportion=proportion)
