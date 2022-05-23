import numpy as np


def king_model(r, k, r_c, r_t):
    """
    https://articles.adsabs.harvard.edu/pdf/1962AJ.....67..471K
    """
    f = (
        k
        * ((1 / np.sqrt(1 + (r / r_c) ** 2)) - (1 / np.sqrt(1 + (r_t / r_c) ** 2)))
        ** 2
    )
    return f


def modified_king_model(r, sigma_naught, r_c, alpha, bg):
    """
    king model with an additional fitting parameter
    """
    sigma = bg + (sigma_naught / (1 + (r / r_c) ** alpha))
    return sigma


def trunc_radius(sigma_0, r_c, alpha, sigma_bg):
    """
    set to 1.5bg =  bg + (peak)/( 1 + (r/r_c)^alpha)
    0.5bg = (peak)/( 1 + (r/r_c)^alpha)
    """
    # can change 1.5 to any vale. Here the truncation radius is when the baground
    # projected density is roughly half of the clusted projected density.
    r_trunc = (r_c**alpha * ((sigma_0 / ((1.5 - 1) * sigma_bg) - 1))) ** (1 / alpha)
    return r_trunc
