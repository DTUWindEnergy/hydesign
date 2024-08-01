import numpy as np
def expand_to_lifetime(x, life_h = 25*365*24, weeks_per_season_per_year=None):
    """
    Expands (by repeating) a given variable to match an expected lifetime length.

    If weeks_per_season_per_year is an int then it will first build a year out of the selected weeks

    Parameters
    ----------
    x: input variable
    life_h: lifetime in hours.
    weeks_per_season_per_year: None or int.

    Returns
    -------
    x_ext: extended variable
    """
    if weeks_per_season_per_year == None:

        # Extend the data to match the expected lifetime
        len_x = len(x)
        N_repeats = int(np.ceil(life_h/len_x))

    else:
        x_ext = np.array([])

        # extend selected weeks into a year: 4 season of 13 weeks + one extra day.
        for x_batch in split_in_batch(x,weeks_per_season_per_year*7*24):
            x_ext = np.append( x_ext, np.tile(x_batch,20)[:24*7*13] )
        x_ext = np.append( x_ext, x_batch[-24:] )

        # extend the constructed year to match the expected lifetime
        x = x_ext
        N_repeats = int(np.ceil(life_h/365*24))

    return np.tile(x,N_repeats)[:life_h]

def split_in_batch(array, N):
    batch = []
    counter = 0
    while counter * N < len(array):
        if (counter + 1) * N > len(array):
            end = len(array)
        else:
            end = (counter + 1) * N
        batch += [array[counter * N:end]]
        counter += 1
    return batch