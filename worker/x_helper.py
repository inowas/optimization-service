# Our helper function to create x values from an array
def g(array):
    x = sum([a * 2 ** exponent
             for exponent, a in enumerate(array)])

    return x


def g_mod(array, const):
    # Notice: We assume the function to be used with an input array of max size 12 for testing purposes.
    # Notice: With uniform distributed random values as input array, this function still doesn't generate
    # uniform distributed x values. This is due to the fact that each part of the polynom is multiplied
    # with the corresponding number of the array and both sides either from -4 or +4 approaching 0 results
    # in less impact and thus in a value closer to 0! For this case we will shift the function by a number
    # that moves the functions modal away from 0.
    # x += 10000
    x = g(array) + const

    return x