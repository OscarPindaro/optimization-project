def is_power_of_two(n):
    """
    Tells if n is a power of 2
    :param n: number that is checked
    :return: True is n is a power of 2
    """
    return (n != 0) and (n & (n - 1) == 0)


def binary_decomposition(x):
    """
    Given the number X, returns the binary digits of x
    :param x: integer number
    :return: A list of the digits of X, where the first element is the most significative
    """
    to_ret = list()
    while x > 0:
        to_ret.append(x % 2)
        x = x // 2
    to_ret.reverse()
    return to_ret


def extend_binary_decomposition(digits_list, n_digits):
    """
    Extends the digit list to length n_digits with zeros
    :param digits_list: initial digit list
    :param n_digits: number of digits desired
    :return: an extended digit list, where the first n_digits - len(digits_list) digits are 0
    """
    assert len(digits_list) <= n_digits, "The digit list has {}, more than the required {} digits".format(
        len(digits_list), n_digits)
    missing_digits = n_digits - len(digits_list)
    zeros = [0 for i in range(missing_digits)]
    zeros.extend(digits_list)
    return zeros
