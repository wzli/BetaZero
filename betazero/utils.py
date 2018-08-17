import numpy as np
from math import floor


def clip(value, upper, lower=0):
    return max(lower, min(upper, value))


def value_to_index(value, length):
    return clip(floor(length * 0.5 * (1 + value)), length - 1)


def value_to_shift_index(value, length):
    return clip(floor(length * 0.5 * value), length - 1, -length + 1)


def one_hot_pdf(value, length):
    pdf = np.zeros(length)
    pdf[value_to_index(value, length)] = 1
    return pdf


def shift_pdf(pdf, value):
    shift_index = round((pdf.shape[0] - 1) * 0.5 * value)
    shift_index = value_to_shift_index(value, pdf.shape[0])
    if shift_index == 0:
        return pdf
    else:
        shifted_pdf = np.zeros(pdf.shape)
        if shift_index > 0:
            shifted_pdf[shift_index:] = pdf[:-shift_index]
        elif shift_index < 0:
            shifted_pdf[:shift_index] = pdf[-shift_index:]
        total = np.sum(shifted_pdf)
        if total == 0:
            if shift_index > 0:
                shifted_pdf[-1] = 1
            else:
                shifted_pdf[0] = 1
        else:
            shifted_pdf = shifted_pdf / np.sum(shifted_pdf)
        return shifted_pdf


def max_pdf(pdfs):
    pdfs = np.asarray(pdfs)
    cdfs = np.zeros((pdfs.shape[0], pdfs.shape[1] + 1))
    np.cumsum(pdfs, axis=1, out=cdfs[:, 1:])
    max_cdf = np.prod(cdfs, axis=0)
    max_pdf = np.diff(max_cdf)
    max_pdf = max_pdf / np.sum(max_pdf)
    return max_pdf


def expected_value(pdf, return_variance=False):
    support = np.arange(pdf.shape[-1]) / (pdf.shape[-1] - 1)
    expected = np.average(support, weights=pdf)
    if not return_variance:
        return expected
    variance = np.average(support**2, weights=pdf) - expected**2
    return expected, variance


def symetric_arrays(array, rotational_symetry, vertical_symetry,
                    horizontal_symetry):
    """generate list of symetrically equivalent arrays"""
    symetric_arrays = [array]
    if rotational_symetry:
        symetric_arrays.append(
            np.rot90(array, axes=(array.ndim - 2, array.ndim - 1)))
    if vertical_symetry:
        symetric_arrays.extend([
            np.flip(symetric_array, array.ndim - 1)
            for symetric_array in symetric_arrays
        ])
    if horizontal_symetry:
        symetric_arrays.extend([
            np.flip(symetric_array, array.ndim - 2)
            for symetric_array in symetric_arrays
        ])
    return symetric_arrays


def ascii_board(board, perspective=1):
    full_ascii_board = np.chararray((board.shape[0] + 1, board.shape[1] + 1))
    full_ascii_board[0, :] = np.char.mod('%d', np.arange(board.shape[0] + 1))
    full_ascii_board[:, 0] = np.char.mod('%d', np.arange(board.shape[1] + 1))
    full_ascii_board[0, 0] = '.'
    ascii_board = full_ascii_board[1:, 1:]
    board *= perspective
    ascii_board[board == 0] = '.'
    ascii_board[board > 0] = 'X'
    ascii_board[board < 0] = 'O'
    return '\n'.join((' '.join((piece for piece in row))
                      for row in full_ascii_board.decode('utf-8')))


def parse_grid_input(board_size):
    while True:
        try:
            move_index = tuple([
                int(token) - 1 for token in input(
                    'your turn, enter "row col ...": ').split(' ')
            ])
        except ValueError:
            print("INTEGER PARSING ERROR")
            continue
        if len(move_index) != len(board_size):
            print("INVALID INDEX DIMENSION")
            continue
        if any((i < 0 or i >= axis_len
                for i, axis_len in zip(move_index, board_size))):
            print("INVALID INDEX RANGE")
            continue
        return move_index
