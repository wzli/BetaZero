import numpy as np
from math import floor

def clip(value, upper, lower=0):
    return max(lower, min( upper, value))

def value_to_index(value, length):
    return clip(floor(length * 0.5 * (1 + value)), length-1)

def value_to_shift_index(value, length):
    return clip(floor(length * 0.5 * value), length-1, -length+1)

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
            shifted_pdf = shifted_pdf/np.sum(shifted_pdf)
        return shifted_pdf

def max_pdf(pdfs):
    pdfs = np.asarray(pdfs)
    cdfs = np.zeros((pdfs.shape[0], pdfs.shape[1] + 1))
    np.cumsum(pdfs, axis=1, out=cdfs[:,1:])
    max_cdf = np.prod(cdfs, axis=0)
    max_pdf = np.diff(max_cdf)
    max_pdf = max_pdf/np.sum(max_pdf)
    return max_pdf
