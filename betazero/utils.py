import numpy as np

def max_pdf(pdfs):
    pdfs = np.asarray(pdfs)
    cdfs = np.zeros((pdfs.shape[0], pdfs.shape[1] + 1))
    np.cumsum(pdfs, axis=1, out=cdfs[:,1:])
    max_cdf = np.prod(cdfs, axis=0)
    max_pdf = np.diff(max_cdf)
    max_pdf = max_pdf/np.sum(max_pdf)
    return max_pdf

def shift_pdf(pdf, value):
    shift_index = round((pdf.shape[0] - 1) * 0.5 * value)
    if shift_index == 0:
        return pdf
    else:
        shifted_pdf = np.zeros(pdf.shape)
        if shift_index > 0:
            shifted_pdf[shift_index:] = pdf[:-shift_index]
        elif shift_index < 0:
            shifted_pdf[:shift_index] = pdf[-shift_index:]
        total = np.sum(shifted_pdf)
        if total is 0:
            if shift_index > 0:
                shifted_pdf[-1] = 1
            else:
                shifted_pdf[0] = 1
        else:
            shifted_pdf = shifted_pdf/np.sum(shifted_pdf)
        return shifted_pdf

def one_hot_pdf(value, max_length):
    pdf = np.zeros(max_length)
    value_index = round((max_length - 1) * 0.5 * (1 + value))
    pdf[value_index] = 1
    return pdf
