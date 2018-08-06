import numpy as np

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def max_pdf(pdfs):
    pdfs = np.asarray(pdfs)
    cdfs = np.zeros((pdfs.shape[0], pdfs.shape[1] + 1))
    np.cumsum(pdfs, axis=1, out=cdfs[:,1:])
    max_cdf = np.prod(cdfs, axis=0)
    max_pdf = np.diff(max_cdf)
    max_pdf = max_pdf/np.sum(max_pdf)
    return max_pdf

def one_hot_pdf(value, max_range, max_length):
    pdf = np.zeros(max_length)
    value_index = round((max_length - 1) * 0.5 * (1 + (value / max_range)))
    pdf[value_index] = 1
    return pdf

def sample_pdf(pdf):
    return np.random.choice(pdf.shape, p=pdf)
