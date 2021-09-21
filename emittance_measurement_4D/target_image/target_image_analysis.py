import sys
from datetime import datetime
import collections

import numpy as np
from scipy import optimize as opt
from skimage import filters
from skimage import transform



PIXEL_WIDTH = 1.0 / 1.77


class Image:
    def __init__(self, Z, pixel_width=1):
        self.Z = Z
        self.Zf = None
        self.n_rows, self.n_cols = Z.shape
        self.xx = np.array(list(range(self.n_cols))).astype(float)
        self.yy = np.array(list(range(self.n_rows))).astype(float)
        self.xx -= np.mean(self.xx)
        self.yy -= np.mean(self.yy)
        self.pixel_width = pixel_width
        self.width = abs(self.xx[-1] - self.xx[0])
        self.height = abs(self.yy[-1] - self.yy[0])
        self.set_pixel_width(pixel_width)
        self.X, self.Y = np.meshgrid(self.xx, self.yy)
        
        self.Zfit = None
        self.cov = None
        self.mean_x, self.mean_y = None, None
        self.cx, self.cy, self.angle = None, None, None
        
    def set_pixel_width(self, pixel_width):
        self.xx *= pixel_width
        self.yy *= pixel_width
        
    def filter(self, sigma, **kws):
        self.Zf = filters.gaussian(self.Z, sigma=sigma, **kws)
        return self.Zf
        
    def fit_gauss2d(self, use_filtered=False):
        """Fit 2D Gaussian to the image."""
        Z = self.Zf if use_filtered else self.Z 
        self.Zfit, params = fit_gauss2d(self.X, self.Y, Z)
        sig_xx, sig_yy, sig_xy, mean_x, mean_y, amp = params
        self.c1, self.c2, self.angle = rms_ellipse_dims(sig_xx, sig_yy, sig_xy)
        self.cov = np.array([[sig_xx, sig_xy], [sig_xy, sig_yy]])
        self.mean_x = mean_x
        self.mean_y = mean_y

        
def read_files(filenames):
    TFile = collections.namedtuple('TFile', ['filename', 'timestamp'])
    tfiles = []
    for filename in filenames:
        datetime_str = filename.split('image_')[-1].split('.dat')[0]
        date_str, time_str = datetime_str.split('_')
        year, month, day = [int(s) for s in date_str.split('.')]
        hour, minute, second = [int(s) for s in time_str.split('.')]
        tfiles.append(TFile(filename, datetime(year, month, day, hour, minute, second)))
        
    tfiles = sorted(tfiles, key=lambda tfile: tfile.timestamp)
    
    images = []
    for tfile in tfiles:
        Z = np.loadtxt(tfile.filename).reshape(200, 400)
        Z = np.vstack([np.zeros((100, 400)), Z, np.zeros((100, 400))]) # make square image
        images.append(Image(Z, pixel_width=PIXEL_WIDTH))
    return images


def fit_gauss2d(X, Y, Z):
    def gauss2d(XY, sig_xx, sig_yy, sig_xy, mean_x, mean_y, amp):
        X, Y = XY
        x = X - mean_x
        y = Y - mean_y
        det = sig_xx * sig_yy - sig_xy**2
        Z = amp * np.exp(-0.5 * (sig_yy*x**2 + sig_xx*y**2 - 2*sig_xy*x*y) / det)
        return Z.ravel()
    
    XY = (X, Y)
    p0 = (1., 1., 0., 1., 1., 1.)
    params, _ = opt.curve_fit(gauss2d, XY, Z.ravel(), p0=p0)
    Zfit = gauss2d(XY, *params).reshape(Z.shape)
    return Zfit, params


def rms_ellipse_dims(sig_xx, sig_yy, sig_xy):
    angle = -0.5 * np.arctan2(2 * sig_xy, sig_xx - sig_yy)
    sn, cs = np.sin(angle), np.cos(angle)
    c1 = np.sqrt(abs(sig_xx*cs**2 + sig_yy*sn**2 - 2*sig_xy*sn*cs))
    c2 = np.sqrt(abs(sig_xx*sn**2 + sig_yy*cs**2 + 2*sig_xy*sn*cs))
    return c1, c2, angle