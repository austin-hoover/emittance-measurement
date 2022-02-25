import math

from java.awt import Color
from java.awt import Dimension
from java.awt import GridBagLayout
from java.awt import GridBagConstraints
from java.awt.geom import Ellipse2D
from javax.swing import BorderFactory
from javax.swing import JPanel

from xal.extension.widgets.plot import BasicGraphData
from xal.extension.widgets.plot import CurveData
from xal.extension.widgets.plot import FunctionGraphsJPanel

# Local
import analysis
import utils


CYCLE_COLORBLIND = [
    Color(0.0, 0.44705882, 0.69803922),
    Color(0.83529412, 0.36862745, 0.0),
    Color(0.0, 0.61960784, 0.45098039),
    Color(0.8, 0.4745098, 0.65490196),
    Color(0.94117647, 0.89411765, 0.25882353),
    Color(0.3372549, 0.70588235, 0.91372549),
]
CYCLE_538 = [
    Color(0.0, 0.5607843137254902, 0.8352941176470589),
    Color(0.9882352941176471, 0.30980392156862746, 0.18823529411764706),
    Color(0.8980392156862745, 0.6823529411764706, 0.2196078431372549),
    Color(0.42745098039215684, 0.5647058823529412, 0.30980392156862746),
    Color(0.5450980392156862, 0.5450980392156862, 0.5450980392156862),
    Color(0.5058823529411764, 0.058823529411764705, 0.48627450980392156),
]
GRID_COLOR = Color(245, 245, 245)
DIM_TO_INT = {"x": 0, "xp": 1, "y": 2, "yp": 3}


def rotate(x, y, phi):
    """Rotate point (x, y) clockwise by phi radians."""
    sn, cs = math.sin(phi), math.cos(phi)
    x_rot = +cs * x + sn * y
    y_rot = -sn * x + cs * y
    return x_rot, y_rot


def ellipse_points(cx, cy, tilt=0.0, points=50):
    """Return array of x and y points on ellipse.
    
    Parameters
    ----------
    cx, cy : float
        Length of orizontal and vertical semi-axes, respectively.
    tilt : float
        Tilt angle below horizontal axis.
    points : int
        Number of points to use.
        
    Returns
    -------
    xx, yy : list, shape (points,)
        List of x and y points along on the ellipse boundary.
    """
    xx, yy = [], []
    for psi in utils.linspace(0, 2 * math.pi, points):
        x = cx * math.cos(psi)
        y = cy * math.sin(psi)
        x, y = rotate(x, y, tilt)
        xx.append(x)
        yy.append(y)
    return xx, yy


class PlotPanel(FunctionGraphsJPanel):
    """Variant of `FunctionGraphsJPanel."""
    def __init__(self, xlabel="", ylabel="", title="", grid=True):
        FunctionGraphsJPanel.__init__(self)
        self.setName(title)
        self.setAxisNames(xlabel, ylabel)
        self.setGraphBackGroundColor(Color.white)
        self.setGridLineColor(GRID_COLOR)
        if grid == "y" or not grid:
            self.setGridLinesVisibleX(False)
        if grid == "x" or not grid:
            self.setGridLinesVisibleY(False)


class LinePlotPanel(PlotPanel):
    """Class for 2D line plots."""
    def __init__(
        self,
        xlabel="",
        ylabel="",
        title="",
        n_lines=2,
        lw=3,
        ms=0,
        grid=True,
        cycle=None,
    ):
        PlotPanel.__init__(self, xlabel, ylabel, title, grid)
        etched_border = BorderFactory.createEtchedBorder()
        self.setBorder(etched_border)
        self.setLegendButtonVisible(False)
        self.n_lines = n_lines
        self.data_list = [BasicGraphData() for _ in range(n_lines)]
        if cycle is None:
            cycle = CYCLE_COLORBLIND
        for data, color in zip(self.data_list, cycle):
            data.setGraphColor(color)
            data.setLineThick(lw)
            data.setGraphPointSize(ms)

    def set_data(self, x_list, y_list):
        """Replot with provided data."""
        if not x_list or not y_list:
            return
        if len(utils.shape(y_list)) == 1:  # single list provided
            y_list = [y_list]
        if len(utils.shape(x_list)) == 1:  # single list provided
            x_list = len(y_list) * [x_list]
        self.removeAllGraphData()
        for data, x, y in zip(self.data_list, x_list, y_list):
            data.addPoint(x, y)
            self.addGraphData(data)

    def legend(self, labels):
        """Display a legend."""
        self.setLegendButtonVisible(True)
        for label, data in zip(labels, self.data_list):
            data.setGraphProperty("Legend", label)

    def ellipse(self, cx, cy, tilt=0.0, points=50, lw=4):
        """Plot an ellipse."""
        xvals, yvals = ellipse_points(cx, cy, tilt, points)
        curve_data = CurveData()
        curve_data.setPoints(xvals, yvals)
        curve_data.setLineWidth(lw)
        self.addCurveData(curve_data)

    def plot(
        self, xvals, yvals, yerrs=None, color=None, lw=None, ms=None, ebar_only=False
    ):
        """Add data to the plot."""
        data = BasicGraphData()
        if yerrs is None:
            for x, y in zip(xvals, yvals):
                if x is not None and y is not None:
                    data.addPoint(x, y)
        else:
            for x, y, yerr in zip(xvals, yvals, yerrs):
                if yerr is None:
                    data.addPoint(x, y)
                else:
                    data.addPoint(x, y, yerr)
        if color is not None:
            data.setGraphColor(color)
        if lw is not None:
            data.setLineThick(lw)
            if lw == 0:
                data.setDrawLinesOn(False)
        if ms is not None:
            data.setGraphPointSize(ms)
            if ms == 0:
                data.setDrawPointsOn(False)
        if ebar_only:
            # Keeps error bars but don't draw markers.
            shape = Ellipse2D.Double()
            data.setGraphPointShape(shape)
        self.addGraphData(data)

    def set_xlim(self, xmin, xmax, xstep):
        """Set the horizontal axes limits."""
        self.setLimitsAndTicksX(xmin, xmax, xstep)

    def set_ylim(self, ymin, ymax, ystep):
        """Set the vertical axes limits."""
        self.setLimitsAndTicksY(ymin, ymax, ystep)


class CornerPlotPanel(JPanel):
    """Class for 'corner plot'.

    The 'corner plot' is a matrix of pairwise plots. For four-dimensional
    phase space data, this is a 4x4 matrix. Here, we display the
    two-dimensional projections of the covariance matrix (the six lower-diagonal
    entries in the matrix.
    """
    def __init__(self, grid=False, figsize=None, ticklabels=False):
        JPanel.__init__(self)
        self.setLayout(GridBagLayout())
        if figsize:
            self.setPreferredSize(Dimension(*figsize))

        constraints = GridBagConstraints()
        constraints.fill = GridBagConstraints.BOTH
        constraints.gridwidth = 1
        constraints.gridheight = 1
        constraints.weightx = 0.5
        constraints.weighty = 0.5

        dims = ["x", "xp", "y", "yp"]
        xdims = dims[:-1]
        ydims = dims[1:]
        self.plots = dict()
        for ydim in ydims:
            for xdim in xdims:
                i = DIM_TO_INT[ydim] - 1
                j = DIM_TO_INT[xdim]
                if j <= i:
                    plot = LinePlotPanel(grid=grid)
                    constraints.gridx = j
                    constraints.gridy = i
                    if j == 0:
                        plot.setAxisNameY(ydim)
                    if i == 2:
                        plot.setAxisNameX(xdim)
                    self.add(plot, constraints)
                    key = "".join([xdim, "-", ydim])
                    self.plots[key] = plot

    def rms_ellipses(self, Sigma, lw=4, points=100):
        """Plot projections of the ellipsoid defined by x^T Sigma x = 1/4."""
        max_coords = [math.sqrt(Sigma.get(i, i)) for i in range(4)]
        for key, panel in self.plots.items():
            dim1, dim2 = key.split("-")
            phi, c1, c2 = analysis.rms_ellipse_dims(Sigma, dim1, dim2)
            panel.ellipse(c1, c2, phi, lw=lw, points=points)
            scale = 2.0
            hmax = scale * max_coords[DIM_TO_INT[dim1]]
            vmax = scale * max_coords[DIM_TO_INT[dim2]]
            panel.set_xlim(-hmax, hmax, hmax)
            panel.set_ylim(-vmax, vmax, vmax)

    def clear(self):
        """Clear all subplots."""
        for panel in self.plots.values():
            panel.removeAllGraphData()
            panel.removeAllCurveData()
