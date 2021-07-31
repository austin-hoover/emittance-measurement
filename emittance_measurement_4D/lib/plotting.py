from java.awt import BorderLayout
from java.awt import Color
from java.awt import Component
from java.awt import Dimension
from java.awt import FlowLayout
from java.awt import Font
from java.awt import GridLayout
from java.awt import GridBagLayout
from java.awt import GridBagConstraints
from java.awt.event import ActionListener
from java.awt.event import WindowAdapter
from javax.swing import BorderFactory
from javax.swing import BoxLayout
from javax.swing import GroupLayout
from javax.swing import JButton
from javax.swing import JComboBox
from javax.swing import JFrame
from javax.swing import JLabel
from javax.swing import JPanel
from javax.swing import JProgressBar
from javax.swing import JTable
from javax.swing import JTabbedPane
from javax.swing import JTextField
from javax.swing import JFormattedTextField

from xal.extension.widgets.plot import BasicGraphData
from xal.extension.widgets.plot import FunctionGraphsJPanel

# Local
import utils


# 'Colorblind' color cycle
COLOR_CYCLE = [
    Color(0.0, 0.44705882, 0.69803922),
    Color(0.83529412, 0.36862745, 0.0),
    Color(0.0, 0.61960784, 0.45098039),
    Color(0.8, 0.4745098, 0.65490196),
    Color(0.94117647, 0.89411765, 0.25882353),
    Color(0.3372549, 0.70588235, 0.91372549),
]

GRID_COLOR = Color(245, 245, 245)


class PlotPanel(FunctionGraphsJPanel):
    
    def __init__(self, xlabel='', ylabel='', title='', grid=True):
        FunctionGraphsJPanel.__init__(self)
        self.setName(title)
        self.setAxisNames(xlabel, ylabel)
        self.setGraphBackGroundColor(Color.white)   
        self.setGridLineColor(GRID_COLOR)
        if grid == 'y' or not grid:
            self.setGridLinesVisibleX(False)
        if grid == 'x' or not grid:
            self.setGridLinesVisibleY(False)
        

class LinePlotPanel(PlotPanel):
    """Class for 2D line plots."""
    def __init__(self, xlabel='', ylabel='', title='', n_lines=2, lw=3, ms=0, grid=True):
        PlotPanel.__init__(self, xlabel, ylabel, title, grid)
        etched_border = BorderFactory.createEtchedBorder()
        self.setBorder(etched_border)
        self.setLegendButtonVisible(False)
        self.n_lines = n_lines
        self.data_list = [BasicGraphData() for _ in range(n_lines)]
        for data, color in zip(self.data_list, COLOR_CYCLE):
            data.setGraphColor(color)
            data.setLineThick(lw)
            data.setGraphPointSize(ms)
    
    def set_data(self, x, y_list):
        """Set the graph data.
        
        x : list
            List of x values.
        y_list : list or list of lists
            The set of y values for each line. So this can be a list of
            numbers or a list of lists of numbers.
        """
        if len(utils.shape(y_list)) == 1:
            y_list = [y_list]
        self.removeAllGraphData()
        for data, y in zip(self.data_list, y_list):
            data.addPoint(x, y)  
            self.addGraphData(data) 
            
    def set_xlim(self, xmin, xmax, xstep):
        self.setLimitsAndTicksX(xmin, xmax, xstep)
        
    def set_ylim(self, ymin, ymax, ystep):
        self.setLimitsAndTicksX(ymin, ymax, ystep)
        
        
    
class CornerPlotPanel(JPanel):
    def __init__(self, grid=False, figsize=(600, 440)):
        JPanel.__init__(self)
        self.setLayout(GridBagLayout())
        self.setPreferredSize(Dimension(*figsize))
        
        constraints = GridBagConstraints()
        constraints.fill = GridBagConstraints.BOTH
        constraints.gridwidth = 1
        constraints.gridheight = 1
        constraints.weightx = 0.5
        constraints.weighty = 0.5
        
        dim_to_int = {'x':0, 'xp':1, 'y':2, 'yp':3}
        dims = ['x', 'xp', 'y', 'yp']
        xdims = dims[:-1]
        ydims = dims[1:]
        self.plots = dict()
        for ydim in ydims:
            for xdim in xdims:
                i = dim_to_int[ydim] - 1
                j = dim_to_int[xdim]
                if j <= i:
                    plot = PlotPanel(grid=grid)
                    constraints.gridx = j
                    constraints.gridy = i
                    if j == 0:
                        plot.setAxisNameY(ydim)
                    if i == 2:
                        plot.setAxisNameX(xdim)
                    self.add(plot, constraints)
                    key = ''.join([xdim, ',', ydim])
                    self.plots[key] = plot