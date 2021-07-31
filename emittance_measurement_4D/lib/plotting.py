from java.awt import BorderLayout
from java.awt import Color
from java.awt import Component
from java.awt import Dimension
from java.awt import FlowLayout
from java.awt import Font
from java.awt import GridLayout
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


class LinePlotPanel(FunctionGraphsJPanel):
    """Class for 2D line plots."""
    def __init__(self, xlabel='', ylabel='', title='', n_lines=2, lw=3, ms=0, grid=True):
        etched_border = BorderFactory.createEtchedBorder()
        self.setBorder(etched_border)
        self.setLegendButtonVisible(False)
        self.setName(title)
        self.setAxisNames(xlabel, ylabel)
        self.setBorder(etched_border)
        self.setGraphBackGroundColor(Color.white)   
        self.setGridLineColor(Color(245, 245, 245))
        if grid == 'y' or not grid:
            self.setGridLinesVisibleX(False)
        if grid == 'x' or not grid:
            self.setGridLinesVisibleY(False)
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