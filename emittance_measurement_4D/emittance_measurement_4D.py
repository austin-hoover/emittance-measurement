"""
To do:
    * The GUI-building parts of the code are hard to read. 
    * Some of the GUI components have hard-coded dimensions, so they 
      don't scale with the window.
"""
import math
import time
import sys

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
from javax.swing.event import CellEditorListener
from javax.swing.table import AbstractTableModel
from java.text import NumberFormat
from java.text import DecimalFormat

from xal.extension.widgets.plot import BasicGraphData
from xal.extension.widgets.plot import FunctionGraphsJPanel
from xal.smf import AcceleratorSeqCombo
from xal.smf.data import XMLDataManager

from lib import utils
from lib.analysis_panel import AnalysisPanel
from lib.phase_controller_panel import PhaseControllerPanel
from lib.time_and_date_lib import DateAndTimeText


class EmittanceMeasurement4D:

    def __init__(self):

        # Create panels.
        self.phase_controller_panel = PhaseControllerPanel()
        self.analysis_panel = AnalysisPanel()
        
        # Add panels to tabbed pane.
        self.pane = JTabbedPane(JTabbedPane.TOP)
        self.pane.addTab('Phase controller', self.phase_controller_panel)        
        self.pane.addTab('Analysis', self.analysis_panel)
        
        # Add tabbed pane to frame.
        self.frame = JFrame("4D Emittance Measurement")
        self.frame.setSize(Dimension(1100, 900))
        self.frame.add(self.pane)
        
        # Add time stamp at the bottom of the frame.
        time_text = DateAndTimeText()
        time_panel = JPanel(BorderLayout())
        time_panel.add(time_text.getTimeTextField(), BorderLayout.CENTER)
        self.frame.add(time_panel, BorderLayout.SOUTH)        

    def launch(self):
        """Launch the GUI."""
        class WindowCloser(WindowAdapter):
            
            def __init__(self):
                return
                
            def windowClosing(self, event):
                """Reset the real machine to its default state before closing window."""
                sys.exit(1)

        self.frame.addWindowListener(WindowCloser())
        self.frame.show() 
        
            
# Launch application
#-------------------------------------------------------------------------------
EmittanceMeasurement4D().launch()
