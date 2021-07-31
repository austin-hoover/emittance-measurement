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
from lib.time_and_date_lib import DateAndTimeText
from lib.phase_controller import PhaseController

from lib.analysis_panel import AnalysisPanel
from lib.phase_controller_panel import PhaseControllerPanel
                

class GUI:
    """Graphical user interface for phase controller.
    
    Attributes
    ----------
    phase_controller : PhaseController
        This object calculates the optics needed to obtain a certain phase 
        advance at the wire-scanner. It also updates the live machine to 
        reflect the model.
    model_fields_list : List
        Holds the model field strengths for each scan index. It is filled when 
        the `Calculate model fields` button is pressed.
    """
    def __init__(self):

        # Create panels.
        self.phase_controller_panel = PhaseControllerPanel()
        self.analysis_panel = AnalysisPanel()
        
        # Add panels to tabbed pane.
        self.pane = JTabbedPane(JTabbedPane.TOP)
        self.pane.addTab('Phase controller', self.phase_controller_panel)        
        self.pane.addTab('Analysis', self.analysis_panel)
        
        # Add tabbed pane to frame.
        self.frame = JFrame("RTBT Phase Controller")
        self.frame.setSize(Dimension(1000, 800))
        self.frame.add(self.pane)
        
        # Add time stamp at the bottom of the frame.
        time_text = DateAndTimeText()
        time_panel = JPanel(BorderLayout())
        time_panel.add(time_text.getTimeTextField(), BorderLayout.CENTER)
        self.frame.add(time_panel, BorderLayout.SOUTH)
        
        
    def get_field_set_kws(self):
        """Get key words for setting the live quads fields.
        
        We need these because the application restores the machine to its original
        state when it is closed.
        """
        panel = self.phase_controller_panel
        field_set_kws = {
            'sleep_time': float(panel.sleep_time_text_field.getText()),
            'max_frac_change': float(panel.max_frac_change_text_field.getText()),
        }
        return field_set_kws

    def launch(self):
        """Launch the GUI."""
        class WindowCloser(WindowAdapter):
            def __init__(self, phase_controller, field_set_kws):
                self.phase_controller = phase_controller
                self.field_set_kws = field_set_kws
                
            def windowClosing(self, event):
                """Reset the real machine to its default state before closing window."""
                if self.phase_controller.machine_has_changed:
                    print 'Restoring machine to default state.'
                    self.phase_controller.restore_default_optics('live', **self.field_set_kws)
                sys.exit(1)

        field_set_kws = self.get_field_set_kws()
        phase_controller = self.phase_controller_panel.phase_controller
        window_closer = WindowCloser(phase_controller, field_set_kws)
        self.frame.addWindowListener(window_closer)
        self.frame.show() 
        
            
# Launch application
#-------------------------------------------------------------------------------
gui = GUI()
gui.launch()