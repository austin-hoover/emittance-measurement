import math
import time
import sys

from java.awt import BorderLayout
from java.awt import Color
from java.awt import Dimension
from java.awt import GridLayout
from java.awt.event import ActionListener
from java.awt.event import WindowAdapter
from javax.swing import BorderFactory
from javax.swing import BoxLayout
from javax.swing import GroupLayout
from javax.swing import JComboBox
from javax.swing import JFrame
from javax.swing import JLabel
from javax.swing import JPanel
from javax.swing import JTextField

from xal.extension.widgets.plot import BasicGraphData
from xal.extension.widgets.plot import FunctionGraphsJPanel
from xal.smf import AcceleratorSeqCombo
from xal.smf.data import XMLDataManager

from lib.time_and_date_lib import DateAndTimeText
from lib.phase_controller import PhaseController
from lib.utils import linspace



class GUI:
    
    def __init__(self):
        
        # Create frame
        #------------------------------------------------------------------------
        self.frame = JFrame("RTBT Phase Controller")
        self.frame.getContentPane().setLayout(BorderLayout())
        time_text = DateAndTimeText()
        time_panel = JPanel(BorderLayout())
        time_panel.add(time_text.getTimeTextField(), BorderLayout.CENTER)
        self.frame.add(time_panel, BorderLayout.SOUTH)

        # Add text fields on left panel
        #------------------------------------------------------------------------
        self.left_panel = JPanel()  
        layout = GroupLayout(self.left_panel)
        self.left_panel.setLayout(layout)
        layout.setAutoCreateContainerGaps(True)
        layout.setAutoCreateGaps(True)
        group_labels = layout.createParallelGroup()
        group_fields = layout.createParallelGroup()
        group_rows = layout.createSequentialGroup()
        layout.setHorizontalGroup(layout.createSequentialGroup()
                                  .addGroup(group_labels)
                                  .addGroup(group_fields))
        layout.setVerticalGroup(group_rows)

        def add_field(label, field):
            group_labels.addComponent(label)
            group_fields.addComponent(field)
            group_rows.addGroup(
                layout.createParallelGroup()
                    .addComponent(label)
                    .addComponent(
                        field, 
                        GroupLayout.PREFERRED_SIZE, 
                        GroupLayout.DEFAULT_SIZE, 
                        GroupLayout.PREFERRED_SIZE
                    )
            )

        text_field_width = 12

        energy_label = JLabel('Energy [GeV]')
        energy_text_field = JTextField('1.0', text_field_width)
        add_field(energy_label, energy_text_field)

        ref_ws_id_label = JLabel('Ref. wire-scanner')
        ws_ids = ['RTBT_Diag:WS20', 'RTBT_Diag:WS21', 
                  'RTBT_Diag:WS23', 'RTBT_Diag:WS24']
        ref_ws_id_dropdown = JComboBox(ws_ids);
        ref_ws_id_dropdown.setSelectedIndex(3);
        add_field(ref_ws_id_label, ref_ws_id_dropdown)

        phase_coverage_label = JLabel('Phase coverage [deg]')
        phase_coverage_text_field = JTextField('180.0', text_field_width)
        add_field(phase_coverage_label, phase_coverage_text_field)

        n_steps_label = JLabel('Total steps')
        n_steps_text_field = JTextField('12', text_field_width)
        add_field(n_steps_label, n_steps_text_field)

        max_beta_label = JLabel("<html>Max. &beta; [m/rad]<html>")
        max_beta_text_field = JTextField('40.0', text_field_width)
        add_field(max_beta_label, max_beta_text_field)

        sleep_time_label = JLabel('Sleep time [s]')
        sleep_time_text_field = JTextField('0.5', text_field_width)
        add_field(sleep_time_label, sleep_time_text_field)

        max_frac_change_label = JLabel('Max. frac. field change')
        max_frac_change_text_field = JTextField('0.01', text_field_width)
        add_field(max_frac_change_label, max_frac_change_text_field)

        self.frame.add(self.left_panel, BorderLayout.WEST)
        
        # Create plotting panels
        #------------------------------------------------------------------------
        self.beta_plot_panel = LinePlotPanel(xlabel='Position [m]', 
                                             ylabel='[m/rad]', 
                                             title='Model beta function vs. position')
        self.phase_plot_panel = LinePlotPanel(xlabel='Position [m]', 
                                              ylabel='Phase adv. mod 2pi', 
                                              title='Model phase advance vs. position')
        self.bpm_plot_panel = LinePlotPanel(xlabel='BPM', 
                                            ylabel='Amplitude [mm]', 
                                            title='BMP amplitudes')
        self.right_panel = JPanel()
        self.right_panel.setLayout(BoxLayout(self.right_panel, BoxLayout.Y_AXIS))
        self.right_panel.add(self.beta_plot_panel)
        self.right_panel.add(self.phase_plot_panel)
        self.right_panel.add(self.bpm_plot_panel)
        self.frame.add(self.right_panel, BorderLayout.CENTER)

    def launch(self):
        
        class WindowCloser(WindowAdapter):
            def windowClosing(self, windowEvent):
                sys.exit(1)
        
        self.frame.addWindowListener(WindowCloser())
        self.frame.setSize(Dimension(1000, 800))
        self.frame.show()        
        
        
class LinePlotPanel(JPanel):

    def __init__(self, xlabel='', ylabel='', title=''):
        self.setLayout(GridLayout(1, 1))
        etched_border = BorderFactory.createEtchedBorder()
        self.setBorder(etched_border)
        self.graph = FunctionGraphsJPanel()
        self.graph.setLegendButtonVisible(False)
        self.graph.setName(title)
        self.graph.setAxisNames(xlabel, ylabel)
        self.graph.setBorder(etched_border)
        self.graph.setGraphBackGroundColor(Color.white)
        self.graph.setGridLineColor(Color(245, 245, 245))
        self.add(self.graph)
        self.data = BasicGraphData()

    def set_data(self, x, y):
        self.graph.removeAllGraphData()
        self.data.addPoint(x, y)  
        self.graph.addGraphData(self.data)
            
            
gui = GUI()
gui.launch()