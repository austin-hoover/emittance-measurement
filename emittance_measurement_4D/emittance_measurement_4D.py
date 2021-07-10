import math
import time
import sys

from java.awt import BorderLayout
from java.awt import Color
from java.awt import Dimension
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
from javax.swing import JTextField
from javax.swing.event import DocumentListener

from xal.extension.widgets.plot import BasicGraphData
from xal.extension.widgets.plot import FunctionGraphsJPanel
from xal.smf import AcceleratorSeqCombo
from xal.smf.data import XMLDataManager

from lib.time_and_date_lib import DateAndTimeText
from lib.phase_controller import PhaseController
from lib.utils import linspace


COLOR_CYCLE = [
    Color(0.0, 0.44705882, 0.69803922),
    Color(0.83529412, 0.36862745, 0.0),
    Color(0.0, 0.61960784, 0.45098039),
    Color(0.8, 0.4745098, 0.65490196),
    Color(0.94117647, 0.89411765, 0.25882353),
    Color(0.3372549, 0.70588235, 0.91372549),
]

class GUI:
    
    def __init__(self):
        self.phase_controller = PhaseController()

        # Create frame
        #------------------------------------------------------------------------
        self.frame = JFrame("RTBT Phase Controller")
        self.frame.getContentPane().setLayout(BorderLayout())
        time_text = DateAndTimeText()
        time_panel = JPanel(BorderLayout())
        time_panel.add(time_text.getTimeTextField(), BorderLayout.CENTER)
        self.frame.add(time_panel, BorderLayout.SOUTH)

        # Text fields panel
        #------------------------------------------------------------------------
        self.text_fields_panel = JPanel()  
        layout = GroupLayout(self.text_fields_panel)
        self.text_fields_panel.setLayout(layout)
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
        self.energy_text_field = JTextField('1.0', text_field_width)
        self.energy_text_field.addActionListener(
            EnergyTextFieldListener(self.energy_text_field, self.phase_controller))
        add_field(energy_label, self.energy_text_field)

        ref_ws_id_label = JLabel('Ref. wire-scanner')
        ws_ids = ['RTBT_Diag:WS20', 'RTBT_Diag:WS21', 
                  'RTBT_Diag:WS23', 'RTBT_Diag:WS24']
        self.ref_ws_id_dropdown = JComboBox(ws_ids)
        self.ref_ws_id_dropdown.setSelectedIndex(3)
        self.ref_ws_id_dropdown.addActionListener(
            RefWsIdTextFieldListener(self.ref_ws_id_dropdown, self.phase_controller))
        add_field(ref_ws_id_label, self.ref_ws_id_dropdown)

        phase_coverage_label = JLabel('Phase coverage [deg]')
        self.phase_coverage_text_field = JTextField('180.0', text_field_width)
        add_field(phase_coverage_label, self.phase_coverage_text_field)

        n_steps_label = JLabel('Total steps')
        self.n_steps_text_field = JTextField('12', text_field_width)
        add_field(n_steps_label, self.n_steps_text_field)

        max_beta_label = JLabel("<html>Max. &beta; [m/rad]<html>")
        self.max_beta_text_field = JTextField('40.0', text_field_width)
        add_field(max_beta_label, self.max_beta_text_field)

        sleep_time_label = JLabel('Sleep time [s]')
        self.sleep_time_text_field = JTextField('0.5', text_field_width)
        add_field(sleep_time_label, self.sleep_time_text_field)

        max_frac_change_label = JLabel('Max. frac. field change')
        self.max_frac_change_text_field = JTextField('0.01', text_field_width)
        add_field(max_frac_change_label, self.max_frac_change_text_field)

        # Fill left panel
        self.left_panel = JPanel()
        self.left_panel.setLayout(BoxLayout(self.left_panel, BoxLayout.Y_AXIS))
        
        label = JLabel('Settings')
        font = label.getFont()
        label.setFont(Font(font.name, font.BOLD, int(1.1 * font.size)));
        self.left_panel.add(label)
        
        self.left_panel.add(self.text_fields_panel)
        
        label = JLabel('Actions')
        label.setFont(Font(font.name, font.BOLD, int(1.1 * font.size)));
        self.left_panel.add(label)
        
        self.frame.add(self.left_panel, BorderLayout.WEST)
        
        
        # Create buttons panel
        #------------------------------------------------------------------------ 
        self.buttons_panel = JPanel()
        self.buttons_panel.setLayout(BoxLayout(self.buttons_panel, BoxLayout.Y_AXIS))
        
        self.calc_optics_button = JButton('Calculate model optics')  
        self.buttons_panel.add(self.calc_optics_button)
        
        self.set_optics_button = JButton('Set live optics')
        self.buttons_panel.add(self.set_optics_button)
        
        self.left_panel.add(self.buttons_panel)
        
        
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
    
        
    def update_plots(self):
        beta_x_list, beta_y_list = [], []
        phases_x_list, phases_y_list = [], []
        for params in self.phase_controller.tracked_twiss():
            mu_x, mu_y, alpha_x, alpha_y, beta_x, beta_y, eps_x, eps_y = params
            beta_x_list.append(beta_x)
            beta_y_list.append(beta_y)
            phases_x_list.append(mu_x)
            phases_y_list.append(mu_y)
        positions = self.phase_controller.positions
        self.beta_plot_panel.set_data(positions, beta_x_list, beta_y_list)
        self.phase_plot_panel.set_data(positions, phases_x_list, phases_y_list)

        
    def launch(self):
        
        class WindowCloser(WindowAdapter):
            def windowClosing(self, windowEvent):
                sys.exit(1)
        
        self.frame.addWindowListener(WindowCloser())
        self.frame.setSize(Dimension(1000, 800))
        self.frame.show()        
        
        
        
class EnergyTextFieldListener(ActionListener):
    
    def __init__(self, text_field, phase_controller):
        self.text_field = text_field
        self.phase_controller = phase_controller
        
    def actionPerformed(self, event):
        kin_energy = float(self.text_field.getText())
        self.phase_controller.probe.setKineticEnergy(1e9 * kin_energy)
        print 'Updated kin_energy to {:.3e} [eV]'.format(
            self.phase_controller.probe.getKineticEnergy())

        
class RefWsIdTextFieldListener(ActionListener):
    
    def __init__(self, dropdown, phase_controller):
        self.dropdown = dropdown
        self.phase_controller = phase_controller
        
    def actionPerformed(self, event):
        self.phase_controller.ref_ws_id = self.dropdown.getSelectedItem()
        print 'Updated ref_ws_id to {}'.format(self.phase_controller.ref_ws_id)
        

        
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
        self.data_beta_x = BasicGraphData()
        self.data_beta_y = BasicGraphData()
        for data, color in zip([self.data_beta_x, self.data_beta_y], COLOR_CYCLE):
            data.setGraphColor(color)
            data.setLineThick(3)
            data.setGraphPointSize(0)

    def set_data(self, positions, beta_x, beta_y):
        self.graph.removeAllGraphData()
        self.data_beta_x.addPoint(positions, beta_x)  
        self.data_beta_y.addPoint(positions, beta_y)  
        self.graph.addGraphData(self.data_beta_x)
        self.graph.addGraphData(self.data_beta_y)
            
            
gui = GUI()
gui.launch()