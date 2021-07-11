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
from lib import utils


COLOR_CYCLE = [
    Color(0.0, 0.44705882, 0.69803922),
    Color(0.83529412, 0.36862745, 0.0),
    Color(0.0, 0.61960784, 0.45098039),
    Color(0.8, 0.4745098, 0.65490196),
    Color(0.94117647, 0.89411765, 0.25882353),
    Color(0.3372549, 0.70588235, 0.91372549),
]


ws_ids = ['RTBT_Diag:WS20', 'RTBT_Diag:WS21', 'RTBT_Diag:WS23', 'RTBT_Diag:WS24']
                

class GUI:
    
    def __init__(self, live=True):
        self.live = live
        self.phase_controller = PhaseController()

        # Main frame
        #------------------------------------------------------------------------
        self.frame = JFrame("RTBT Phase Controller")
        self.frame.setSize(Dimension(1000, 800))
        self.frame.getContentPane().setLayout(BorderLayout())
        time_text = DateAndTimeText()
        time_panel = JPanel(BorderLayout())
        time_panel.add(time_text.getTimeTextField(), BorderLayout.CENTER)
        self.frame.add(time_panel, BorderLayout.SOUTH)

        # Text fields panel
        #------------------------------------------------------------------------
        # Labels
        ref_ws_id_label = JLabel('Ref. wire-scanner')
        energy_label = JLabel('Energy [GeV]')
        phase_coverage_label = JLabel('Phase coverage [deg]')
        n_steps_label = JLabel('Total steps')
        max_beta_label = JLabel("<html>Max. &beta; [m/rad]<html>")
        sleep_time_label = JLabel('Sleep time [s]')
        max_frac_change_label = JLabel('Max. frac. field change')
        
        # Text fields / dropdown menus
        text_field_width = 12 
        self.ref_ws_id_dropdown = JComboBox(ws_ids)
        self.energy_text_field = JTextField('1.0', text_field_width)
        self.phase_coverage_text_field = JTextField('180.0', text_field_width)
        self.n_steps_text_field = JTextField('12', text_field_width)
        self.max_beta_text_field = JTextField('40.0', text_field_width)
        self.sleep_time_text_field = JTextField('0.5', text_field_width)
        self.max_frac_change_text_field = JTextField('0.01', text_field_width)
        
        # Add action listeners
        self.energy_text_field.addActionListener(EnergyTextFieldListener(self.energy_text_field, self.phase_controller))
        self.ref_ws_id_dropdown.addActionListener(RefWsIdTextFieldListener(self.ref_ws_id_dropdown, self.phase_controller))
        self.ref_ws_id_dropdown.setSelectedIndex(3)
        
        # Build text fields panel
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
    
        add_field(ref_ws_id_label, self.ref_ws_id_dropdown)
        add_field(energy_label, self.energy_text_field)
        add_field(phase_coverage_label, self.phase_coverage_text_field)
        add_field(n_steps_label, self.n_steps_text_field)
        add_field(max_beta_label, self.max_beta_text_field)
        add_field(sleep_time_label, self.sleep_time_text_field)
        add_field(max_frac_change_label, self.max_frac_change_text_field)
        
        # Buttons panel
        #------------------------------------------------------------------------               
        # Create buttons
        self.calc_optics_button = JButton('Calculate model optics')
        self.set_optics_button = JButton('Set live optics')
        
        # Add action listeners
        
        # Build buttons panel
        self.buttons_panel = JPanel()
        self.buttons_panel.setLayout(BoxLayout(self.buttons_panel, BoxLayout.Y_AXIS))
        self.buttons_panel.add(self.calc_optics_button)        
        self.buttons_panel.add(self.set_optics_button)
        
        # Build left panel
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
        
        self.left_panel.add(self.buttons_panel)
        
        self.frame.add(self.left_panel, BorderLayout.WEST)
        
        # Plotting panels
        #------------------------------------------------------------------------
        self.beta_plot_panel = LinePlotPanel(xlabel='Position [m]', 
                                             ylabel='[m/rad]', 
                                             title='Model beta function vs. position',
                                             n_lines=2)
        self.phase_plot_panel = LinePlotPanel(xlabel='Position [m]', 
                                              ylabel='Phase adv. mod 2pi', 
                                              title='Model phase advance vs. position',
                                              n_lines=2)
        self.bpm_plot_panel = LinePlotPanel(xlabel='BPM', 
                                            ylabel='Amplitude [mm]', 
                                            title='BMP amplitudes',
                                            n_lines=2)
        self.right_panel = JPanel()
        self.right_panel.setLayout(BoxLayout(self.right_panel, BoxLayout.Y_AXIS))
        self.right_panel.add(self.beta_plot_panel)
        self.right_panel.add(self.phase_plot_panel)
        self.right_panel.add(self.bpm_plot_panel)
        self.frame.add(self.right_panel, BorderLayout.CENTER)   
        self.update_plots()
    
        
    def update_plots(self):
        betas_x, betas_y = [], []
        phases_x, phases_y = [], []
        for params in self.phase_controller.tracked_twiss():
            mu_x, mu_y, alpha_x, alpha_y, beta_x, beta_y, eps_x, eps_y = params
            betas_x.append(beta_x)
            betas_y.append(beta_y)
            phases_x.append(mu_x)
            phases_y.append(mu_y)
        positions = self.phase_controller.positions
        self.beta_plot_panel.set_data(positions, [betas_x, betas_y])
        self.phase_plot_panel.set_data(positions, [phases_x, phases_y])

        
    def launch(self):
    
        class WindowCloser(WindowAdapter):
            def __init__(self, phase_controller, field_set_kws, live=True):
                self.phase_controller = phase_controller
                self.field_set_kws = field_set_kws
                self.live = live

            def windowClosing(self, event):
                """Reset the real machine to its default state before closing window."""
                if self.live:
                    self.phase_controller.restore_default_optics('live', **self.field_set_kws)
                sys.exit(1)

        field_set_kws = {
            'sleep_time': float(self.sleep_time_text_field.getText()),
            'max_frac_change': float(self.max_frac_change_text_field.getText()),
        }
        self.frame.addWindowListener(WindowCloser(self.phase_controller, field_set_kws, self.live))
        self.frame.show()        
        
        
class EnergyTextFieldListener(ActionListener):
    """Update the beam kinetic energy from the text field; retrack; replot."""
    def __init__(self, text_field, phase_controller):
        self.text_field = text_field
        self.phase_controller = phase_controller
        
    def actionPerformed(self, event):
        kin_energy = float(self.text_field.getText())
        self.phase_controller.set_kin_energy(kin_energy)
        self.phase_controller.track()
        print 'Updated kin_energy to {:.3e} [eV]'.format(
            self.phase_controller.probe.getKineticEnergy())

        
class RefWsIdTextFieldListener(ActionListener):
    """Update the reference wire-scanner ID from the text field; retrack; replot."""
    def __init__(self, dropdown, phase_controller):
        self.dropdown = dropdown
        self.phase_controller = phase_controller
        
    def actionPerformed(self, event):
        self.phase_controller.ref_ws_id = self.dropdown.getSelectedItem()
        self.phase_controller.track()
        print 'Updated ref_ws_id to {}'.format(self.phase_controller.ref_ws_id)
        
        
class CalculateModelOpticsButtonListener(ActionListener):
    """Calculate the model optics to obtain selected phase advances."""
    def __init__(self, phase_controller):
        self.phase_controller = phase_controller
        
    def actionPerformed(self, event):
        # 1. Get the phase advances from the GUI.
        # 2. Run the solver.
        # 3. Print the output.
        raise NotImplementedError
    
    
class LinePlotPanel(JPanel):
    """Class for 2D line plots."""
    def __init__(self, xlabel='', ylabel='', title='', n_lines=2):
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
        self.n_lines = n_lines
        self.data_list = [BasicGraphData() for _ in range(n_lines)]
        for data, color in zip(self.data_list, COLOR_CYCLE):
            data.setGraphColor(color)
            data.setLineThick(3)
            data.setGraphPointSize(0)

    def set_data(self, x, y_list):
        if len(utils.shape(y_list)) == 1:
            y_list = [y_list]
        self.graph.removeAllGraphData()
        for data, y in zip(self.data_list, y_list):
            data.addPoint(x, y)  
            self.graph.addGraphData(data)        
            
            
gui = GUI()
gui.launch()