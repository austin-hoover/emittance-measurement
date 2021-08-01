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

# Local
import plotting as plt
from phase_controller import PhaseController
import utils
import xal_helpers


RTBT_WS_IDS = ['RTBT_Diag:WS02', 'RTBT_Diag:WS20', 'RTBT_Diag:WS21', 
               'RTBT_Diag:WS23', 'RTBT_Diag:WS24']


class PhaseControllerPanel(JPanel):
    
    def __init__(self):
        JPanel.__init__(self)
        self.setLayout(BorderLayout())
        self.phase_controller = PhaseController()
        self.model_fields_list = []
        # Get wire-scanner positions.
        self.ws_ids = RTBT_WS_IDS
        self.sequence = self.phase_controller.sequence
        self.start_node = self.sequence.getNodeWithId('Begin_Of_RTBT1')
        self.ws_positions = []
        for ws_id in self.ws_ids:
            ws_node = self.sequence.getNodeWithId(ws_id)
            ws_position = self.sequence.getDistanceBetween(self.start_node, ws_node)
            self.ws_positions.append(ws_position)
            
        self.build_panels()
        
    def build_panels(self):
        
        # Model calculation panel
        #------------------------------------------------------------------------
        # Labels
        init_twiss_label = JLabel('Initial Twiss')
        init_twiss_label.setAlignmentX(0)
        ref_ws_id_label = JLabel('Ref. wire-scanner')
        energy_label = JLabel('Energy [GeV]')
        phase_coverage_label = JLabel('Phase coverage [deg]')
        n_steps_label = JLabel('Total steps')
        max_beta_label = JLabel("<html>Max. &beta; [m/rad]<html>")

        # Components
        text_field_width = 11
        self.ref_ws_id_dropdown = JComboBox(self.ws_ids)
        self.init_twiss_table = JTable(InitTwissTableModel(self))
        self.init_twiss_table.setShowGrid(True)
        self.energy_text_field = JTextField('1.0', text_field_width)
        self.phase_coverage_text_field = JTextField('30.0', text_field_width)
        formatter = NumberFormat.getIntegerInstance()
        formatter.setGroupingUsed(False)
        self.n_steps_text_field = JFormattedTextField(formatter)
        self.n_steps_text_field.setValue(12)
        self.max_beta_text_field = JTextField('40.0', text_field_width)
        self.calculate_model_optics_button = JButton('Calculate model optics')

        # Action listeners
        self.energy_text_field.addActionListener(EnergyTextFieldListener(self))
        self.ref_ws_id_dropdown.addActionListener(RefWsIdDropdownListener(self))
        self.ref_ws_id_dropdown.setSelectedIndex(4)
        self.calculate_model_optics_button.addActionListener(CalculateModelOpticsButtonListener(self))   
        self.init_twiss_table.getCellEditor(0, 0).addCellEditorListener(TwissTableListener(self))

        # Build panel
        self.model_calc_panel1 = AlignedLabeledComponentsPanel()
        self.model_calc_panel1.add_row(ref_ws_id_label, self.ref_ws_id_dropdown)
        self.model_calc_panel2 = AlignedLabeledComponentsPanel()
        self.model_calc_panel2.add_row(energy_label, self.energy_text_field)
        self.model_calc_panel2.add_row(phase_coverage_label, self.phase_coverage_text_field)
        self.model_calc_panel2.add_row(n_steps_label, self.n_steps_text_field)
        self.model_calc_panel2.add_row(max_beta_label, self.max_beta_text_field)
        self.model_calc_panel = JPanel()
        self.model_calc_panel.add(init_twiss_label)
        self.model_calc_panel.add(self.init_twiss_table.getTableHeader())
        self.model_calc_panel.add(self.init_twiss_table)
        self.model_calc_panel.setLayout(BoxLayout(self.model_calc_panel, BoxLayout.Y_AXIS))
        self.model_calc_panel.add(self.model_calc_panel1)
        self.model_calc_panel.add(self.model_calc_panel2)


        # Machine update panel
        #------------------------------------------------------------------------ 
        # Labels
        sleep_time_label = JLabel('Sleep time [s]')
        max_frac_change_label = JLabel('Max. frac. field change')
        scan_index_label = JLabel('Scan index')

        # Components        
        self.sleep_time_text_field = JTextField('0.5', text_field_width)
        self.max_frac_change_text_field = JTextField('0.01', text_field_width)
        self.set_live_optics_button = JButton('Set live optics')
        self.quad_settings_table = JTable(QuadSettingsTableModel(self))
        self.quad_settings_table.setShowGrid(True)

        n_steps = int(self.n_steps_text_field.getText())
        scan_indices = ['default'] + list(range(n_steps))
        self.scan_index_dropdown = JComboBox(scan_indices)

        # Action listeners
        self.n_steps_text_field.addActionListener(NStepsTextFieldListener(self))
        self.set_live_optics_button.addActionListener(SetLiveOpticsButtonListener(self))

        # Build panel
        self.machine_update_panel = JPanel()
        self.machine_update_panel.setLayout(BoxLayout(self.machine_update_panel, BoxLayout.Y_AXIS))
        temp_panel = AlignedLabeledComponentsPanel()
        temp_panel.add_row(sleep_time_label, self.sleep_time_text_field)
        temp_panel.add_row(max_frac_change_label, self.max_frac_change_text_field)  
        temp_panel.add_row(scan_index_label, self.scan_index_dropdown)
        self.machine_update_panel.add(temp_panel)
        self.machine_update_panel.add(self.set_live_optics_button)
        self.machine_update_panel.add(self.quad_settings_table.getTableHeader())
        self.machine_update_panel.add(self.quad_settings_table)


        # Build left panel
        #------------------------------------------------------------------------    
        self.left_panel = JPanel()
        self.left_panel.setLayout(BoxLayout(self.left_panel, BoxLayout.Y_AXIS))

        label = JLabel('Compute model')
        font = label.getFont()
        label.setFont(Font(font.name, font.BOLD, int(1.1 * font.size)));
        temp_panel = JPanel()
        temp_panel.setLayout(FlowLayout(FlowLayout.LEFT))
        temp_panel.add(label)
        self.left_panel.add(temp_panel)

        self.left_panel.add(self.model_calc_panel)

        panel = JPanel()        
        temp_panel = JPanel()
        temp_panel.add(self.calculate_model_optics_button)

        self.progress_bar = JProgressBar(0, int(self.n_steps_text_field.getText()))
        self.progress_bar.setValue(0)
        self.progress_bar.setStringPainted(True)
        temp_panel.add(self.progress_bar)

        panel.add(temp_panel)
        self.left_panel.add(panel)

        label = JLabel('Update machine')
        label.setFont(Font(font.name, font.BOLD, int(1.1 * font.size)))
        temp_panel = JPanel()
        temp_panel.setLayout(FlowLayout(FlowLayout.LEFT))
        temp_panel.add(label)
        self.left_panel.add(temp_panel)

        self.left_panel.add(self.machine_update_panel)

        self.add(self.left_panel, BorderLayout.WEST)


        # Plotting panels
        #------------------------------------------------------------------------
        self.beta_plot_panel = plt.LinePlotPanel(
            xlabel='Position [m]', ylabel='[m/rad]', 
            title='Model beta function vs. position',
            n_lines=2, grid='y',
        )
        self.phase_plot_panel = plt.LinePlotPanel(
            xlabel='Position [m]', ylabel='Phase adv. mod 2pi', 
            title='Model phase advance vs. position',
            n_lines=2, grid='y',
        )
        self.bpm_plot_panel = plt.LinePlotPanel(
            xlabel='Position [m]', ylabel='Amplitude [mm]', title='BMP amplitudes',
            n_lines=2, lw=2, ms=5, grid='y',
        )

        # Get BPM positions
        self.bpms = self.phase_controller.sequence.getNodesOfType('BPM')
        ref_node = self.phase_controller.sequence.getNodes()[0]
        seq = self.phase_controller.sequence
        self.bpm_positions = [seq.getDistanceBetween(ref_node, node) for node in self.bpms]

        # Add the plots to the panel.
        self.right_panel = JPanel()
        self.right_panel.setLayout(BoxLayout(self.right_panel, BoxLayout.Y_AXIS))
        self.right_panel.add(self.beta_plot_panel)
        self.right_panel.add(self.phase_plot_panel)
        self.right_panel.add(self.bpm_plot_panel)
        self.add(self.right_panel, BorderLayout.CENTER)   
        self.update_plots()

    def read_bpms(self):
        x_avgs = list([bpm.getXAvg() for bpm in self.bpms])
        y_avgs = list([bpm.getYAvg() for bpm in self.bpms])
        return x_avgs, y_avgs

    def update_plots(self):
        betas_x, betas_y = [], []
        phases_x, phases_y = [], []
        self.phase_controller.track()
        for params in self.phase_controller.tracked_twiss():
            mu_x, mu_y, alpha_x, alpha_y, beta_x, beta_y, eps_x, eps_y = params
            betas_x.append(beta_x)
            betas_y.append(beta_y)
            phases_x.append(mu_x)
            phases_y.append(mu_y)
        positions = self.phase_controller.positions
        self.beta_plot_panel.set_data(positions, [betas_x, betas_y])
        self.phase_plot_panel.set_data(positions, [phases_x, phases_y])
        x_avgs, y_avgs = self.read_bpms()
        self.bpm_plot_panel.set_data(self.bpm_positions, [x_avgs, y_avgs])

        ref_ws_index = self.ws_ids.index(self.ref_ws_id_dropdown.getSelectedItem())
        for plot_panel in [self.beta_plot_panel, self.phase_plot_panel, self.bpm_plot_panel]:
            for i, ws_position in enumerate(self.ws_positions):
                color = Color(150, 150, 150) if i == ref_ws_index else Color(225, 225, 225)
                plot_panel.addVerticalLine(ws_position, color)

    def get_field_set_kws(self):
        field_set_kws = {
            'sleep_time': float(self.sleep_time_text_field.getText()),
            'max_frac_change': float(self.max_frac_change_text_field.getText()),
        }
        return field_set_kws


# Tables
#-------------------------------------------------------------------------------      
class QuadSettingsTableModel(AbstractTableModel):

    def __init__(self, panel):
        self.panel = panel
        self.phase_controller = panel.phase_controller
        self.quad_ids = self.phase_controller.ind_quad_ids
        self.column_names = ['Quad', 'Model [T/m]', 'Live [T/m]']
        self.nf4 = NumberFormat.getInstance()
        self.nf4.setMaximumFractionDigits(4)
        self.nf3 = NumberFormat.getInstance()
        self.nf3.setMaximumFractionDigits(3)

    def getValueAt(self, row, col):
        quad_id = self.quad_ids[row]
        if col == 0:
            return quad_id
        elif col == 1:
            return self.phase_controller.get_field(quad_id, 'model')
        elif col == 2:
            return self.phase_controller.get_field(quad_id, 'live')

    def getColumnCount(self):
        return len(self.column_names)

    def getRowCount(self):
        return len(self.quad_ids)

    def getColumnName(self, col):
        return self.column_names[col]


class InitTwissTableModel(AbstractTableModel):

    def __init__(self, panel):
        self.panel = panel
        self.phase_controller = panel.phase_controller
        self.column_names = ["<html>&alpha;<SUB>x</SUB> [m/rad]<html>",
                             "<html>&alpha;<SUB>y</SUB> [m/rad]<html>", 
                             "<html>&beta;<SUB>x</SUB> [m/rad]<html>", 
                             "<html>&beta;<SUB>y</SUB> [m/rad]<html>"]

    def getValueAt(self, row, col):
        if col == 0:
            return self.phase_controller.init_twiss['alpha_x']
        elif col == 1:
            return self.phase_controller.init_twiss['alpha_y']
        elif col == 2:
            return self.phase_controller.init_twiss['beta_x']
        elif col == 3:
            return self.phase_controller.init_twiss['beta_y']

    def getColumnCount(self):
        return len(self.column_names)

    def getRowCount(self):
        return 1

    def getColumnName(self, col):
        return self.column_names[col]

    def isCellEditable(self, row, col):
        return True

    
# Listeners
#-------------------------------------------------------------------------------
class EnergyTextFieldListener(ActionListener):

    def __init__(self, panel):
        self.panel = panel
        self.text_field = panel.energy_text_field
        self.phase_controller = panel.phase_controller
        
    def actionPerformed(self, event):
        kin_energy = float(self.text_field.getText())
        if kin_energy < 0:
            raise ValueError('Kinetic energy must be postive.')
        self.phase_controller.set_kin_energy(kin_energy)
        self.panel.init_twiss_table.getModel().fireTableDataChanged()
        self.phase_controller.track()
        self.panel.update_plots()
        print 'Updated kinetic energy to {:.3e} [eV]'.format(
            self.phase_controller.probe.getKineticEnergy())

        
class RefWsIdDropdownListener(ActionListener):
    
    def __init__(self, panel):
        self.panel = panel
        self.dropdown = panel.ref_ws_id_dropdown
        self.phase_controller = panel.phase_controller
        
    def actionPerformed(self, event):
        self.phase_controller.ref_ws_id = self.dropdown.getSelectedItem()
        if hasattr(self.panel, 'right_panel'):
            self.panel.update_plots()
        print 'Updated ref_ws_id to {}'.format(self.phase_controller.ref_ws_id)
        
        
class NStepsTextFieldListener(ActionListener):
    
    def __init__(self, panel):
        self.panel = panel
    
    def actionPerformed(self, event):
        n_steps = float(self.panel.n_steps_text_field.getText())
        n_steps = int(n_steps)
        self.panel.scan_index_dropdown.removeAllItems()
        self.panel.scan_index_dropdown.addItem('default')
        for scan_index in range(n_steps):
            self.panel.scan_index_dropdown.addItem(scan_index)
            
             
class TwissTableListener(CellEditorListener):
    
    def __init__(self, panel):
        self.panel = panel
        self.phase_controller = panel.phase_controller
        self.table = panel.init_twiss_table
        self.cell_editor = self.table.getCellEditor(0, 0)

    def editingStopped(self, event):
        value = float(self.cell_editor.getCellEditorValue())
        col = self.table.getSelectedColumn()
        key = ['alpha_x', 'alpha_y', 'beta_x', 'beta_y'][col]
        self.phase_controller.init_twiss[key] = value
        self.table.getModel().fireTableDataChanged()
        self.phase_controller.track()
        self.panel.update_plots()
        print 'Updated initial Twiss:', self.phase_controller.init_twiss


class CalculateModelOpticsButtonListener(ActionListener):

    def __init__(self, panel):
        self.panel = panel
        self.phase_controller = panel.phase_controller
        self.ind_quad_ids = self.phase_controller.ind_quad_ids
        
    def actionPerformed(self, event):
        """Calculate/store correct optics settings for each step in the scan.
        
        Output files
        ------------
        
        Let w = [WS20, WS21, WS23, WS24] be a list of the RTBT wire-scanners.
        Let i be the scan index. The following files are produced for each i:
        - 'model_transfer_matr_elems_i.dat': 
             The jth row in the file gives the 16 transfer matrix elements from 
             s = 0 to wire-scanner w[j]. The elements are written in the order: 
             [M11, M12, M13, M14, M21, M22, M23, M24, M31, M32, M33, M34, M41, 
             M42, M43, M44].
        - 'model_moments_i.dat': 
             The jth row in the file gives [Sigma_11, Sigma_33, Sigma_13] at 
             wire-scanner w[j], where Sigma_mn is the (m, n) entry in the 
             transverse covariance matrix.
        - 'model_fields_i.dat':
             Node ID and field strength of the independent quadrupoles [T/m].
        """
        self.panel.model_fields_list = []
        
        phase_coverage = float(self.panel.phase_coverage_text_field.getText())
        n_steps = int(self.panel.n_steps_text_field.getText())        
        max_beta = float(self.panel.max_beta_text_field.getText())
        beta_lims = (max_beta, max_beta)
        phases = self.phase_controller.get_phases_for_scan(phase_coverage, n_steps)
        print 'index | mu_x  | mu_y [rad]'
        print '---------------------'
        file = open('_output/phases.dat', 'w')
        for scan_index, (mu_x, mu_y) in enumerate(phases):
            print '{:<5} | {:.3f} | {:.3f}'.format(scan_index, mu_x, mu_y)
            file.write('{} {}\n'.format(mu_x, mu_y))
        file.close()
                
        for scan_index, (mu_x, mu_y) in enumerate(phases):
            
            # Set model optics.
            print 'Scan index {}/{}.'.format(scan_index, n_steps - 1)
            print 'Setting phases at {}...'.format(self.phase_controller.ref_ws_id)
            self.phase_controller.set_ref_ws_phases(mu_x, mu_y, beta_lims, verbose=1)
            print 'Setting betas at target...'
            self.phase_controller.constrain_size_on_target(verbose=1)
            max_betas_anywhere = self.phase_controller.max_betas(stop=None)
            print '  Max betas anywhere: {:.3f}, {:.3f}.'.format(*max_betas_anywhere)
            
#             # Save model Twiss vs. position data.
#             filename = '_output/model_twiss_{}.dat'.format(scan_index)
#             xal_helpers.write_traj_to_file(self.phase_controller.tracked_twiss(), 
#                                            self.phase_controller.positions, 
#                                            filename)
# 
#             # Save transfer matrix at each wire-scanner.
#             file = open('_output/model_transfer_mat_elems_{}_{}.dat'.format(scan_index, rec_node_id), 'w')
#             fstr = 16 * '{} ' + '\n'
#             for ws_id in RTBT_WS_IDS:
#                 M = self.phase_controller.transfer_matrix(rec_node_id, ws_id)
#                 elements = [elem for row in M for elem in row]
#                 file.write(fstr.format(*elements))
#             file.close()
# 
#             # Save real space beam moments at each wire-scanner.
#             file = open('_output/model_moments_{}.dat'.format(scan_index), 'w')
#             for ws_id in RTBT_WS_IDS:
#                 (mu_x, mu_y, alpha_x, alpha_y, 
#                  beta_x, beta_y, eps_x, eps_y) = self.phase_controller.twiss(ws_id)
#                 moments = [eps_x * beta_x, eps_y * beta_y, 0.0]
#                 file.write('{} {} {}\n'.format(*moments))
#             file.close()
#     
#             # Save model quadrupole strengths.
#             file = open('_output/model_fields_{}.dat'.format(scan_index), 'w')
#             model_fields = []
#             for quad_id in self.ind_quad_ids:
#                 field = self.phase_controller.get_field(quad_id, 'model')
#                 model_fields.append(field)
#                 file.write('{}, {}\n'.format(quad_id, field))
#             file.close()
            
            # Store the model fields.
            self.panel.model_fields_list.append(model_fields)
            
            # Update the panel progress bar. (This doesn't work currently;
            # we would need to run on a separate thread.)
            self.panel.progress_bar.setValue(scan_index + 1)
                    
            print ''
            
        # Put the model back to its original state.
        self.phase_controller.restore_default_optics('model')
        
        
class SetLiveOpticsButtonListener(ActionListener):
    
    def __init__(self, panel):
        self.panel = panel
        self.phase_controller = panel.phase_controller
        self.ind_quad_ids = self.phase_controller.ind_quad_ids
        
    def actionPerformed(self, action):
        quad_ids = self.phase_controller.ind_quad_ids
        field_set_kws = self.panel.get_field_set_kws()
        scan_index = self.panel.scan_index_dropdown.getSelectedItem()
        print 'Syncing live quads with model...'
        print field_set_kws
        if scan_index == 'default':
            self.phase_controller.restore_default_optics('model')
            self.phase_controller.restore_default_optics('live')
        else:
            scan_index = int(scan_index)
            model_fields = self.panel.model_fields_list[scan_index]
            self.phase_controller.set_fields(quad_ids, model_fields, 'model')
            self.phase_controller.set_fields(quad_ids, model_fields, 'live', **field_set_kws)
        self.panel.quad_settings_table.getModel().fireTableDataChanged()
        self.panel.update_plots()
        
        # Save live quad strengths.
#         file = open('_output/live_fields_{}.dat'.format(scan_index), 'w')
#         for quad_id in self.ind_quad_ids:
#             field = self.phase_controller.get_field(quad_id, 'live')
#             file.write('{}, {}\n'.format(quad_id, field))
#         file.close()
        
        # Save transfer matrix at each wire-scanner.
#         file = open('_output/model_transfer_mat_elems_{}_{}.dat'.format(scan_index, rec_node_id), 'w')
#         fstr = 16 * '{} ' + '\n'
#         for ws_id in self.ws_ids:
#             M = self.phase_controller.transfer_matrix(rec_node_id, ws_id)
#             elements = [elem for row in M for elem in row]
#             file.write(fstr.format(*elements))
#         file.close()
        
        # Save expected Twiss parameters at reconstruction location.
#         file = open('_output/model_twiss_{}.dat'.format(rec_node_id), 'w')
#         (mu_x, mu_y, alpha_x, alpha_y, 
#          beta_x, beta_y, eps_x, eps_y) = self.phase_controller.twiss(rec_node_id)
#         file.write('{} {} {} {}'.format(alpha_x, alpha_y, beta_x, beta_y))
#         file.close()

            
            
# Miscellaneous
#-------------------------------------------------------------------------------
class AlignedLabeledComponentsPanel(JPanel):
    
    def __init__(self):
        JPanel.__init__(self)
        self.layout = GroupLayout(self)
        self.setLayout(self.layout)
        self.layout.setAutoCreateContainerGaps(True)
        self.layout.setAutoCreateGaps(True)
        self.group_labels = self.layout.createParallelGroup()
        self.group_components = self.layout.createParallelGroup()
        self.group_rows = self.layout.createSequentialGroup()
        self.layout.setHorizontalGroup(self.layout.createSequentialGroup()
                                  .addGroup(self.group_labels)
                                  .addGroup(self.group_components))
        self.layout.setVerticalGroup(self.group_rows)

    def add_row(self, label, component):
        self.group_labels.addComponent(label)
        self.group_components.addComponent(component)
        self.group_rows.addGroup(
            self.layout.createParallelGroup()
                .addComponent(label)
                .addComponent(
                    component, 
                    GroupLayout.PREFERRED_SIZE, 
                    GroupLayout.DEFAULT_SIZE, 
                    GroupLayout.PREFERRED_SIZE
                )
        )