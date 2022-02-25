"""This panel controls the RTBT optics."""
from __future__ import print_function

from java.awt import BorderLayout
from java.awt import Color
from java.awt import Dimension
from java.awt import FlowLayout
from java.awt import GridLayout
from java.awt import GridBagLayout
from java.awt import GridBagConstraints
from java.awt import Insets
from java.awt import Font
from java.awt.event import ActionListener
from javax.swing import BorderFactory
from javax.swing import BoxLayout
from javax.swing import GroupLayout
from javax.swing import JButton
from javax.swing import JComboBox
from javax.swing import JLabel
from javax.swing import JPanel
from javax.swing import JProgressBar
from javax.swing import JScrollPane
from javax.swing import JTable
from javax.swing import JTabbedPane
from javax.swing import JTextField
from javax.swing import JFormattedTextField
from javax.swing.event import CellEditorListener
from javax.swing.table import AbstractTableModel
from java.text import NumberFormat

import optics
import plotting as plt
import utils


SLEEP_TIME = 0.5 # Pause between changing quad strengths [seconds]
MAX_FRAC_CHANGE = 0.01 # Maximum fractional change of quad strengths in a single step.
FIELD_SET_KWS = {
    "sleep_time": SLEEP_TIME,
    "max_frac_change": MAX_FRAC_CHANGE,
}

class PhaseControllerPanel(JPanel):
    def __init__(self):
        JPanel.__init__(self)
        self.setLayout(GridBagLayout())
        self.phase_controller = optics.PhaseController(kinetic_energy=1e9)
        self.model_fields_list = []
        self.ws_ids = optics.RTBT_WS_IDS
        self.sequence = self.phase_controller.sequence
        self.start_node = self.sequence.getNodeWithId("Begin_Of_RTBT1")
        self.ws_positions = []
        for ws_id in self.ws_ids:
            ws_node = self.sequence.getNodeWithId(ws_id)
            ws_position = self.sequence.getDistanceBetween(self.start_node, ws_node)
            self.ws_positions.append(ws_position)
        self.build_panels()

    def build_panels(self):
        # Scan optics panel
        # ------------------------------------------------------------------------
        # Components
        self.ref_ws_id_dropdown = JComboBox(["RTBT_Diag:WS20", "RTBT_Diag:WS21",
                                             "RTBT_Diag:WS23", "RTBT_Diag:WS24"])        
        self.init_twiss_table = JTable(InitTwissTableModel(self))
        self.init_twiss_table.setShowGrid(True)
        self.energy_text_field = JTextField("1.000")
        self.phase_coverage_text_field = JTextField("30.0  ")
        formatter = NumberFormat.getIntegerInstance()
        formatter.setGroupingUsed(False)
        self.n_steps_text_field = JFormattedTextField(formatter)
        self.n_steps_text_field.setValue(10)
        self.scan_type_dropdown = JComboBox([1, 2])
        self.max_beta_text_field = JTextField("30.0")
        self.calculate_scan_optics_button = JButton("Calculate optics")

        # Action listeners
        self.energy_text_field.addActionListener(EnergyTextFieldListener(self))
        self.ref_ws_id_dropdown.addActionListener(RefWsIdDropdownListener(self))
        self.ref_ws_id_dropdown.setSelectedIndex(3)
        for component in [
            self.phase_coverage_text_field,
            self.n_steps_text_field,
            self.scan_type_dropdown
        ]:
            component.addActionListener(ScanSettingsListener(self))
        self.calculate_scan_optics_button.addActionListener(
            CalculateScanOpticsButtonListener(self)
        )
        self.init_twiss_table.getCellEditor(0, 0).addCellEditorListener(
            TwissTableListener(self)
        )

        # Build panel
        scan_optics_panel = JPanel()
        scan_optics_panel.setLayout(GridBagLayout())

        c = GridBagConstraints()
        c.anchor = GridBagConstraints.NORTHWEST
        c.gridwidth = GridBagConstraints.REMAINDER
        c.fill = GridBagConstraints.HORIZONTAL
        c.weightx = 1.0

        scan_optics_panel.add(JLabel('Scan optics'), c)
        
        row = JPanel()
        row.setLayout(FlowLayout(FlowLayout.LEFT))
        row.add(JLabel("Ref. wire-scanner"))
        row.add(self.ref_ws_id_dropdown)
        scan_optics_panel.add(row, c)

        row = JPanel()
        row.setLayout(FlowLayout(FlowLayout.LEFT))
        row.add(JLabel("Energy [GeV]"))
        row.add(self.energy_text_field)
        row.add(JLabel("<html>Max. &beta; [m/rad]<html>"))
        row.add(self.max_beta_text_field)
        scan_optics_panel.add(row, c)

        row = JPanel()
        row.setLayout(FlowLayout(FlowLayout.LEFT))
        row.add(JLabel("Coverage [deg]"))
        row.add(self.phase_coverage_text_field)
        row.add(JLabel("Steps"))
        row.add(self.n_steps_text_field)
        row.add(JLabel("Scan type"))
        row.add(self.scan_type_dropdown)
        scan_optics_panel.add(row, c)

        c.weighty = 1.0
        scan_optics_panel.add(JPanel(), c)
        c.weighty = 0.0

        self.phase_scan_plot_panel = plt.LinePlotPanel(
            xlabel='Scan index',
            ylabel='Phase adv.',
            n_lines=2,
            grid='y',
            ms=8,
        )
        self.phase_scan_plot_panel.setBorder(BorderFactory.createEmptyBorder())
        xvals = list(range(10))
        yvals1 = list(range(10))
        yvals2 = list(reversed(range(10)))
        self.phase_scan_plot_panel.set_data(xvals, [yvals1, yvals2])


        # Machine update panel
        # ------------------------------------------------------------------------
        # Components
        self.set_live_optics_button1 = JButton("Set from scan")
        self.set_live_optics_button2 = JButton("Set manually ")
        self.quad_settings_table = JTable(QuadSettingsTableModel(self))
        self.quad_settings_table.setShowGrid(True)
        n_steps = int(self.n_steps_text_field.getText())
        self.scan_index_dropdown = JComboBox(["default"] + list(range(n_steps)))
        self.delta_mux_text_field = JTextField("0.0    ")
        self.delta_muy_text_field = JTextField("0.0    ")

        # Action listeners
        self.n_steps_text_field.addActionListener(NStepsTextFieldListener(self))
        self.set_live_optics_button1.addActionListener(
            SetLiveOpticsButton1Listener(self)
        )
        self.set_live_optics_button2.addActionListener(
            SetLiveOpticsButton2Listener(self)
        )

        # Build panel
        machine_update_panel = JPanel()
        machine_update_panel.setLayout(GridBagLayout())

        c = GridBagConstraints()
        c.anchor = GridBagConstraints.NORTHWEST
        c.gridwidth = GridBagConstraints.REMAINDER
        c.weightx = 1.0

        row = JPanel()
        row.setLayout(FlowLayout(FlowLayout.LEFT))
        row.add(JLabel('Set live optics'))
        machine_update_panel.add(row, c)
        
        row = JPanel()
        row.setLayout(FlowLayout(FlowLayout.LEFT))
        row.add(self.set_live_optics_button1)
        row.add(JLabel("Scan index"))
        row.add(self.scan_index_dropdown)
        machine_update_panel.add(row, c)
        
        row = JPanel()
        row.setLayout(FlowLayout(FlowLayout.LEFT))
        row.add(self.set_live_optics_button2)
        row.add(JLabel("<html>&Delta&mu;<SUB>x</SUB> [deg]<html>"))
        row.add(self.delta_mux_text_field)
        row.add(JLabel("<html>&Delta&mu;<SUB>y</SUB> [deg]<html>"))
        row.add(self.delta_muy_text_field)
        machine_update_panel.add(row, c)
        
        machine_update_panel.add(self.quad_settings_table.getTableHeader(), c)
        machine_update_panel.add(JScrollPane(self.quad_settings_table), c)
        c.weighty = 1.0
        machine_update_panel.add(JPanel(), c)
        c.weighty = 0.0


        # Plot panel
        # ------------------------------------------------------------------------
        self.plot_panel = JPanel()
        self.plot_panel.setLayout(BoxLayout(self.plot_panel, BoxLayout.Y_AXIS))
        self.beta_plot_panel = plt.LinePlotPanel(
            xlabel="Position [m]",
            ylabel="Beta function [m/rad]",
            n_lines=2,
            grid="y",
        )
        self.beta_plot_panel.setLimitsAndTicksY(0.0, 100.0, 10.0)
        self.beta_plot_panel.legend([' x', ' y'])
        self.plot_panel.add(self.beta_plot_panel)
        self.phase_plot_panel = plt.LinePlotPanel(
            xlabel="Position [m]",
            ylabel="Phase adv. mod 2pi [rad]",
            n_lines=2,
            grid="y",
        )
        self.phase_plot_panel.legend([' x', ' y'])
        self.plot_panel.add(self.phase_plot_panel)

        rtbt_length = self.phase_controller.sequence.getDistanceBetween(
            self.phase_controller.sequence.getNodes()[0],
            self.phase_controller.sequence.getNodes()[-1],
        )
        for panel in [self.phase_plot_panel, self.beta_plot_panel]:
            panel.set_xlim(0.0, rtbt_length, 20.0)
            panel.setBorder(BorderFactory.createEmptyBorder())
        self.update_plots()


        # Build the main panel
        # ------------------------------------------------------------------------
        left_panel = JPanel()
        left_panel.setBorder(BorderFactory.createEtchedBorder())

        _pan = JPanel()
        _pan.setLayout(BoxLayout(_pan, BoxLayout.Y_AXIS))
        _pan.add(scan_optics_panel)
        _pan.add(self.phase_scan_plot_panel)
        self.phase_scan_plot_panel.setPreferredSize(Dimension(400, 250))
        row = JPanel()
        row.add(self.calculate_scan_optics_button)
        self.progress_bar = JProgressBar(0, int(self.n_steps_text_field.getText()))
        self.progress_bar.setValue(0)
        self.progress_bar.setStringPainted(True)
        row.add(self.progress_bar)
        _pan.add(row)

        _pan.setBorder(BorderFactory.createEtchedBorder())
        machine_update_panel.setBorder(BorderFactory.createEtchedBorder())

        left_panel.setLayout(GridBagLayout())
        c = GridBagConstraints()
        c.gridwidth = GridBagConstraints.REMAINDER
        c.anchor = GridBagConstraints.NORTHWEST
        c.fill = GridBagConstraints.HORIZONTAL
        left_panel.add(_pan, c)
        c.weighty = 1.0
        left_panel.add(JPanel(), c)
        c.weighty = 0.0
        c.anchor = GridBagConstraints.SOUTHWEST
        left_panel.add(machine_update_panel, c)

        right_panel = JPanel()
        right_panel.setBorder(BorderFactory.createEtchedBorder())
        right_panel.setLayout(BoxLayout(right_panel, BoxLayout.Y_AXIS))
        temp = JPanel()
        temp.add(JButton("Hello"))
        right_panel.add(temp)
        right_panel.add(self.beta_plot_panel)
        right_panel.add(self.phase_plot_panel)

        c = GridBagConstraints()
        c.weightx = 0.01
        c.weighty = 0.5
        c.fill = GridBagConstraints.VERTICAL
        c.anchor = GridBagConstraints.NORTHWEST
        self.add(left_panel, c)
        c.weightx = 1.0
        c.gridwidth = 5
        c.fill = GridBagConstraints.BOTH
        self.add(right_panel, c)



    def update_plots(self):
        # Plot model beta functions and phase advances.
        betas_x, betas_y = [], []
        phases_x, phases_y = [], []
        self.phase_controller.track()
        for params in self.phase_controller.tracked_twiss():
            mux, muy, alpha_x, alpha_y, beta_x, beta_y, eps_x, eps_y = params
            betas_x.append(beta_x)
            betas_y.append(beta_y)
            phases_x.append(mux)
            phases_y.append(muy)
        positions = self.phase_controller.positions
        self.beta_plot_panel.set_data(positions, [betas_x, betas_y])
        self.phase_plot_panel.set_data(positions, [phases_x, phases_y])

        # Add a vertical line at each wire-scanner locations.
        ref_ws_index = self.ws_ids.index(self.ref_ws_id_dropdown.getSelectedItem())
        for plot_panel in [self.beta_plot_panel, self.phase_plot_panel]:
            for i, ws_position in enumerate(self.ws_positions):
                color = (
                    Color(150, 150, 150) if i == ref_ws_index else Color(225, 225, 225)
                )
                plot_panel.addVerticalLine(ws_position, color)

        # Show phase scan
        phase_coverage = float(self.phase_coverage_text_field.getText())
        n_steps = int(self.n_steps_text_field.getText())
        scan_type = self.scan_type_dropdown.getSelectedItem()
        phases = self.phase_controller.get_phases_for_scan(
            phase_coverage, n_steps, scan_type
        )
        self.phase_scan_plot_panel.set_data(
            list(range(n_steps)),
            [[mux for (mux, muy) in phases],
             [muy for (mux, muy) in phases]],
        )


# Tables
# -------------------------------------------------------------------------------
class QuadSettingsTableModel(AbstractTableModel):
    def __init__(self, panel):
        self.panel = panel
        self.phase_controller = panel.phase_controller
        self.quad_ids = self.phase_controller.ind_quad_ids
        self.column_names = ["Quad", "Model [T/m]", "Live [T/m]"]
        self.decimals = 3

    def getValueAt(self, row, col):
        quad_id = self.quad_ids[row]
        if col == 0:
            return quad_id
        elif col == 1:
            return round(self.phase_controller.get_field(quad_id, "model"), self.decimals)
        elif col == 2:
            return round(self.phase_controller.get_field(quad_id, "live"), self.decimals)

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
        self.decimals = 3
        self.names = [
            "<html>&alpha;<SUB>x</SUB><html>",
            "<html>&alpha;<SUB>y</SUB><html>",
            "<html>&beta;<SUB>x</SUB> [m/rad]<html>",
            "<html>&beta;<SUB>y</SUB> [m/rad]<html>",
        ]

    def getValueAt(self, row, col):
        if col == 0:
            return self.names[row]
        if row == 0:
            return round(self.phase_controller.init_twiss["alpha_x"], self.decimals)
        elif row == 1:
            return round(self.phase_controller.init_twiss["alpha_y"], self.decimals)
        elif row == 2:
            return round(self.phase_controller.init_twiss["beta_x"], self.decimals)
        elif row == 3:
            return round(self.phase_controller.init_twiss["beta_y"], self.decimals)

    def getColumnCount(self):
        return 2

    def getRowCount(self):
        return 4

    def getColumnName(self, col):
        if col == 0:
            return 'Initial Twiss parameter'
        elif col == 1:
            return 'Value'

    def isCellEditable(self, row, col):
        return True


# Listeners
# -------------------------------------------------------------------------------
class EnergyTextFieldListener(ActionListener):
    def __init__(self, panel):
        self.panel = panel
        self.text_field = panel.energy_text_field
        self.phase_controller = panel.phase_controller

    def actionPerformed(self, event):
        kin_energy = 1e9 * float(self.text_field.getText())
        if kin_energy < 0.0:
            raise ValueError("Kinetic energy must be positive.")
        self.phase_controller.set_kinetic_energy(kin_energy)
        self.panel.init_twiss_table.getModel().fireTableDataChanged()
        self.phase_controller.track()
        self.panel.update_plots()
        print(
            "Updated kinetic energy to {:.3e} [eV]".format(
                self.phase_controller.probe.getKineticEnergy()
            )
        )


class RefWsIdDropdownListener(ActionListener):
    def __init__(self, panel):
        self.panel = panel
        self.dropdown = panel.ref_ws_id_dropdown
        self.phase_controller = panel.phase_controller

    def actionPerformed(self, event):
        self.phase_controller.ref_ws_id = self.dropdown.getSelectedItem()
        if hasattr(self.panel, "plot_panel"):
            self.panel.update_plots()
        print("Updated ref_ws_id to {}".format(self.phase_controller.ref_ws_id))


class NStepsTextFieldListener(ActionListener):
    def __init__(self, panel):
        self.panel = panel

    def actionPerformed(self, event):
        n_steps = float(self.panel.n_steps_text_field.getText())
        n_steps = int(n_steps)
        self.panel.scan_index_dropdown.removeAllItems()
        self.panel.scan_index_dropdown.addItem("default")
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
        row = self.table.getSelectedRow()
        col = self.table.getSelectedColumn()
        key = ["alpha_x", "alpha_y", "beta_x", "beta_y"][row]
        self.phase_controller.init_twiss[key] = value
        self.table.getModel().fireTableDataChanged()
        self.phase_controller.track()
        self.panel.update_plots()
        print("Updated initial Twiss:", self.phase_controller.init_twiss)


class CalculateScanOpticsButtonListener(ActionListener):
    def __init__(self, panel):
        self.panel = panel
        self.phase_controller = panel.phase_controller
        self.ind_quad_ids = self.phase_controller.ind_quad_ids

    def actionPerformed(self, event):
        """Calculate/store correct optics settings for each step in the scan."""
        self.panel.model_fields_list = []

        # Start from the default optics.
        self.phase_controller.restore_default_optics("model")
        self.phase_controller.track()

        # Make a list of phase advances.
        phase_coverage = float(self.panel.phase_coverage_text_field.getText())
        n_steps = int(self.panel.n_steps_text_field.getText())
        max_beta = float(self.panel.max_beta_text_field.getText())
        beta_lims = (max_beta, max_beta)
        scan_type = self.panel.scan_type_dropdown.getSelectedItem()
        phases = self.phase_controller.get_phases_for_scan(
            phase_coverage, n_steps, scan_type
        )
        print("index | mux  | muy [rad]")
        print("---------------------")
        for scan_index, (mux, muy) in enumerate(phases):
            print("{:<5} | {:.3f} | {:.3f}".format(scan_index, mux, muy))

        # Compute the optics needed for step in the scan.
        self.panel.progress_bar.setValue(0)
        self.panel.progress_bar.setMaximum(n_steps)

        for scan_index, (mux, muy) in enumerate(phases):
            # Set the model optics.
            print("Scan index = {}/{}.".format(scan_index, n_steps - 1))
            print("Setting phases at {}...".format(self.phase_controller.ref_ws_id))
            self.phase_controller.set_ref_ws_phases(mux, muy, beta_lims, verbose=1)

            # Constrain beam size on target if it's too far from the default.
            beta_x_target, beta_y_target = self.phase_controller.beta_funcs("RTBT:Tgt")
            beta_x_default, beta_y_default = self.phase_controller.default_betas_at_target
            frac_change_x = abs(beta_x_target - beta_x_default) / beta_x_default
            frac_change_y = abs(beta_y_target - beta_y_default) / beta_y_default
            tol = 0.05
            if frac_change_x > tol or frac_change_y > tol:
                print("Setting betas at target...")
                self.phase_controller.constrain_size_on_target(verbose=1)
            max_betas_anywhere = self.phase_controller.max_betas(stop=None)
            print("  Max betas anywhere: {:.3f}, {:.3f}.".format(*max_betas_anywhere))

            # Save the model quadrupole strengths.
            model_fields = []
            for quad_id in self.ind_quad_ids:
                field = self.phase_controller.get_field(quad_id, "model")
                model_fields.append(field)

            # Store the model fields.
            self.panel.model_fields_list.append(model_fields)

            # Update the panel progress bar. (This doesn't work currently;
            # we would need to run on a separate thread.)
            self.panel.progress_bar.setValue(scan_index + 1)
            print()

        # Put the model back to its original state.
        self.phase_controller.restore_default_optics("model")
        self.phase_controller.track()


class ScanSettingsListener(ActionListener):
    def __init__(self, panel):
        self.panel = panel

    def actionPerformed(self, action):
        self.panel.update_plots()


class SetLiveOpticsButton1Listener(ActionListener):
    def __init__(self, panel):
        self.panel = panel
        self.phase_controller = panel.phase_controller
        self.ind_quad_ids = self.phase_controller.ind_quad_ids

    def actionPerformed(self, action):
        quad_ids = self.phase_controller.ind_quad_ids
        scan_index = self.panel.scan_index_dropdown.getSelectedItem()
        print("Syncing live quads with model...")
        print(FIELD_SET_KWS)
        if scan_index == "default":
            self.phase_controller.restore_default_optics("model")
            self.phase_controller.restore_default_optics("live")
        else:
            scan_index = int(scan_index)
            fields = self.panel.model_fields_list[scan_index]
            self.phase_controller.set_fields(quad_ids, fields, "model")
            self.phase_controller.set_fields(quad_ids, fields, "live", **FIELD_SET_KWS)
        self.panel.quad_settings_table.getModel().fireTableDataChanged()
        self.panel.update_plots()
        print("Done.")


class SetLiveOpticsButton2Listener(ActionListener):
    def __init__(self, panel):
        self.panel = panel
        self.phase_controller = panel.phase_controller
        self.ind_quad_ids = self.phase_controller.ind_quad_ids

    def actionPerformed(self, action):
        # Set the phase advance at the reference wire-scanner.
        self.phase_controller.restore_default_optics("model")
        self.phase_controller.track()
        mux0, muy0 = self.phase_controller.phases(self.phase_controller.ref_ws_id)
        delta_mux = utils.radians(float(self.panel.delta_mux_text_field.getText()))
        delta_muy = utils.radians(float(self.panel.delta_muy_text_field.getText()))
        mux = utils.put_angle_in_range(mux0 + delta_mux)
        muy = utils.put_angle_in_range(muy0 + delta_muy)
        print(
            "mux0, muy0 = {}, {} [deg]".format(utils.degrees(mux0), utils.degrees(muy0))
        )
        print("mux, muy = {}, {} [deg]".format(utils.degrees(mux), utils.degrees(muy)))
        print("Setting model phase advances...")
        self.phase_controller.set_ref_ws_phases(mux, muy, verbose=2)

        # Constrain the beam size on the target if it's too far from the default.
        beta_x_target, beta_y_target = self.phase_controller.beta_funcs("RTBT:Tgt")
        beta_x_default, beta_y_default = self.phase_controller.default_betas_at_target
        frac_change_x = abs(beta_x_target - beta_x_default) / beta_x_default
        frac_change_y = abs(beta_y_target - beta_y_default) / beta_y_default
        tol = 0.05
        if frac_change_x > tol or frac_change_y > tol:
            print("Setting betas at target...")
            self.phase_controller.constrain_size_on_target(verbose=1)
        max_betas_anywhere = self.phase_controller.max_betas(stop=None)
        print("  Max betas anywhere: {:.3f}, {:.3f}.".format(*max_betas_anywhere))

        # Sync the live optics with the model.
        print("Syncing live quads with model...")
        print(FIELD_SET_KWS)
        self.phase_controller.sync_live_with_model(**FIELD_SET_KWS)
        self.panel.quad_settings_table.getModel().fireTableDataChanged()
        self.panel.update_plots()
        print("Done.")


# Miscellaneous
# -------------------------------------------------------------------------------
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
        self.layout.setHorizontalGroup(
            self.layout.createSequentialGroup()
            .addGroup(self.group_labels)
            .addGroup(self.group_components)
        )
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
                GroupLayout.PREFERRED_SIZE,
            )
        )
