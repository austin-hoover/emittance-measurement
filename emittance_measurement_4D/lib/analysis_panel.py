"""Analyze wire-scanner files.

To do: clean up.
"""
from __future__ import print_function
import os
import math
from collections import Counter

from Jama import Matrix

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
from java.awt.geom import Ellipse2D
from javax.swing import BorderFactory
from javax.swing import BoxLayout
from javax.swing import DefaultCellEditor
from javax.swing.table import DefaultTableModel
from javax.swing import GroupLayout
from javax.swing import JButton
from javax.swing import JCheckBox
from javax.swing import JComboBox
from javax.swing import JFileChooser
from javax.swing import JFrame
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
from java.text import DecimalFormat
from java.text import NumberFormat

from xal.extension.widgets.plot import BasicGraphData
from xal.extension.widgets.plot import FunctionGraphsJPanel
from xal.model.probe import Probe
from xal.model.probe.traj import Trajectory
from xal.sim.scenario import AlgorithmFactory
from xal.sim.scenario import ProbeFactory
from xal.sim.scenario import Scenario
from xal.service.pvlogger.sim import PVLoggerDataSource
from xal.smf import Accelerator
from xal.smf import AcceleratorSeq
from xal.smf.data import XMLDataManager
from xal.tools.beam import Twiss
from xal.tools.beam.calc import CalculationsOnBeams
from xal.tools.beam.calc import CalculationsOnRings

# Local
import analysis
from optics import TransferMatrixGenerator
import optics
import plotting as plt
import utils
import xal_helpers


class AnalysisPanel(JPanel):
    def __init__(self):
        JPanel.__init__(self)
        self.setLayout(BorderLayout())
        self.reconstruction_node_id = "RTBT_Diag:BPM17"
        self.accelerator = XMLDataManager.loadDefaultAccelerator()
        self.sequence = self.accelerator.getComboSequence("RTBT")
        self.kinetic_energy = 1e9  # [eV]
        self.tmat_generator = TransferMatrixGenerator(
            self.sequence, self.kinetic_energy
        )
        self.node_ids = [node.getId() for node in self.sequence.getNodes()]
        self.model_twiss = dict()
        self.design_twiss = dict()
        self.clear_data()
        self.build_panel()

    def clear_data(self):
        self.groups = 0
        self.grouped_meas_indices = None
        self.measurements = []
        self.moments_dict = dict()
        self.tmats_dict = dict()
        self.beam_stats = None
        self.model_twiss = dict()

    def build_panel(self):
        # Top panel
        # -------------------------------------------------------------------------------
        self.load_files_button = JButton("Load files")
        self.load_files_button.addActionListener(LoadFilesButtonListener(self))
        self.clear_files_button = JButton("Clear files")
        self.clear_files_button.addActionListener(ClearFilesButtonListener(self))
        self.export_data_button = JButton("Export data")
        self.export_data_button.addActionListener(
            ExportDataButtonListener(self, "_output")
        )
        self.meas_index_label = JLabel("Measurement to plot")
        self.meas_index_dropdown = JComboBox([0])
        self.meas_index_dropdown.addActionListener(MeasIndexDropdownListener(self))
        self.kinetic_energy_label = JLabel("Energy [GeV]")
        self.kinetic_energy_text_field = JTextField("1.000")
        self.kinetic_energy_text_field.addActionListener(
            KinEnergyTextFieldListener(self)
        )

        self.top_top_panel = JPanel()
        self.top_top_panel.setLayout(FlowLayout(FlowLayout.LEFT))
        self.top_top_panel.add(self.load_files_button)
        self.top_top_panel.add(self.clear_files_button)
        self.top_top_panel.add(self.export_data_button)
        self.top_top_panel.add(self.kinetic_energy_label)
        self.top_top_panel.add(self.kinetic_energy_text_field)
        self.top_top_panel.add(self.meas_index_label)
        self.top_top_panel.add(self.meas_index_dropdown)

        self.profile_plots_panel = JPanel()
        self.profile_plots_panel.setLayout(
            BoxLayout(self.profile_plots_panel, BoxLayout.X_AXIS)
        )
        cycle = plt.CYCLE_538
        self.profile_plot_panels = [
            plt.LinePlotPanel(n_lines=5, grid="y", xlabel="x [mm]", cycle=cycle),
            plt.LinePlotPanel(n_lines=5, grid="y", xlabel="y [mm]", cycle=cycle),
            plt.LinePlotPanel(n_lines=5, grid="y", xlabel="u [mm]", cycle=cycle),
        ]
        for panel in self.profile_plot_panels:
            panel.setNumberFormatX(DecimalFormat("#.#"))
            panel.setNumberFormatY(DecimalFormat("#.##"))
            self.profile_plots_panel.add(panel)

        top_panel = JPanel()
        top_panel.setLayout(BorderLayout())
        top_panel.setPreferredSize(Dimension(1100, 225))
        top_panel.add(self.top_top_panel, BorderLayout.NORTH)
        top_panel.add(self.profile_plots_panel)

        # Bottom panel
        # -------------------------------------------------------------------------------
        self.group_meas_table = JTable(DefaultTableModel([['Measurement 0', 'Group 0']], ['Measurement', 'Group']))
        self.group_meas_table.getColumnModel().getColumn(1).setPreferredWidth(25)
        self.group_label = JLabel('Group')
        self.group_dropdown = JComboBox(['0'])
        self.group_dropdown.addActionListener(GroupDropdownListener(self))
        self.reconstruct_covariance_button = JButton("Reconstruct covariance matrix")
        self.reconstruct_covariance_button.addActionListener(
            ReconstructCovarianceButtonListener(self)
        )
        self.reconstruction_point_label = JLabel("Reconstruction point")
        self.reconstruction_point_dropdown = JComboBox(self.node_ids)
        self.reconstruction_point_dropdown.setSelectedItem(self.reconstruction_node_id)
        self.reconstruction_point_dropdown.addActionListener(
            ReconstructionPointDropdownListener(self)
        )
        self.frac_noise_text_field = JTextField('0.030')
        self.n_trials_text_field = JTextField('2000')
        self.results_table = JTable(ResultsTableModel(self))
        self.results_table.setShowGrid(True)
        self.norm_label = JLabel("Normalization")
        self.norm_dropdown = JComboBox(["None", "2D", "4D"])
        self.norm_dropdown.addActionListener(NormDropdownListener(self))
        self.keep_physical_checkbox = JCheckBox("Keep answer physical", False)
        self.persevere_checkbox = JCheckBox('Persevere', True)

        bottom_left_panel = JPanel()
        bottom_left_panel.setLayout(BoxLayout(bottom_left_panel, BoxLayout.Y_AXIS))

        bottom_left_panel.add(JScrollPane(self.group_meas_table))

        bottom_left_top_panel = JPanel()
        bottom_left_top_panel.setLayout(BoxLayout(bottom_left_top_panel, BoxLayout.Y_AXIS))
        row = JPanel()
        row.setLayout(FlowLayout(FlowLayout.LEFT))
        row.add(self.reconstruct_covariance_button)
        bottom_left_top_panel.add(row)
        row = JPanel()
        row.setLayout(FlowLayout(FlowLayout.LEFT))
        row.add(self.reconstruction_point_label)
        row.add(self.reconstruction_point_dropdown)
        row.add(self.keep_physical_checkbox)
        bottom_left_top_panel.add(row)
        row = JPanel()
        row.setLayout(FlowLayout(FlowLayout.LEFT))
        row.add(JLabel('Frac. noise'))
        row.add(self.frac_noise_text_field)
        row.add(JLabel('Trials'))
        row.add(self.n_trials_text_field)
        row.add(self.persevere_checkbox)
        bottom_left_top_panel.add(row)

        row = JPanel()
        row.setLayout(FlowLayout(FlowLayout.LEFT))
        row.add(self.group_label)
        row.add(self.group_dropdown)
        bottom_left_top_panel.add(row)
        bottom_left_panel.add(bottom_left_top_panel)

        bottom_left_bottom_panel = JPanel()
        bottom_left_bottom_panel.setLayout(BorderLayout())
        bottom_left_bottom_panel.add(
            self.results_table.getTableHeader(), BorderLayout.NORTH
        )
        bottom_left_bottom_panel.add(self.results_table)
        bottom_left_panel.add(bottom_left_bottom_panel)

        self.bottom_right_panel_A = JPanel()
        self.bottom_right_panel_A.setLayout(BorderLayout())
        self.bottom_right_panel_A_top = JPanel()
        self.bottom_right_panel_A_top.add(self.norm_label)
        self.bottom_right_panel_A_top.add(self.norm_dropdown)
        self.corner_plot_panel = plt.CornerPlotPanel()
        for panel in self.corner_plot_panel.plots.values():
            panel.xMarkersOn(False)
            panel.yMarkersOn(False)
        self.bottom_right_panel_A.add(self.bottom_right_panel_A_top, BorderLayout.NORTH)
        self.bottom_right_panel_A.add(self.corner_plot_panel, BorderLayout.CENTER)

        self.bottom_right_panel_B = JPanel()
        self.bottom_right_panel_B.setLayout(
            BoxLayout(self.bottom_right_panel_B, BoxLayout.Y_AXIS)
        )
        self.emittance_plot_panels = [
            plt.LinePlotPanel(
                n_lines=4, grid="y", ylabel="[mm mrad]", title="Emittance"
            ),
            plt.LinePlotPanel(
                n_lines=4,
                grid="y",
                xlabel="Group",
                ylabel="[mm^2 mrad^2]",
                title="4D emittance",
            ),
        ]
        for panel in self.emittance_plot_panels:
            panel.setNumberFormatX(DecimalFormat("#."))
            panel.setNumberFormatY(DecimalFormat("#.##"))
            self.bottom_right_panel_B.add(panel)

        self.bottom_right_pane = JTabbedPane()
        self.bottom_right_pane.addTab("Phase space projections", self.bottom_right_panel_A)
        self.bottom_right_pane.addTab("Emittances", self.bottom_right_panel_B)

        # Build the bottom panel.
        bottom_panel = JPanel()
        bottom_panel.setBorder(BorderFactory.createEtchedBorder())
        bottom_panel.setLayout(GridBagLayout())
        c = GridBagConstraints()

        c.gridx = 0
        c.gridy = 0
        c.weightx = 0.25
        c.weighty = 0.5
        c.fill = GridBagConstraints.BOTH
        c.anchor = GridBagConstraints.WEST
        bottom_panel.add(bottom_left_panel, c)

        c.gridx = 1
        c.gridy = 0
        c.gridwidth = 2
        c.weightx = 1.0
        c.gridwidth = GridBagConstraints.REMAINDER
        c.fill = GridBagConstraints.BOTH
        bottom_panel.add(self.bottom_right_pane, c)

        # Build the main panel
        self.setLayout(GridBagLayout())
        c = GridBagConstraints()
        c.gridy = 0
        c.weightx = 1.0
        c.weighty = 0.4
        c.fill = GridBagConstraints.BOTH
        self.add(top_panel, c)
        c.gridy = 1
        c.weightx = 1.0
        c.weighty = 1.0
        c.gridheight = 4
        c.fill = GridBagConstraints.BOTH
        self.add(bottom_panel, c)

    def reset_groups(self):
        data = [[self.measurements[i].filename_short, 'Group {}'.format(i)]
                for i in range(len(self.measurements))]
        col_names = ['Measurement', 'Group']
        n_meas = len(self.measurements)
        self.groups = n_meas
        group_options = JComboBox(['Group {}'.format(i) for i in range(self.groups)])
        group_options.addActionListener(GroupOptionsListener(self))
        self.group_meas_table.setModel(DefaultTableModel(data, col_names))
        self.group_meas_table.getColumnModel().getColumn(1).setCellEditor(
            DefaultCellEditor(group_options)
        )
        self.group_meas_table.setFillsViewportHeight(True)
        self.group_meas_table.getModel().fireTableDataChanged()
        self.group_dropdown.removeAllItems()
        for i in range(n_meas):
            self.group_dropdown.addItem(str(i))

    def update_plots(self):
        measurements = self.measurements
        tmats_dict = self.tmats_dict
        moments_dict = self.moments_dict

        # Clear the plots if there is no data.
        if not measurements:
            for panel in self.profile_plot_panels:
                panel.removeAllGraphData()
            self.corner_plot_panel.clear()
            for panel in self.emittance_plot_panels:
                panel.removeAllGraphData()
            return

        for panel in self.emittance_plot_panels:
            panel.removeAllGraphData()

        # Plot profiles for selected measurement.
        meas_index = int(self.meas_index_dropdown.getSelectedItem())
        measurement = measurements[meas_index]
        xpos, ypos, upos = [], [], []
        xraw_list, yraw_list, uraw_list = [], [], []
        for node_id in measurement.node_ids:
            profile = measurement[node_id]
            xpos = profile.hor.pos
            ypos = profile.ver.pos
            upos = profile.dia.pos
            xraw_list.append(profile.hor.raw)
            yraw_list.append(profile.ver.raw)
            uraw_list.append(profile.dia.raw)
        self.profile_plot_panels[0].set_data(xpos, xraw_list)
        self.profile_plot_panels[1].set_data(ypos, yraw_list)
        self.profile_plot_panels[2].set_data(upos, uraw_list)

        # Set axis limits.
        n_ticks = 5.0
        self.profile_plot_panels[0].set_xlim(
            min(xpos), max(xpos), (max(xpos) - min(xpos)) / n_ticks
        )
        self.profile_plot_panels[1].set_xlim(
            min(ypos), max(ypos), (max(ypos) - min(ypos)) / n_ticks
        )
        self.profile_plot_panels[2].set_xlim(
            min(upos), max(upos), (max(upos) - min(upos)) / n_ticks
        )
        pad = 0.05
        hmax_x = (1.0 + pad) * max([max(raw) for raw in xraw_list])
        hmax_y = (1.0 + pad) * max([max(raw) for raw in yraw_list])
        hmax_u = (1.0 + pad) * max([max(raw) for raw in uraw_list])
        hmin = min(
            [
                min(raw)
                for raw_list in [xraw_list, yraw_list, uraw_list]
                for raw in raw_list
            ]
        )
        hmin -= pad * abs(max(hmax_x, hmax_y, hmax_u))
        n_ticks = 5.0
        self.profile_plot_panels[0].set_ylim(hmin, hmax_x, (hmax_x - hmin) / n_ticks)
        self.profile_plot_panels[1].set_ylim(hmin, hmax_y, (hmax_y - hmin) / n_ticks)
        self.profile_plot_panels[2].set_ylim(hmin, hmax_u, (hmax_u - hmin) / n_ticks)

        # Stop if we haven't reconstructed the covariance matrix yet.
        if not self.beam_stats:
            return

        # Plot the 2D projections of the rms ellipsoid (x^T Sigma x = 1).
        group = int(self.group_dropdown.getSelectedItem())
        beam_stats = self.beam_stats[group]
        Sigma = beam_stats.Sigma
        V = utils.identity_matrix(4)
        norm = self.norm_dropdown.getSelectedItem()
        if norm == "2D":
            alpha_x = beam_stats.alpha_x
            alpha_y = beam_stats.alpha_y
            beta_x = beam_stats.beta_x
            beta_y = beam_stats.beta_y
            V = analysis.V_matrix_uncoupled(alpha_x, alpha_y, beta_x, beta_y)
        elif norm == "4D":
            U = Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
            SigmaU = Sigma.times(U)
            eig = SigmaU.eig()
            V = eig.getV()
        Vinv = V.inverse()
        Sigma = Vinv.times(Sigma.times(Vinv.transpose()))
        self.corner_plot_panel.clear()
        self.corner_plot_panel.rms_ellipses(Sigma)

        # Plot reconstruction lines.
        def possible_points(M, sig_xx, sig_yy):
            Minv = M.inverse()
            x_max = math.sqrt(sig_xx)
            y_max = math.sqrt(sig_yy)
            x_vals, xp_vals, y_vals, yp_vals = [], [], [], []
            for slope in [-100.0, 100.0]:
                vec_1 = utils.list_to_col_mat([x_max, slope, 0, 0])
                vec_0 = Minv.times(vec_1)
                vec_0 = Vinv.times(vec_0)
                x_vals.append(vec_0.get(0, 0))
                xp_vals.append(vec_0.get(1, 0))
                vec_1 = utils.list_to_col_mat([0, 0, y_max, slope])
                vec_0 = Minv.times(vec_1)
                vec_0 = Vinv.times(vec_0)
                y_vals.append(vec_0.get(2, 0))
                yp_vals.append(vec_0.get(3, 0))
            return x_vals, xp_vals, y_vals, yp_vals

        xxp_panel = self.corner_plot_panel.plots["x-xp"]
        yyp_panel = self.corner_plot_panel.plots["y-yp"]
        node_ids = sorted(list(tmats_dict))
        for node_id, color in zip(node_ids, plt.CYCLE_538):
            for meas_index, (M, (sig_xx, sig_yy, sig_uu, sig_xy)) in enumerate(
                    zip(tmats_dict[node_id], moments_dict[node_id])
            ):
                if meas_index in self.grouped_meas_indices[group]:
                    M = Matrix(M)
                    x_vals, xp_vals, y_vals, yp_vals = possible_points(M, sig_xx, sig_yy)
                    xxp_panel.plot(x_vals, xp_vals, color=color, ms=0, lw=2)
                    yyp_panel.plot(y_vals, yp_vals, color=color, ms=0, lw=2)

        # Plot emittance vs. group number.
        groups = []
        eps_x_list = []
        eps_y_list = []
        eps_1_list = []
        eps_2_list = []
        eps_x_eps_y_list = []
        eps_1_eps_2_list = []
        ran_eps_x_mean_list = []
        ran_eps_y_mean_list = []
        ran_eps_1_mean_list = []
        ran_eps_2_mean_list = []
        ran_eps_x_eps_y_mean_list = []
        ran_eps_1_eps_2_mean_list = []
        ran_eps_x_std_list = []
        ran_eps_y_std_list = []
        ran_eps_1_std_list = []
        ran_eps_2_std_list = []
        ran_eps_x_eps_y_std_list = []
        ran_eps_1_eps_2_std_list = []
        for group, beam_stats in enumerate(self.beam_stats):
            groups.append(group)
            eps_x_list.append(beam_stats.eps_x)
            eps_y_list.append(beam_stats.eps_y)
            eps_1_list.append(beam_stats.eps_1)
            eps_2_list.append(beam_stats.eps_2)
            eps_x_eps_y_list.append(beam_stats.eps_x * beam_stats.eps_y)
            if beam_stats.eps_1 and beam_stats.eps_2:
                eps_1_eps_2_list.append(beam_stats.eps_1 * beam_stats.eps_2)
            else:
                eps_1_eps_2_list.append(None)

            ran_eps_x_mean_list.append(beam_stats.ran_eps_x_mean)
            ran_eps_y_mean_list.append(beam_stats.ran_eps_y_mean)
            ran_eps_1_mean_list.append(beam_stats.ran_eps_1_mean)
            ran_eps_2_mean_list.append(beam_stats.ran_eps_2_mean)
            ran_eps_x_eps_y_mean_list.append(beam_stats.ran_eps_x_eps_y_mean)
            ran_eps_1_eps_2_mean_list.append(beam_stats.ran_eps_1_eps_2_mean)
            ran_eps_x_std_list.append(beam_stats.ran_eps_x_std)
            ran_eps_y_std_list.append(beam_stats.ran_eps_y_std)
            ran_eps_1_std_list.append(beam_stats.ran_eps_1_std)
            ran_eps_2_std_list.append(beam_stats.ran_eps_2_std)
            ran_eps_x_eps_y_std_list.append(beam_stats.ran_eps_x_eps_y_std)
            ran_eps_1_eps_2_std_list.append(beam_stats.ran_eps_1_eps_2_std)

        plt_kws = dict(lw=None, ms=None)
        n_ticks = 5.0

        data_lists = [eps_x_list, eps_y_list, eps_1_list, eps_2_list]
        mean_lists = [
            ran_eps_x_mean_list,
            ran_eps_y_mean_list,
            ran_eps_1_mean_list,
            ran_eps_2_mean_list,
        ]
        std_lists = [
            ran_eps_x_std_list,
            ran_eps_y_std_list,
            ran_eps_1_std_list,
            ran_eps_2_std_list,
        ]
        for i, (data_list, mean_list, std_list) in enumerate(
            zip(data_lists, mean_lists, std_lists)
        ):
            self.emittance_plot_panels[0].plot(
                groups, data_list, color=plt.CYCLE_COLORBLIND[i], **plt_kws
            )
            self.emittance_plot_panels[0].plot(
                groups,
                mean_list,
                yerrs=std_list,
                color=plt.CYCLE_COLORBLIND[i],
                lw=0,
                ebar_only=True,
            )
        # Set y axis limits.
        hmax = 1.1 * max([max(data_list) for data_list in data_lists])
        self.emittance_plot_panels[0].set_ylim(0.0, hmax, hmax / n_ticks)

        data_lists = [eps_x_eps_y_list, eps_1_eps_2_list]
        mean_lists = [ran_eps_x_eps_y_mean_list, ran_eps_1_eps_2_mean_list]
        std_lists = [ran_eps_x_eps_y_std_list, ran_eps_1_eps_2_std_list]
        colors = [Color.RED, Color.BLUE]
        for i, (data_list, mean_list, std_list) in enumerate(
            zip(data_lists, mean_lists, std_lists)
        ):
            self.emittance_plot_panels[1].plot(
                groups, data_list, color=colors[i], **plt_kws
            )
            self.emittance_plot_panels[1].plot(
                groups,
                mean_list,
                yerrs=std_list,
                color=colors[i],
                lw=0,
                ebar_only=True,
            )
        # Set y axis limits.
        hmax = 1.1 * max([max(data_list) for data_list in data_lists])
        self.emittance_plot_panels[1].set_ylim(0.0, hmax, hmax / n_ticks)

        # Pad the x axis limits.
        for plot_panel in self.emittance_plot_panels:
            plot_panel.set_xlim(-1, groups[-1] + 1, 1.0)

        graph_data_list = self.emittance_plot_panels[0].getAllGraphData()
        for i, label in enumerate(["eps_x", "eps_y", "eps_1", "eps_2"]):
            gd = graph_data_list[2 * i]
            gd.setGraphProperty("Legend", label)
        graph_data_list = self.emittance_plot_panels[1].getAllGraphData()
        for i, label in enumerate(["eps_x*eps_y", "eps_1*eps_2"]):
            gd = graph_data_list[2 * i]
            gd.setGraphProperty("Legend", label)
        for plot_panel in self.emittance_plot_panels:
            plot_panel.setLegendVisible(True)
            plot_panel.setLegendButtonVisible(True)

    def compute_model_twiss(self):
        """Compute the model Twiss parameters at the reconstruction point.
        
        We use the pvloggerid of the first measurement, assuming the optics
        don't change between measurements. The method assumes this. If the
        user selects a reconstruction point downstream of QH18 in the RTBT, 
        and if the optics were varied during the scan, then the method 
        doesn't work.
        """
        if not self.measurements:
            return

        alpha_x, alpha_y, beta_x, beta_y = optics.compute_model_twiss(
            self.reconstruction_node_id,
            self.kinetic_energy,
            pvloggerid=self.measurements[0].pvloggerid,
        )
        self.model_twiss["alpha_x"] = alpha_x
        self.model_twiss["alpha_y"] = alpha_y
        self.model_twiss["beta_x"] = beta_x
        self.model_twiss["beta_y"] = beta_y

    def ws_phases(self):
        """Compute model phase advance to each wire-scanner for each measurement.
        
        Returns dict. Each key is a wire-scanner id. Each value is an (N, 2) list --
        where N is the number of loaded measurements -- of the horizontal and 
        vertical phase advances from the start of the RTBT to the wire-scanner. The
        units are radians mod 2pi. 
        """
        if not self.measurements:
            return

        phases_dict = dict()

        for measurement in self.measurements:
            pvl_data_source = PVLoggerDataSource(measurement.pvloggerid)

            # Get the model optics at the RTBT entrance in the Ring.
            sequence = self.accelerator.getComboSequence("Ring")
            scenario = Scenario.newScenarioFor(sequence)
            scenario = pvl_data_source.setModelSource(sequence, scenario)
            scenario.resync()
            tracker = AlgorithmFactory.createTransferMapTracker(sequence)
            probe = ProbeFactory.getTransferMapProbe(sequence, tracker)
            probe.setKineticEnergy(self.kinetic_energy)
            scenario.setProbe(probe)
            scenario.run()
            trajectory = probe.getTrajectory()
            calculator = CalculationsOnRings(trajectory)
            state = trajectory.stateForElement("Begin_Of_Ring3")
            twiss_x, twiss_y, twiss_z = calculator.computeMatchedTwissAt(state)

            # Track envelope probe through RTBT.
            sequence = self.accelerator.getComboSequence("RTBT")
            scenario = Scenario.newScenarioFor(sequence)
            scenario = pvl_data_source.setModelSource(sequence, scenario)
            scenario.resync()
            tracker = AlgorithmFactory.createEnvelopeTracker(sequence)
            tracker.setUseSpacecharge(False)
            probe = ProbeFactory.getEnvelopeProbe(sequence, tracker)
            probe.setBeamCurrent(0.0)
            probe.setKineticEnergy(self.kinetic_energy)
            eps_x = eps_y = 20e-5  # [mm mrad] (arbitrary)
            twiss_x = Twiss(twiss_x.getAlpha(), twiss_x.getBeta(), eps_x)
            twiss_y = Twiss(twiss_y.getAlpha(), twiss_y.getBeta(), eps_y)
            twiss_z = Twiss(0, 1, 0)
            probe.initFromTwiss([twiss_x, twiss_y, twiss_z])
            scenario.setProbe(probe)
            scenario.run()
            trajectory = probe.getTrajectory()
            calculator = CalculationsOnBeams(trajectory)

            for ws_id in measurement.node_ids:
                state = trajectory.stateForElement(ws_id)
                mu_x, mu_y, _ = calculator.computeBetatronPhase(state).toArray()
                if ws_id not in phases_dict:
                    phases_dict[ws_id] = []
                phases_dict[ws_id].append([mu_x, mu_y])
        return phases_dict


# Tables
# -------------------------------------------------------------------------------
class GroupOptionsListener(ActionListener):
    def __init__(self, panel):
        self.panel = panel
        self.table = panel.group_meas_table

    def actionPerformed(self, event):
        self.panel.groups = len(self.panel.measurements)
        if self.panel.groups == 0:
            return
        counter = Counter(
            [self.table.getModel().getValueAt(row, 1) for row in range(self.table.getRowCount())]
        )
        self.panel.groups = len(counter.keys())
        self.panel.group_dropdown.removeAllItems()
        for group in range(self.panel.groups):
            self.panel.group_dropdown.addItem(str(group))
        print('There are now {} groups'.format(self.panel.groups))


class ResultsTableModel(AbstractTableModel):
    def __init__(self, panel):
        self.panel = panel
        self.column_names = ["Parameter", "Model", "Measured", "Mean", "Std."]
        self.parameter_names = [
            "<html>&epsilon;<SUB>1</SUB> [mm mrad]<html>",
            "<html>&epsilon;<SUB>2</SUB> [mm mrad]<html>",
            "<html>&epsilon;<SUB>x</SUB> [mm mrad]<html>",
            "<html>&epsilon;<SUB>y</SUB> [mm mrad]<html>",
            "C",
            "<html>&beta;<SUB>x</SUB> [m/rad]<html>",
            "<html>&beta;<SUB>y</SUB> [m/rad]<html>",
            "<html>&alpha;<SUB>x</SUB><html>",
            "<html>&alpha;<SUB>y</SUB><html>",
        ]

    def round(self, value, decimals=4):
        if type(value) is not float:
            return value
        return round(value, 4)

    def getValueAt(self, row, col):
        no_meas_data = not self.panel.measurements
        no_calc_data = not self.panel.beam_stats
        if not no_calc_data:
            group = int(self.panel.group_dropdown.getSelectedItem())
            beam_stats = self.panel.beam_stats[group]
        if col == 0:
            return self.parameter_names[row]
        if col == 1:
            if no_meas_data or row < 5:
                return "-"
            if not self.panel.model_twiss:
                self.panel.compute_model_twiss()
            if row == 5:
                return self.round(self.panel.model_twiss["beta_x"])
            if row == 6:
                return self.round(self.panel.model_twiss["beta_y"])
            if row == 7:
                return self.round(self.panel.model_twiss["alpha_x"])
            if row == 8:
                return self.round(self.panel.model_twiss["alpha_y"])
        if no_calc_data:
            return "-"
        if col == 2:
            data = [
                beam_stats.eps_1,
                beam_stats.eps_2,
                beam_stats.eps_x,
                beam_stats.eps_y,
                beam_stats.coupling_coeff,
                beam_stats.beta_x,
                beam_stats.beta_y,
                beam_stats.alpha_x,
                beam_stats.alpha_y,
            ]
        elif col == 3:
            data = [
                beam_stats.ran_eps_1_mean,
                beam_stats.ran_eps_2_mean,
                beam_stats.ran_eps_x_mean,
                beam_stats.ran_eps_y_mean,
                "",
                beam_stats.ran_beta_x_mean,
                beam_stats.ran_beta_y_mean,
                beam_stats.ran_alpha_x_mean,
                beam_stats.ran_alpha_y_mean,
            ]
        elif col == 4:
            data = [
                beam_stats.ran_eps_1_std,
                beam_stats.ran_eps_2_std,
                beam_stats.ran_eps_x_std,
                beam_stats.ran_eps_y_std,
                "",
                beam_stats.ran_beta_x_std,
                beam_stats.ran_beta_y_std,
                beam_stats.ran_alpha_x_std,
                beam_stats.ran_alpha_y_std,
            ]
        return self.round(data[row])

    def getColumnCount(self):
        return len(self.column_names)

    def getRowCount(self):
        return 9

    def getColumnName(self, col):
        return self.column_names[col]


# Listeners
# -------------------------------------------------------------------------------
class LoadFilesButtonListener(ActionListener):
    def __init__(self, panel):
        self.panel = panel
        self.tmat_generator = panel.tmat_generator

    def actionPerformed(self, event):
        # Open file chooser dialog.
        file_chooser = JFileChooser(os.getcwd())
        file_chooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES)
        file_chooser.setMultiSelectionEnabled(True)
        return_value = file_chooser.showOpenDialog(self.panel)
        selected_items = file_chooser.getSelectedFiles()
        # Only keep files, not directories.
        files = []
        for item in selected_items:
            if item.isDirectory():
                files.extend(item.listFiles())
            else:
                files.append(item)
        filenames = [filename.toString() for filename in files]
        # Read the files and add to the list of measurements.
        measurements = analysis.process(filenames)
        self.panel.measurements.extend(measurements)
        # Make dictionaries of measured moments and transfer matrices at each wire-scanner.
        self.panel.moments_dict, self.panel.tmats_dict = analysis.get_scan_info(
            self.panel.measurements,
            self.tmat_generator,
            self.panel.reconstruction_node_id
        )
        # Update GUI
        self.panel.meas_index_dropdown.removeAllItems()
        for meas_index in range(1 if not self.panel.measurements else len(self.panel.measurements)):
            self.panel.meas_index_dropdown.addItem(meas_index)
        self.panel.update_plots()
        self.panel.results_table.getModel().fireTableDataChanged()
        self.panel.reset_groups()


class ClearFilesButtonListener(ActionListener):
    def __init__(self, panel):
        self.panel = panel

    def actionPerformed(self, event):
        self.panel.clear_data()
        self.panel.update_plots()
        self.panel.results_table.getModel().fireTableDataChanged()
        self.panel.reset_groups()
        print("Cleared data.")


class ExportDataButtonListener(ActionListener):
    def __init__(self, panel, folder):
        self.panel = panel
        self.folder = folder

    def actionPerformed(self, event):
        print("Exporting data...")
        utils.delete_files_not_folders(self.folder)
        measurements = self.panel.measurements
        tmats_dict = self.panel.tmats_dict
        moments_dict = self.panel.moments_dict
        ws_ids = self.panel.measurements[0].node_ids

        # Transfer matrices
        file = open(os.path.join(self.folder, "transfer_mats.dat"), "w")
        for ws_id in ws_ids:
            for tmat in tmats_dict[ws_id]:
                tmat_elems = [elem for row in tmat for elem in row]
                fstr = 17 * "{} " + "\n"
                file.write(fstr.format(ws_id, *tmat_elems))
        file.close()

        # Measured moments
        file = open(os.path.join(self.folder, "moments.dat"), "w")
        for ws_id in ws_ids:
            for moments in moments_dict[ws_id]:
                fstr = 5 * "{} " + "\n"
                file.write(fstr.format(ws_id, *moments))
        file.close()

        # Phase advances
        phases_dict = self.panel.ws_phases()
        file = open(os.path.join(self.folder, "phase_adv.dat"), "w")
        for ws_id in ws_ids:
            phases = phases_dict[ws_id]
            for (mux, muy) in phases:
                file.write("{} {} {}\n".format(ws_id, mux, muy))
        file.close()

        # Profile data
        file1 = open(os.path.join(self.folder, "pos_x.dat"), "w")
        file2 = open(os.path.join(self.folder, "raw_x.dat"), "w")
        for ws_id in ws_ids:
            for measurement in measurements:
                profile = measurement[ws_id]
                for file in [file1, file2]:
                    file.write(ws_id + " ")
                for pos, raw in zip(profile.hor.pos, profile.hor.raw):
                    file1.write("{} ".format(pos))
                    file2.write("{} ".format(raw))
                for file in [file1, file2]:
                    file.write("\n")
        for file in [file1, file2]:
            file.close()
        file1 = open(os.path.join(self.folder, "pos_y.dat"), "w")
        file2 = open(os.path.join(self.folder, "raw_y.dat"), "w")
        for ws_id in ws_ids:
            for measurement in measurements:
                profile = measurement[ws_id]
                for file in [file1, file2]:
                    file.write(ws_id + " ")
                for pos, raw in zip(profile.ver.pos, profile.ver.raw):
                    file1.write("{} ".format(pos))
                    file2.write("{} ".format(raw))
                for file in [file1, file2]:
                    file.write("\n")
        for file in [file1, file2]:
            file.close()
        file1 = open(os.path.join(self.folder, "pos_u.dat"), "w")
        file2 = open(os.path.join(self.folder, "raw_u.dat"), "w")
        for ws_id in ws_ids:
            for measurement in measurements:
                profile = measurement[ws_id]
                for file in [file1, file2]:
                    file.write(ws_id + " ")
                for pos, raw in zip(profile.dia.pos, profile.dia.raw):
                    file1.write("{} ".format(pos))
                    file2.write("{} ".format(raw))
                for file in [file1, file2]:
                    file.write("\n")
        for file in [file1, file2]:
            file.close()

        # Other info
        file = open(os.path.join(self.folder, "info.dat"), "w")
        file.write(
            "reconstruction_point = {}\n".format(self.panel.reconstruction_node_id)
        )
        file.write("beam_energy_GeV = {}\n".format(self.panel.kinetic_energy * 1e-9))
        file.close()

        print("Done. Files are in folder: '_output'")


class KinEnergyTextFieldListener(ActionListener):
    def __init__(self, panel):
        self.panel = panel

    def actionPerformed(self, event):
        kinetic_energy = 1e9 * float(self.panel.kinetic_energy_text_field.getText())
        self.panel.kinetic_energy = kinetic_energy
        self.panel.tmat_generator.set_kinetic_energy(kinetic_energy)
        self.panel.results_table.getModel().fireTableDataChanged()
        self.panel.reset_groups()
        print(
            "Updated reconstruction kinetic energy to {:.3e} [eV].".format(
                kinetic_energy
            )
        )


class MeasIndexDropdownListener(ActionListener):
    def __init__(self, panel):
        self.panel = panel
        self.dropdown = panel.meas_index_dropdown

    def actionPerformed(self, event):
        if self.dropdown.getSelectedItem() is not None:
            self.panel.update_plots()


class ReconstructionPointDropdownListener(ActionListener):
    def __init__(self, panel):
        self.panel = panel
        self.dropdown = panel.reconstruction_point_dropdown

    def actionPerformed(self, event):
        reconstruction_node_id = self.dropdown.getSelectedItem()
        moments_dict, tmats_dict = analysis.get_scan_info(
            self.panel.measurements, self.panel.tmat_generator, reconstruction_node_id
        )
        self.panel.reconstruction_node_id = reconstruction_node_id
        self.panel.moments_dict = moments_dict
        self.panel.tmats_dict = tmats_dict
        self.panel.corner_plot_panel.clear()
        for emittance_plot_panel in self.panel.emittance_plot_panels:
            emittance_plot_panel.removeAllGraphData()
        self.panel.compute_model_twiss()
        self.panel.results_table.getModel().fireTableDataChanged()


class GroupDropdownListener(ActionListener):
    def __init__(self, panel):
        self.panel = panel

    def actionPerformed(self, event):
        self.panel.results_table.getModel().fireTableDataChanged()
        self.panel.update_plots()


class ReconstructCovarianceButtonListener(ActionListener):
    def __init__(self, panel):
        self.panel = panel

    def actionPerformed(self, event):
        measurements = self.panel.measurements
        moments_dict = self.panel.moments_dict
        tmats_dict = self.panel.tmats_dict
        constr = self.panel.keep_physical_checkbox.isSelected()
        node_ids = list(moments_dict)

        if not measurements:
            raise ValueError("No wire-scanner files have been loaded.")

        # Group the measurements.
        grouped_meas_indices = dict()
        for meas_index, measurement in enumerate(measurements):
            string = self.panel.group_meas_table.getModel().getValueAt(meas_index, 1)
            group = int(string.split(' ')[-1])
            if group not in grouped_meas_indices:
                grouped_meas_indices[group] = []
            grouped_meas_indices[group].append(meas_index)
        self.panel.grouped_meas_indices = grouped_meas_indices

        # Reconstruct at each group.
        print('Reconstructing...')
        self.panel.beam_stats = []
        for group in range(len(grouped_meas_indices)):
            # Collect the moments and transfer matrices in this group.
            _meas_indices = grouped_meas_indices[group]
            tmats_list, moments_list_xy, moments_list_uu = [], [], []
            for node_id in node_ids:
                for meas_index in _meas_indices:
                    tmats_list.append(tmats_dict[node_id][meas_index])
                    sig_xx, sig_yy, sig_uu, sig_xy = moments_dict[node_id][meas_index]
                    moments_list_xy.append([sig_xx, sig_yy, sig_xy])
                    moments_list_uu.append([sig_xx, sig_yy, sig_uu])
            # Reconstruct using measured moments.
            Sigma = analysis.reconstruct(tmats_list, moments_list_xy, constr=constr, verbose=2)
            # Reconstruct with noise.
            Sigmas = analysis.reconstruct_random_trials(
                tmats_list,
                moments_list_uu,
                frac_err=float(self.panel.frac_noise_text_field.getText()),
                n_trials=int(self.panel.n_trials_text_field.getText()),
                persevere=bool(self.panel.persevere_checkbox.isSelected())
            )
            # Save statistics.
            stats = analysis.BeamStats(Sigma, Sigmas)
            self.panel.beam_stats.append(stats)
            # Display results.
            stats.print_all()
            print("Random trials:")
            print(
                "    means =",
                stats.ran_eps_x_mean,
                stats.ran_eps_y_mean,
                stats.ran_eps_1_mean,
                stats.ran_eps_2_mean,
            )
            print(
                "    stds =",
                stats.ran_eps_x_std,
                stats.ran_eps_y_std,
                stats.ran_eps_1_std,
                stats.ran_eps_2_std,
            )
            print()

        # Update the panel.
        self.panel.results_table.getModel().fireTableDataChanged()
        self.panel.update_plots()


class NormDropdownListener(ActionListener):
    """Normalize the plotted phase space."""
    def __init__(self, panel):
        self.panel = panel

    def actionPerformed(self, event):
        self.panel.update_plots()
