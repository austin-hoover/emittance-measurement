"""
To to:
    * Plot reconstruction lines.
    * Model Twiss parameters.
"""
import os
import math
import random
from pprint import pprint

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
from javax.swing import BorderFactory
from javax.swing import BoxLayout
from javax.swing import GroupLayout
from javax.swing import JButton
from javax.swing import JComboBox
from javax.swing import JFileChooser
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

# Local
import analysis
import plotting as plt
import utils
import xal_helpers


REC_NODE_ID = 'Begin_Of_RTBT1'


class AnalysisPanel(JPanel):
    
    def __init__(self, kin_energy=1e9):
        JPanel.__init__(self)
        self.setLayout(BorderLayout())
        self.kin_energy = kin_energy
        self.rec_node_id = REC_NODE_ID
        self.accelerator = XMLDataManager.loadDefaultAccelerator()
        self.sequence = self.accelerator.getComboSequence('RTBT')
        self.scenario = Scenario.newScenarioFor(self.sequence)
        self.tmat_generator = analysis.TransferMatrixGenerator(self.sequence, self.kin_energy)
        self.node_ids = [node.getId() for node in self.sequence.getNodes()]
        self.clear_data()
        self.build_panel()
        
    def clear_data(self):
        self.measurements = []
        self.moments_dict = dict()
        self.tmats_dict = dict()
        self.beam_stats = None
        
    def build_panel(self):
        print 'Kinetic energy is hard coded. Do not delete this message until this is fixed.'
        
        # Top panel
        #-------------------------------------------------------------------------------
        self.load_files_button = JButton('Load files')
        self.load_files_button.addActionListener(LoadFilesButtonListener(self))
        
        self.clear_files_button = JButton('Clear files') 
        self.clear_files_button.addActionListener(ClearFilesButtonListener(self))
        
        self.meas_index_label = JLabel('Measurement index to plot')
        self.meas_index_dropdown = JComboBox([0])
        self.meas_index_dropdown.addActionListener(MeasIndexDropdownListener(self))
        
        self.top_top_panel = JPanel()
        self.top_top_panel.setLayout(FlowLayout(FlowLayout.LEFT))
        self.top_top_panel.add(self.load_files_button)
        self.top_top_panel.add(self.clear_files_button)
        self.top_top_panel.add(self.meas_index_label)
        self.top_top_panel.add(self.meas_index_dropdown)
        
        self.profile_plots_panel = JPanel()
        self.profile_plots_panel.setLayout(BoxLayout(self.profile_plots_panel, BoxLayout.X_AXIS))
        self.profile_plot_panels = [
            plt.LinePlotPanel(n_lines=5, grid='y', title='Horizontal (x)'),
            plt.LinePlotPanel(n_lines=5, grid='y', title='Vertical (y)'),
            plt.LinePlotPanel(n_lines=5, grid='y', title='Diagonal (u)'),
        ]
        for panel in self.profile_plot_panels:
            self.profile_plots_panel.add(panel)

        self.top_panel = JPanel()
        self.top_panel.setLayout(BorderLayout())
        self.top_panel.setPreferredSize(Dimension(1100, 200))
        self.top_panel.add(self.top_top_panel, BorderLayout.NORTH)
        self.top_panel.add(self.profile_plots_panel)
        
        # Bottom panel
        #-------------------------------------------------------------------------------
        self.reconstruct_covariance_button = JButton('Reconstruct covariance matrix')  
        self.reconstruct_covariance_button.addActionListener(ReconstructCovarianceButtonListener(self))
        self.rec_point_label = JLabel('Reconstruction point')
        self.rec_point_dropdown = JComboBox(self.node_ids)
        self.rec_point_dropdown.addActionListener(RecPointDropdownListener(self))
        self.max_iter_label = JLabel('max iter')
        self.max_iter_text_field = JTextField('100', 5)
        self.llsq_solver_label = JLabel('LLSQ solver')
        self.llsq_solver_dropdown = JComboBox(['lsmr', 'exact'])
        self.tol_label = JLabel('tol')
        self.tol_text_field = JTextField('1e-12')
        self.results_table = JTable(ResultsTableModel(self))
        self.results_table.setShowGrid(True)
        
        self.bottom_left_panel = JPanel()
        self.bottom_left_panel.setLayout(BorderLayout())
        bottom_left_top_panel = JPanel()
        bottom_left_top_panel.setLayout(BoxLayout(bottom_left_top_panel, BoxLayout.Y_AXIS))
        temp_panel = JPanel()
        temp_panel.setLayout(FlowLayout(FlowLayout.LEFT))
        temp_panel.add(self.reconstruct_covariance_button)
        bottom_left_top_panel.add(temp_panel)
        bottom_left_top_panel1 = JPanel()
        bottom_left_top_panel1.setLayout(FlowLayout(FlowLayout.LEFT))
        bottom_left_top_panel1.add(self.rec_point_label)
        bottom_left_top_panel1.add(self.rec_point_dropdown)
        bottom_left_top_panel2 = JPanel()
        bottom_left_top_panel2.setLayout(FlowLayout(FlowLayout.LEFT))
        bottom_left_top_panel2.add(self.llsq_solver_label)
        bottom_left_top_panel2.add(self.llsq_solver_dropdown)
        bottom_left_top_panel2.add(self.max_iter_label)
        bottom_left_top_panel2.add(self.max_iter_text_field)
        bottom_left_top_panel2.add(self.tol_label)
        bottom_left_top_panel2.add(self.tol_text_field)
        bottom_left_top_panel.add(bottom_left_top_panel1)
        bottom_left_top_panel.add(bottom_left_top_panel2)
        self.bottom_left_panel.add(bottom_left_top_panel, BorderLayout.NORTH)
        bottom_left_bottom_panel = JPanel()
        bottom_left_bottom_panel.setLayout(BorderLayout())
        bottom_left_bottom_panel.add(self.results_table.getTableHeader(), BorderLayout.NORTH)
        bottom_left_bottom_panel.add(self.results_table)
        self.bottom_left_panel.add(bottom_left_bottom_panel)
        
        self.corner_plot_panel = plt.CornerPlotPanel()
        self.corner_plot_panel.setPreferredSize(Dimension(650, 455))
        
        self.bottom_panel = JPanel()
        self.bottom_panel.setBorder(BorderFactory.createLineBorder(Color.black))
        self.bottom_panel.setLayout(BorderLayout())
        self.bottom_panel.setPreferredSize(Dimension(1100, 500))
        self.bottom_panel.add(self.bottom_left_panel, BorderLayout.WEST)
        self.bottom_panel.add(self.corner_plot_panel, BorderLayout.EAST)
        
        # Build the main panel
        self.add(self.top_panel, BorderLayout.NORTH)
        self.add(self.bottom_panel, BorderLayout.SOUTH)
        
    def update_plots(self):
        measurements = self.measurements
        tmats_dict = self.tmats_dict
        moments_dict = self.moments_dict
        beam_stats = self.beam_stats
        
        # Clear the plots if there is no data.
        if not measurements: 
            for panel in self.profile_plot_panels:
                panel.removeAllGraphData()
            for panel in self.corner_plot_panel.plots.values():
                panel.removeAllGraphData()
                panel.removeAllCurveData()
            return
        
        # Plot profiles for selected measurement.
        meas_index = int(self.meas_index_dropdown.getSelectedItem())
        measurement = measurements[meas_index]
        xpos, ypos, ypos = [], [], []
        xraw_list, yraw_list, uraw_list = [], [], []
        for node_id in measurement.node_ids:
            profile = measurement.profiles[node_id]
            xpos = profile.hor.pos
            ypos = profile.ver.pos
            upos = profile.dia.pos
            xraw_list.append(profile.hor.raw)
            yraw_list.append(profile.ver.raw)
            uraw_list.append(profile.dia.raw)
        self.profile_plot_panels[0].set_data(xpos, xraw_list)
        self.profile_plot_panels[1].set_data(ypos, yraw_list)
        self.profile_plot_panels[2].set_data(upos, uraw_list)
        
        # Stop if we haven't reconstructed the covariance matrix yet.
        if not self.beam_stats:
            return
    
        # Plot the 2D projections of the rms ellipsoid (x^T Sigma x = 1).
        self.corner_plot_panel.rms_ellipses(self.beam_stats.Sigma)
            
        # Plot the reconstruction lines.
        xxp_panel = self.corner_plot_panel.plots['x,xp']
        yyp_panel = self.corner_plot_panel.plots['y,yp']
        xmax = xxp_panel.getCurrentMaxX()
        xpmax = xxp_panel.getCurrentMaxY()
        ymax = yyp_panel.getCurrentMaxX()
        ypmax = yyp_panel.getCurrentMaxY()
        print 'max x', xmax
        print 'max xp', xpmax
        print 'max y', ymax
        print 'max yp', ypmax
        
        def possible_points(M, sig_xx, sig_yy):
            Minv = M.inverse()
            x_max = 2.0 * math.sqrt(sig_xx)
            y_max = 2.0 * math.sqrt(sig_yy)
            x_vals, xp_vals, y_vals, yp_vals = [], [], [], []
            for slope in (-20, 20):
                vec_1 = Matrix([[x_max], [slope], [0.], [0.]])
                vec_0 = Minv.times(vec_1)
                x_vals.append(vec_0.get(0, 0))
                y_vals.append(vec_0.get(2, 0))
                xp_vals.append(vec_0.get(1, 0))
                yp_vals.append(vec_0.get(3, 0))
            return x_vals, xp_vals, y_vals, yp_vals
        
        node_ids = sorted(list(tmats_dict))
        for node_id, color in zip(node_ids, plt.COLOR_CYCLE):
            print node_id, color
            for M, (sig_xx, sig_yy, sig_xy) in zip(tmats_dict[node_id], moments_dict[node_id]):
                M = Matrix(M)
                x_vals, xp_vals, y_vals, yp_vals = possible_points(M, sig_xx, sig_yy)
                xxp_panel.plot(x_vals, xp_vals, color=color, ms=0)
                yyp_panel.plot(y_vals, yp_vals, color=color, ms=0)
                
            
# Tables
#-------------------------------------------------------------------------------      
class ResultsTableModel(AbstractTableModel):

    def __init__(self, panel):
        self.panel = panel
        self.column_names = ['Parameters', 'Measured', 'Model']
        self.parameter_names = [
            "<html>&epsilon;<SUB>1</SUB> [mm mrad]<html>",
            "<html>&epsilon;<SUB>2</SUB> [mm mrad]<html>",
            "<html>&epsilon;<SUB>x</SUB> [mm mrad]<html>",
            "<html>&epsilon;<SUB>y</SUB> [mm mrad]<html>",
            "C",
            "<html>&beta;<SUB>x</SUB> [m/rad]<html>",
            "<html>&beta;<SUB>y</SUB> [m/rad]<html>",
            "<html>&alpha;<SUB>x</SUB> [rad]<html>",
            "<html>&alpha;<SUB>y</SUB> [rad]<html>",
        ]

    def getValueAt(self, row, col):
        beam_stats = self.panel.beam_stats
        if col == 0:
            return self.parameter_names[row]
        elif col == 1:
            if beam_stats is None:
                return '-'
            data = [beam_stats.eps_1, beam_stats.eps_2,
                    beam_stats.eps_x, beam_stats.eps_y,
                    beam_stats.coupling_coeff,
                    beam_stats.beta_x, beam_stats.beta_y,
                    beam_stats.alpha_x, beam_stats.alpha_y]
            return data[row]
        elif col == 2:
            if row < 5:
                return '-'
            # Get model Twiss.
            return '-'

    def getColumnCount(self):
        return 3

    def getRowCount(self):
        return 9

    def getColumnName(self, col):
        return self.column_names[col]

            
# Listeners
#-------------------------------------------------------------------------------  
class LoadFilesButtonListener(ActionListener):
    
    def __init__(self, panel):
        self.panel = panel
        self.tmat_generator = panel.tmat_generator
    
    def actionPerformed(self, event):
        
        # Open file chooser dialog.
        file_chooser = JFileChooser(os.getcwd())
        file_chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY)
        return_value = file_chooser.showOpenDialog(self.panel);
        directory = file_chooser.getSelectedFile()
        if not directory or not directory.isDirectory():
            raise ValueError('Invalid directory.')
        files = directory.listFiles()
        
        # Parse each file.
        measurements = []
        for file in files:
            filename = file.toString()
            filename_short = filename.split('/')[-1]
            if 'WireAnalysisFmt' not in filename or analysis.is_harp_file(filename):
                continue
            measurements.append(analysis.Measurement(filename))
                    
        # Sort files by timestamp (oldest to newest).
        measurements = sorted(measurements, key=lambda measurement: measurement.timestamp)
        
        # Remove measurements without PVLoggerID.
        measurements = [measurement for measurement in measurements
                        if measurement.pvloggerid > 0 
                        and measurement.pvloggerid is not None]
        
        # Print loaded file names.
        for measurement in measurements:
            filename = measurement.filename.split('/')[-1]
            print "Loaded file '{}'  pvloggerid = {}".format(filename, measurement.pvloggerid)
        
        # Get dictionary of transfer matrices and measured moments.
        tmats_dict = analysis.get_tmats_dict(measurements, self.tmat_generator, self.panel.rec_node_id)
        moments_dict = analysis.get_moments_dict(measurements)
        print 'All files have been analyzed.'
            
        # Save data and update GUI.
        self.panel.measurements = measurements
        self.panel.moments_dict = moments_dict
        self.panel.tmats_dict = tmats_dict
        self.panel.meas_index_dropdown.removeAllItems()
        for meas_index in range(1 if not measurements else len(measurements)):
            self.panel.meas_index_dropdown.addItem(meas_index)
        self.panel.update_plots()
        
        
class ClearFilesButtonListener(ActionListener):
    
    def __init__(self, panel):
        self.panel = panel
    
    def actionPerformed(self, event):
        self.panel.clear_data()   
        self.panel.update_plots()
        print 'Cleared data.'
        
        
class MeasIndexDropdownListener(ActionListener):
    
    def __init__(self, panel):
        self.panel = panel
        self.dropdown = panel.meas_index_dropdown
        
    def actionPerformed(self, event):
        if self.dropdown.getSelectedItem() is not None:
            self.panel.update_plots()
            
            
class RecPointDropdownListener(ActionListener):
    
    def __init__(self, panel):
        self.panel = panel
        self.dropdown = panel.rec_point_dropdown
        
    def actionPerformed(self, event):
        rec_node_id = self.dropdown.getSelectedItem()
        measurements = self.panel.measurements
        tmat_generator = self.panel.tmat_generator
        self.panel.tmats_dict = analysis.get_tmats_dict(measurements, tmat_generator, rec_node_id)
        self.panel.rec_node_id = rec_node_id
              
            
class ReconstructCovarianceButtonListener(ActionListener):
    
    def __init__(self, panel):
        self.panel = panel
        
    def actionPerformed(self, event):
        measurements = self.panel.measurements
        moments_dict = self.panel.moments_dict
        tmats_dict = self.panel.tmats_dict
        
        if not measurements:
            raise ValueError('No wire-scanner files have been loaded.')

        # Form list of transfer matrices and measured moments.
        moments_list, tmats_list = [], []
        for meas_node_id in measurements[0].node_ids:
            moments_list.extend(moments_dict[meas_node_id])
            tmats_list.extend(tmats_dict[meas_node_id])
        # Reconstruct the covariance matrix.
        solver = self.panel.llsq_solver_dropdown.getSelectedItem()
        max_iter = int(self.panel.max_iter_text_field.getText())
        lsmr_tol = float(self.panel.tol_text_field.getText())
        Sigma = analysis.reconstruct(tmats_list, moments_list, verbose=2, 
                                     solver=solver, max_iter=max_iter, lsmr_tol=lsmr_tol)
        beam_stats = analysis.BeamStats(Sigma)
        beam_stats.print_all()
        self.panel.beam_stats = beam_stats
        # Update panel.
        self.panel.results_table.getModel().fireTableDataChanged()
        self.panel.update_plots()