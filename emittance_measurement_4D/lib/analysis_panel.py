import os
import random
from pprint import pprint

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


DIAG_WIRE_ANGLE = utils.radians(-45.0)
REC_NODE_ID = 'Begin_Of_RTBT1'


class AnalysisPanel(JPanel):
    
    def __init__(self, kin_energy=1e9):
        JPanel.__init__(self)
        self.setLayout(BorderLayout())
        self.kin_energy = kin_energy
        self.clear_data()
        self.build_panel()
        
    def clear_data(self):
        self.measurements = []
        self.moments_dict = dict()
        self.tmats_dict = dict()
        self.beam_stats = None
        
    def build_panel(self):
        print 'Kinetic energy is hard coded. Do not delete this message until this is fixed.'
        accelerator = XMLDataManager.loadDefaultAccelerator()
        self.sequence = accelerator.getComboSequence('RTBT')
        self.scenario = Scenario.newScenarioFor(self.sequence)
        self.tmat_generator = analysis.TransferMatrixGenerator(self.sequence, self.kin_energy)
        self.node_ids = [node.getId() for node in self.sequence.getNodes()]
        
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
        self.top_panel.setPreferredSize(Dimension(1200, 250))
        self.top_panel.add(self.top_top_panel, BorderLayout.NORTH)
        self.top_panel.add(self.profile_plots_panel)
        
        # Bottom panel
        #-------------------------------------------------------------------------------
        self.rec_point_dropdown = JComboBox(self.node_ids)
        self.reconstruct_covariance_button = JButton('Reconstruct covariance matrix')  
        self.reconstruct_covariance_button.addActionListener(ReconstructCovarianceButtonListener(self))
        self.results_table = JTable(ResultsTableModel(self))
        self.results_table.setShowGrid(True)
        
        self.bottom_left_panel = JPanel()
        self.bottom_left_panel.setLayout(BoxLayout(self.bottom_left_panel, BoxLayout.Y_AXIS))
        self.bottom_left_panel.add(self.reconstruct_covariance_button)
        self.bottom_left_panel.add(self.results_table.getTableHeader())
        self.bottom_left_panel.add(self.results_table)
        self.bottom_right_panel = plt.CornerPlotPanel(figsize=(700, 440))
        self.bottom_panel = JPanel()
        self.bottom_panel.setBorder(BorderFactory.createLineBorder(Color.black))
        self.bottom_panel.setLayout(BorderLayout())
        self.bottom_panel.setPreferredSize(Dimension(1200, 600))
        self.bottom_panel.add(self.bottom_left_panel, BorderLayout.WEST)
        self.bottom_panel.add(self.bottom_right_panel, BorderLayout.EAST)
        
        # Build the main panel
        self.add(self.top_panel, BorderLayout.NORTH)
        self.add(self.bottom_panel, BorderLayout.SOUTH)
        
    def update_plots(self):
        measurements = self.measurements
        if not measurements:
            for plot_panel in self.profile_plot_panels:
                plot_panel.removeAllGraphData()
            return
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
            return 1.0

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
            measurement = analysis.Measurement(filename)
            measurements.append(measurement)
            print "Loaded file '{}'  pvloggerid = {}".format(filename_short, measurement.pvloggerid)
            
        # Sort files by timestamp (oldest to newest).
        measurements = sorted(measurements, key=lambda measurement: measurement.timestamp)
        
        # Form dictionary of transfer matrices and measured moments.
        # This is convenient for later reference.
        moments_dict, tmats_dict = dict(), dict()
        for measurement in measurements:
            filename_short = measurement.filename.split('/')[-1]
            if measurement.pvloggerid < 0 or measurement.pvloggerid is None:
                print "Skipping '{}' because it doesn't have a PVLoggerID.".format(filename_short)
                continue
            print "Analyzing file '{}'".format(filename_short)
            self.tmat_generator.sync(measurement.pvloggerid)
            for meas_node_id in measurement.node_ids:
                # Store measured moments.
                profile = measurement.profiles[meas_node_id]
                sig_xx = profile.hor.stats['Sigma'].rms**2
                sig_yy = profile.ver.stats['Sigma'].rms**2
                sig_uu = profile.dia.stats['Sigma'].rms**2
                sig_xy = analysis.get_sig_xy(sig_xx, sig_yy, sig_uu, DIAG_WIRE_ANGLE)
                moments = [sig_xx, sig_yy, sig_xy]
                if meas_node_id not in moments_dict:
                    moments_dict[meas_node_id] = []
                moments_dict[meas_node_id].append(moments)
                # Store transfer matrix from reconstruction node to wire-scanner.
                tmat = self.tmat_generator.transfer_matrix(REC_NODE_ID, meas_node_id)
                if meas_node_id not in tmats_dict:
                    tmats_dict[meas_node_id] = []
                tmats_dict[meas_node_id].append(tmat)
        print 'All files have been analyzed.'
            
        # Save data and update GUI.
        self.panel.measurements = measurements
        self.panel.moments_dict = moments_dict
        self.panel.tmats_dict = tmats_dict
        self.panel.meas_index_dropdown.removeAllItems()
        for meas_index in range(len(measurements)):
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
        # Reconstruct and print results.
        Sigma = analysis.reconstruct(tmats_list, moments_list, verbose=2, solver='lsmr')
        beam_stats = analysis.BeamStats(Sigma)
        beam_stats.print_all()
        self.panel.beam_stats = beam_stats
        # Update results table.
        self.panel.results_table.getModel().fireTableDataChanged()