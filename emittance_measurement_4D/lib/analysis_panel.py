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
        
    def build_panel(self):
        print 'Kinetic energy is hard coded. Do not delete this message until this is fixed.'
        accelerator = XMLDataManager.loadDefaultAccelerator()
        self.sequence = accelerator.getComboSequence('RTBT')
        self.scenario = Scenario.newScenarioFor(self.sequence)
        self.tmat_generator = analysis.TransferMatrixGenerator(self.sequence, self.kin_energy)
        
        # Buttons
        self.load_files_button = JButton('Load files')
        self.clear_files_button = JButton('Clear files')
        self.reconstruct_covariance_button = JButton('Reconstruct covariance matrix')   
        
        # Action listeners
        self.load_files_button.addActionListener(LoadFilesButtonListener(self))
        self.clear_files_button.addActionListener(ClearFilesButtonListener(self))
        self.reconstruct_covariance_button.addActionListener(
            ReconstructCovarianceButtonListener(self))
        
        # Add components.
        # -----------------------------------
        
        # The top panel will have buttons to load and clear files, as well as plots of 
        # the beam profiles at each wire-scanner at a selected scan index.
        self.temp_panel = JPanel()
        self.temp_panel.add(self.load_files_button)
        self.temp_panel.add(self.clear_files_button)
        
        self.profile_plots_panel = JPanel()
        self.profile_plots_panel.setLayout(BoxLayout(self.profile_plots_panel, BoxLayout.X_AXIS))
#         self.profile_plots_panel.setPreferredSize()
        n_lines = 1
        self.profile_plot_panels = [
            plt.LinePlotPanel(xlabel='x [mm]', n_lines=n_lines, grid='y'),
            plt.LinePlotPanel(xlabel='y [mm]', n_lines=n_lines, grid='y'),
            plt.LinePlotPanel(xlabel='u [mm]', n_lines=n_lines, grid='y'),
        ]
        for panel in self.profile_plot_panels:
            self.profile_plots_panel.add(panel)
        
        self.top_panel = JPanel()
        self.top_panel.setLayout(BorderLayout())
        self.top_panel.setPreferredSize(Dimension(1200, 200))
#         self.top_panel.setLayout(BoxLayout(self.top_panel, BoxLayout.X_AXIS))
        self.top_panel.add(self.temp_panel, BorderLayout.WEST)
        self.top_panel.add(self.profile_plots_panel)


        # The bottom left panel reconstructs the covariance matrix and 
        # prints a table of results.
        self.bottom_left_panel = JPanel()
        self.bottom_left_panel.setLayout(BoxLayout(self.bottom_left_panel, BoxLayout.Y_AXIS))
        self.bottom_left_panel.add(self.reconstruct_covariance_button)
        self.bottom_left_panel.add(JTextField('[Parameter table]'))
        
        
        # The bottom right panel plots the ellipse defined by x^T Sigma x = 1.  
        self.bottom_right_panel = plt.CornerPlotPanel()

        # Build the bottom panel.
        self.bottom_panel = JPanel()
        self.bottom_panel.setLayout(BorderLayout())
        self.bottom_panel.setPreferredSize(Dimension(1200, 600))
        self.bottom_panel.add(self.bottom_left_panel, BorderLayout.WEST)
        self.bottom_panel.add(self.bottom_right_panel)
        
        
        self.add(self.top_panel, BorderLayout.NORTH)
#         self.add(self.profile_plots_panel)
        self.add(self.bottom_panel, BorderLayout.SOUTH)

        
        for panel in self.profile_plot_panels:
            panel.set_data(utils.linspace(0, 1, 10), utils.linspace(0, 1, 10))

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
            
        # Save data.
        self.panel.measurements = measurements
        self.panel.moments_dict = moments_dict
        self.panel.tmats_dict = tmats_dict
        
        
class ClearFilesButtonListener(ActionListener):
    
    def __init__(self, panel):
        self.panel = panel
    
    def actionPerformed(self, event):
        self.panel.clear_data()        
        print 'Cleared data.'
              
            
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
        ACTIVE_NODE_IDS = measurements[0].node_ids # Read this from GUI later.
        moments_list, tmats_list = [], []
        for meas_node_id in ACTIVE_NODE_IDS:
            moments_list.extend(moments_dict[meas_node_id])
            tmats_list.extend(tmats_dict[meas_node_id])
        # Reconstruct and print results.
        Sigma = analysis.reconstruct(tmats_list, moments_list, verbose=2, solver='lsmr')
        beam_stats = analysis.BeamStats(Sigma)
        print ''
        beam_stats.print_all()