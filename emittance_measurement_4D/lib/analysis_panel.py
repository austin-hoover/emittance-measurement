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
import utils
import xal_helpers


DIAG_WIRE_ANGLE = utils.radians(-45.0)
REC_NODE_ID = 'Begin_Of_RTBT1'


class AnalysisPanel(JPanel):
    
    def __init__(self, kin_energy=1e9):
        JPanel.__init__(self)
        self.setLayout(BorderLayout())
        self.kin_energy = kin_energy
        self.start_fresh()
        self.build_panel()
        
    def start_fresh(self):
        self.measurements = []
        self.moments_dict, self.moments_list = dict(), []
        self.tmats_dict, self.tmats_list = dict(), []
        
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
        
        # Add components to panel.
        self.panel1 = JPanel()
        self.panel1.add(self.load_files_button)
        self.panel1.add(self.clear_files_button)
        
        self.panel2 = JPanel()
        self.panel2.add(self.reconstruct_covariance_button)
        
        self.add(self.panel1, BorderLayout.WEST)
        self.add(self.panel2)
    


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
                
        # Form list of transfer matrices and measured moments.
        moments_list, tmats_list = [], []
        for measurement in measurements:
            for meas_node_id in measurement.node_ids:
                moments_list.extend(moments_dict[meas_node_id])
                tmats_list.extend(tmats_dict[meas_node_id])
            
        # Save everything.
        self.panel.measurements = measurements
        self.panel.moments_list = moments_list
        self.panel.moments_dict = moments_dict
        self.panel.tmats_list = tmats_list
        self.panel.tmats_dict = tmats_dict
        
        
class ClearFilesButtonListener(ActionListener):
    
    def __init__(self, panel):
        self.panel = panel
    
    def actionPerformed(self, event):
        self.panel.start_fresh()        
        print 'Cleared data.'
              
            
class ReconstructCovarianceButtonListener(ActionListener):
    
    def __init__(self, panel):
        self.panel = panel
        
    def actionPerformed(self, event):
        if not self.panel.measurements:
            raise ValueError('No wire-scanner files have been loaded.')
        tmats_list = self.panel.tmats_list
        moments_list = self.panel.moments_list
        Sigma = analysis.reconstruct(tmats_list, moments_list, verbose=2, solver='lsmr')
        beam_stats = analysis.BeamStats(Sigma)
        
        print ''
        beam_stats.print_all()