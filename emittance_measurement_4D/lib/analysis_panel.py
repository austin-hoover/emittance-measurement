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
from java.text import NumberFormat
from java.text import DecimalFormat

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

from analysis import beam
from analysis.reconstruct import Measurement
from analysis.reconstruct import reconstruct
from analysis.reconstruct import get_sig_xy
from helpers import list_from_xal_matrix
from lib import utils


DIAG_WIRE_ANGLE = utils.radians(-45.0)
REC_NODE_ID = 'Begin_Of_RTBT1'


class AnalysisPanel(JPanel):
    
    def __init__(self, kin_energy=1e9):
        JPanel.__init__(self)
        self.setLayout(BorderLayout())
        self.kin_energy = kin_energy
        self.measurements = []
        self.moments_dict = dict()
        self.moments_list = []
        self.tmats_dict = dict()
        self.tmats_list = []
        self.build_panel()
        
    def build_panel(self):
        print 'Kinetic energy is hard coded. Do not delete this message until this is fixed.'
        accelerator = XMLDataManager.loadDefaultAccelerator()
        self.sequence = accelerator.getComboSequence('RTBT')
        self.scenario = Scenario.newScenarioFor(self.sequence)
        self.tmat_generator = TransferMatrixGenerator(self.sequence, self.kin_energy)
        
        
        self.load_files_button = JButton('Load wire-scan files')
        self.load_files_button.addActionListener(LoadFilesButtonListener(self))
        
        self.add(self.load_files_button)
        
        
        
def is_harp_file(filename):
    file = open(filename)
    for line in file:
        if 'Harp' in line:
            return True
    return False
        
        
class ReconstructCovarianceButtonListener(ActionListener):
    
    def __init__(self, panel):
        self.panel = panel
        
    def actionPerformed(self, event):
        tmats_list = self.panel.tmats_list
        moments_list = self.panel.moments_list
        Sigma = reconstruct(tmats_list, moments_list, verbose=2, solver='lsmr')
        beam_stats = beam.BeamStats(Sigma)
        beam_stats.print_all()


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
            if 'WireAnalysisFmt' not in filename or is_harp_file(filename):
                continue
            measurement = Measurement(filename)
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
                sig_xy = get_sig_xy(sig_xx, sig_yy, sig_uu, DIAG_WIRE_ANGLE)
                moments = [sig_xx, sig_yy, sig_xy]
                if meas_node_id not in moments_dict:
                    moments_dict[meas_node_id] = []
                moments_dict[meas_node_id].append(moments)
                # Store transfer matrix from reconstruction node to wire-scanner.
                tmat = self.tmat_generator.transfer_matrix(REC_NODE_ID, meas_node_id)
                if meas_node_id not in tmats_dict:
                    tmats_dict[meas_node_id] = []
                tmats_dict[meas_node_id].append(tmat)
                
        # Form list of transfer matrices and measured moments.
        moments_list, tmats_list = [], []
        for measurement in measurements:
            for meas_node_id in measurement.node_ids:
                moments_list.extend(moments_dict[meas_node_id])
                tmats_list.extend(tmats_dict[meas_node_id])
            
        # Save everything to the main panel.
        self.panel.measurements = measurements
        self.panel.moments_list = moments_list
        self.panel.moments_dict = moments_dict
        self.panel.tmats_list = tmats_list
        self.panel.tmats_dict = tmats_dict
                
        
class TransferMatrixGenerator:
    
    def __init__(self, sequence, kin_energy):
        self.sequence = sequence
        self.scenario = Scenario.newScenarioFor(sequence)
        self.kin_energy = kin_energy
        
    def sync(self, pvloggerid):
        """Sync model with machine state from PVLoggerID."""
        pvl_data_source = PVLoggerDataSource(pvloggerid)
        self.scenario = pvl_data_source.setModelSource(self.sequence, self.scenario)
        self.scenario.resync()
    
    def transfer_matrix(self, start_node_id=None, stop_node_id=None):
        """Return transfer matrix elements from start to node entrance.
        
        The node ids can be out of order.
        """
        # Set default start and stop nodes.
        if start_node_id is None:
            start_node_id = self.sequence.getNodes()[0].getId()
        if stop_node_id is None:
            stop_node_id = self.ref_ws_id     
        # Check if the nodes are in order. If they are not, flip them and
        # remember to take the inverse at the end.
        reverse = False
        node_ids = [node.getId() for node in self.sequence.getNodes()]
        if node_ids.index(start_node_id) > node_ids.index(stop_node_id):
            start_node_id, stop_node_id = stop_node_id, start_node_id
            reverse = True
        # Run the scenario.
        tracker = AlgorithmFactory.createTransferMapTracker(self.sequence)
        probe = ProbeFactory.getTransferMapProbe(self.sequence, tracker)
        probe.setKineticEnergy(self.kin_energy)
        self.scenario.setProbe(probe)
        self.scenario.run()
        # Get transfer matrix from upstream to downstream node.
        trajectory = probe.getTrajectory()
        state1 = trajectory.stateForElement(start_node_id)
        state2 = trajectory.stateForElement(stop_node_id)
        M1 = state1.getTransferMap().getFirstOrder()
        M2 = state2.getTransferMap().getFirstOrder()
        M = M2.times(M1.inverse())
        if reverse:
            M = M.inverse()
        # Return list of shape (4, 4).
        M = list_from_xal_matrix(M)
        M = [row[:4] for row in M[:4]]
        return M