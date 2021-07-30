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