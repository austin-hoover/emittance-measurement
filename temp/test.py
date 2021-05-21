from xal.smf import Accelerator
from xal.smf.data import XMLDataManager
from xal.sim.scenario import Scenario

accelerator = XMLDataManager.loadDefaultAccelerator()
sequence = accelerator.getComboSequence('Ring')
scenario = Scenario.newScenarioFor(sequence)

rf_ids = ['Ring_RF:Cav01', 'Ring_RF:Cav02', 'Ring_RF:Cav03', 'Ring_RF:Cav04']

for node in sequence.getNodesOfType('rebuncher'):
    print node.getId(), node.getCavAmpAvg(), node.getCavFreq()
    
exit()