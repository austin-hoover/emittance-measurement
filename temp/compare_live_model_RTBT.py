from xal.smf import Accelerator
from xal.smf.data import XMLDataManager
from xal.sim.scenario import Scenario

accelerator = XMLDataManager.loadDefaultAccelerator()
sequence = accelerator.getComboSequence('RTBT')
scenario = Scenario.newScenarioFor(sequence)

print 'Node ID       | model   | live'
print '-------------------------------'
for node in sequence.getNodesOfType('quad'):
    if 'Ring' in node.getId():
        continue
    elements = scenario.elementsMappedTo(node)
    B_model = elements[0].getMagField()
    B_live = node.getField()
    print '{} | {:<6.3f} | {:.3f}'.format(node.getId(), B_model, B_live)
    
exit()