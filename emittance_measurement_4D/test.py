from xal.smf import Accelerator
from xal.smf import AcceleratorSeq 
from xal.smf.data import XMLDataManager

accelerator = XMLDataManager.loadDefaultAccelerator()
sequence = accelerator.getComboSequence('Ring')

