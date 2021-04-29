import time
from pprint import pprint
from lib.phase_controller import PhaseController
from lib.phase_controller import init_twiss, design_betas_at_target
from lib.helpers import loadRTBT



sequence = loadRTBT()

C = PhaseController(sequence, 'RTBT_Diag:WS24', init_twiss)

mux, muy = C.get_ref_ws_phases()
C.set_ref_ws_phases(mux + 0.25, muy, verbose=2)
C.set_betas_at_target(design_betas_at_target, verbose=2)


exit()