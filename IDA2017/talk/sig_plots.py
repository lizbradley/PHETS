import numpy as np

from signals.signals import Signal, BaseTrajectory
from signals.plots import ts

from Tools import sec_to_samp

# piano_sig = BaseTrajectory(
# 	'../../datasets/time_series/C135B/49-C135B.txt',
# 	crop=sec_to_samp((1.72132, 1.7679)),
# )
# plot_signal('../../paper/talk/piano_sig.png', piano_sig, title='upright piano')
#
#
# viol_sig = BaseTrajectory(
# 	'../../datasets/time_series/viol/40-viol.txt',
# 	crop=(35000, 140000)
# )
#
#
# plot_signal('../../paper/talk/viol_sig.png', viol_sig, title='viol')
#
# clarinet_sig = BaseTrajectory(
# 	'../../datasets/time_series/clarinet/sustained/high_quality/40-clarinet-HQ.txt',
# 	crop=(75000, 180000)
# )
#
#
# plot_signal('../../paper/talk/clarinet_sig.png', clarinet_sig, title='clarinet')
#


upright_sig = BaseTrajectory(
	'../../datasets/time_series/piano_revisit/C144F/a440/07-consolidated.txt',

)


ts('../../paper/sigs/fig_6/upright_sig.png', upright_sig, title='upright piano')

grand_sig = BaseTrajectory(
	'../../datasets/time_series/piano_revisit/C134C/a440/07- C134C-consolidated.txt',
)


ts('../../paper/sigs/fig_6/grand_sig.png', grand_sig, title='grand piano')
