from common import *
from plot import generate_plots
from common import Protocol
from tables import generate_tables
from glob import glob


qp.environ['SAMPLE_SIZE'] = 100

protocol = Protocol.VAR_D3_PREV

result_dir = './results/*'

results = [Result.load(r) for r in glob(result_dir)]

# ------------------------------------
# tables
# ------------------------------------
generate_tables(protocol, results)

# ------------------------------------
# plots
# ------------------------------------
generate_plots(protocol, results)





