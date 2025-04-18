import sys
import os

# Obtain the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the sys.path
sys.path.append(parent_dir)

# Import the codebase module
from codebase import CodeBase
from vgqeccode import VGQECCode

from original_code import RepetitionCode, PerfectCode


class HybridScheme(VGQECCode):
    def __init__(self):
        super().__init__()
        self.rec_parameters = None
        self.num_para_rec = None

    def set_parameters_rec(self, para):
        self.rec_parameters = para
        self.gen_rec_kraus()

from .par_three import VGQEC_three_hybrid
from .par_five import VGQEC_five_hybrid