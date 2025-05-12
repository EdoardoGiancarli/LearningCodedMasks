r"""
 ___   ____     ___    ____                                   
|_ _| |  _ \   / _ \  / ___|                                  
 | |  | |_) | | | | | \___ \                                  
 | |  |  _ <  | |_| |  ___) |                                 
|___| |_| \_\  \___/  |____/                              _   
|  \/  | __ _ _ __   __ _  __ _  ___ _ __ ___   ___ _ __ | |_ 
| |\/| |/ _` | '_ \ / _` |/ _` |/ _ \ '_ ` _ \ / _ \ '_ \| __|
| |  | | (_| | | | | (_| | (_| |  __/ | | | | |  __/ | | | |_ 
|_|  |_|\__,_|_| |_|\__,_|\__, |\___|_| |_| |_|\___|_| |_|\__|
                          |___/                               


This package contains useful methods for handling IROS-based analyses.
"""

from .analyze import perform_iros
from .analyze import gen_params_log
from .analyze import compute_params
from .analyze import compare_w_catalog
from .analyze import dict2df

from .handle import save_iros_output, load_iros_output
from .handle import save_iros_data, load_iros_data
from .handle import save_sky, load_sky
from .handle import fit_WCS
from .handle import camera_composition

from .show import crop
from .show import plot_distr
from .show import plot_sequence
from .show import enhance_slices
from .show import make_sky
from .show import plot_sky

from .stats import iros_radec_accuracy
from .stats import iros_radec_accuracy_finecoord
from .stats import iros_fluence
from .stats import iros_sources_res


# end