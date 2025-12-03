r"""

__        ___                                    ____  _                   _ _       
\ \      / (_)___  ___ _ __ ___   __ _ _ __     / ___|(_)_ __ ___  ___    |_ _|_ __   ___
 \ \ /\ / /| / __|/ _ \ '_ ` _ \ / _` | '_ \    \___ \| | '_ ` _ \/ __|    | || '_ \ / __|
  \ V  V / | \__ \  __/ | | | | | (_| | | | |    ___) | | | | | | \__ \    | || | | | (__ _ 
   \_/\_/  |_|___/\___|_| |_| |_|\__,_|_| |_|   |____/|_|_| |_| |_|___/   |___|_| |_|\___(_)



Package for tests catalogue generation as input to the WISEMAN Monte Carlo simulator [1].

Ref:
    [1] Ceraudo, F. et al. Development of the end-to-end simulator of the WFM camera,
        Vol. 13093 of Society of Photo-Optical Instrumentation Engineers (SPIE)
        Conference Series, 130936T (2024)
"""


# --- SETUP ---
from .sims import define_unit_pointings
from .sims import init_cameras_mask
from .sims import config_pdf

# --- SOURCE SIM ---
from .sims import simul_coords
from .sims import simul_fluxes
from .sims import handmade_fluxes
from .sims import get_sources

# --- DATABASE HANDLING ---
from .sims import gen_data_log
from .sims import build_record
from .sims import make_catalog


# end