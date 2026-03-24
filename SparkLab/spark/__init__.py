r"""
                              _      _ 
 ___   _ __     __ _   _ __  | | __ | |
/ __| | '_ \   / _` | | '__| | |/ / | |
\__ \ | |_) | | (_| | | |    |   <  |_|
|___/ | .__/   \__,_| |_|    |_|\_\ (_)
      |_|                              

"""

__author__ = 'Edoardo Giancarli'
__version__ = '0.1'


from .handle import save_dataset
from .handle import load_dataset
from .handle import save_model
from .handle import load_model

from .processing import process_data
from .processing import get_dataloaders


# end