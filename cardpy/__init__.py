# __init__.py
from cardpy import Data_Import
from cardpy import Data_Saving
from cardpy import Data_Sorting
from cardpy import Data_Processing
from cardpy import Tools
from cardpy import Colormaps
from cardpy import FT_Operators

# GUI_Tools is imported lazily: it requires tkinter, which is unavailable on
# headless systems and on Python builds without Tk support. Importing it
# eagerly would make `import cardpy` fail entirely in those environments.
# Access via `from cardpy import GUI_Tools` or `cardpy.GUI_Tools` as usual.
def __getattr__(name):
    if name == 'GUI_Tools':
        import importlib
        return importlib.import_module('cardpy.GUI_Tools')
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
