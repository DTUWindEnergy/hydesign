import pandas as pd
import numpy as np
import ast
import os
import xarray as xr

from hydesign.weather.weather import extract_weather_for_HPP, select_years
from hydesign.look_up_tables import lut_filepath
from hydesign.assembly.hpp_assembly_base import hpp_model as hpp_model_base
from hydesign.assembly.hpp_assembly_BM import hpp_model as hpp_model_BM
from hydesign.assembly.hpp_assembly_P2X import hpp_model_P2X
from hydesign.assembly.hpp_assembly_constantoutput import hpp_model_constant_output
from hydesign.assembly.hpp_assembly_hybridization_pv import hpp_model as hpp_model_hybridize_pv
from hydesign.assembly.hpp_assembly_hybridization_wind import hpp_model as hpp_model_hybridize_wind


def hpp_model(excel_file_path, verbose=True, **kwargs):
    """Initialization of the hybrid power plant evaluator

    Parameters
    ----------
    excel_file_path : Path to the Excel file containing parameters
    kwargs : Optional keyword arguments to overwrite parameters from the Excel file
    """

    excel_data = pd.read_excel(excel_file_path, sheet_name=None)

    operation_usecase = excel_data['dashboard'].iloc[1, 2]
    if operation_usecase == "Base":
        hpp = hpp_model_base(excel_data)
    elif operation_usecase == "BM":
        hpp = hpp_model_BM(excel_data)
    elif operation_usecase == "P2X":
        hpp = hpp_model_P2X(excel_data)
    elif operation_usecase == "Constant output":
        hpp = hpp_model_constant_output(excel_data)
    elif operation_usecase == "hybridisation_pv":
        hpp = hpp_model_hybridize_pv(excel_data)
    elif operation_usecase == "hybridisation_wind":
        hpp = hpp_model_hybridize_wind(excel_data)
    else:
        print("Invalid choice")

    return hpp


def mkdir(dir_):
    if str(dir_).startswith('~'):
        dir_ = str(dir_).replace('~', os.path.expanduser('~'))
    try:
        os.stat(dir_)
    except BaseException:
        try:
            os.mkdir(dir_)
            #Path(dir_).mkdir(parents=True, exist_ok=True)
        except BaseException:
            pass
    return dir_