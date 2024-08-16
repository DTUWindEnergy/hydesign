import numpy as np
from docplex.mp.model import Model
import inspect
import os

def expand_to_lifetime(x, life_h = 25*365*24, weeks_per_season_per_year=None):
    """
    Expands (by repeating) a given variable to match an expected lifetime length.

    If weeks_per_season_per_year is an int then it will first build a year out of the selected weeks

    Parameters
    ----------
    x: input variable
    life_h: lifetime in hours.
    weeks_per_season_per_year: None or int.

    Returns
    -------
    x_ext: extended variable
    """
    if weeks_per_season_per_year == None:

        # Extend the data to match the expected lifetime
        len_x = len(x)
        N_repeats = int(np.ceil(life_h/len_x))

    else:
        x_ext = np.array([])

        # extend selected weeks into a year: 4 season of 13 weeks + one extra day.
        for x_batch in split_in_batch(x,weeks_per_season_per_year*7*24):
            x_ext = np.append( x_ext, np.tile(x_batch,20)[:24*7*13] )
        x_ext = np.append( x_ext, x_batch[-24:] )

        # extend the constructed year to match the expected lifetime
        x = x_ext
        N_repeats = int(np.ceil(life_h/365*24))

    return np.tile(x,N_repeats)[:life_h]

def split_in_batch(array, N):
    batch = []
    counter = 0
    while counter * N < len(array):
        if (counter + 1) * N > len(array):
            end = len(array)
        else:
            end = (counter + 1) * N
        batch += [array[counter * N:end]]
        counter += 1
    return batch


def print_cplex_model_details(mdl: Model):
    # Print model name
    print(f"Model name: {mdl.name}")

    # Print number of variables
    num_vars = mdl.number_of_variables
    print(f"Number of variables: {num_vars}")

    # Print number of constraints
    num_constraints = mdl.number_of_constraints
    print(f"Number of constraints: {num_constraints}")

    # Print objective function
    objective = mdl.objective_expr
    sense = "Minimize" if mdl.is_minimized() else "Maximize"
    print(f"Objective function: {sense} {objective}")

    # Print variables details
    print("\nVariables:")
    for var in mdl.iter_variables():
        print(f"  - {var}: {var.lb} <= {var} <= {var.ub}")

    # Print constraints details
    print("\nConstraints:")
    for cst in mdl.iter_constraints():
        print(f"  - {cst.name}: ") # {cst.lhs} {cst.sense} {cst.rhs}

    # Print model parameters
    print("\nParameters:")
    for param in mdl.parameters:
        print(f"  - {param.name}: {param.value}")


def insert_ems_section(section):
    code = inspect.getsource(section)
    # Split the code into lines
    lines = code.splitlines()

    # Remove the function definition line (the first line) and remove leading indentation from the remaining lines
    indented_code = "\n".join(line[4:] if line.startswith("    ") else line for line in lines[1:]) +"\n"

    # Save the combined string to a Python (.py) file
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the full path for the new file in the same directory as the script
    file_path = os.path.join(script_dir, "builded_ems.py")

    # Write to the file
    with open(file_path, "a") as file:
        file.write(indented_code)

def append_ems_section(code, section):
    section_code = inspect.getsource(section)
    # Split the code into lines
    lines = section_code.splitlines()

    # Remove the function definition line (the first line) and remove leading indentation from the remaining lines
    indented_code = "\n".join(line[4:] if line.startswith("    ") else line for line in lines[1:]) +"\n"

    code = code + indented_code

    return code

def get_path_builded_ems():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the full path for the new file in the same directory as the script
    file_path = os.path.join(script_dir, "builded_ems.py")
    return file_path


