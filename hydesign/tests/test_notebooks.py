import os
from pathlib import Path

import pytest
import xarray

import hydesign
from hydesign.tests.notebook import Notebook


def get_notebooks():
    def get(path):
        return [
            Notebook(path + f)
            for f in [f for f in os.listdir(path) if f.endswith(".ipynb")]
        ]

    path = os.path.dirname(hydesign.__file__) + "/../docs/notebooks/"
    return get(path)  # + get(path + "exercises/")


notebooks = get_notebooks()
excluded = [
    "constant_output.ipynb",  # takes 6 min
    "evaluate_with_reliability.ipynb",  # takes 3 min
    "PyWake_P2X_Example.ipynb",  # takes 2 min
    "Simple_Sizing_Example.ipynb",  # takes 18 min
    "Simple_Sizing_P2X_Example.ipynb",  # takes ??
    "sizing_with_reliability.ipynb",  # takes ??
]
notebooks = [nb for nb in notebooks if os.path.basename(nb.filename) not in excluded]


@pytest.mark.parametrize(
    "notebook", notebooks, ids=[os.path.basename(nb.filename) for nb in notebooks]
)
def test_notebooks(notebook):
    import matplotlib.pyplot as plt

    if str(
        Path(notebook.filename).relative_to(
            os.path.dirname(hydesign.__file__) + "/../docs/notebooks/"
        )
    ) in ["Optimization.ipynb"]:
        return

    def no_show(*args, **kwargs):
        pass

    plt.show = no_show  # disable plt show that requires the user to close the plot

    try:
        # print(notebook.filename)
        plt.rcParams.update({"figure.max_open_warning": 0})
        # with warnings.catch_warnings:
        #    warnings.simplefilter('error')
        notebook.check_code()
        notebook.check_links()
        notebook.remove_empty_end_cell()
        notebook.check_pip_header()
    except Exception as e:
        raise Exception(notebook.filename + " failed") from e
    finally:
        plt.close("all")
        plt.rcParams.update({"figure.max_open_warning": 20})
        xarray.set_options(display_expand_data=True)


if __name__ == "__main__":
    # print("\n".join([f.filename for f in get_notebooks()]))
    path = os.path.dirname(hydesign.__file__) + "/../docs/notebooks/"
    f = "Quickstart.ipynb"
    # test_notebooks(Notebook(path + f))
    import time

    for n in notebooks:
        base_path = os.path.dirname(hydesign.__file__) + "/../docs/notebooks/"
        test_report_path = os.path.join(base_path, f"{n.filename}_test_report.txt")
        if not os.path.exists(test_report_path):
            start = time.time()
            print(f"Testing {n.filename}")
            test_notebooks(n)
            print(f"Tested {n.filename} in {time.time() - start:.2f} seconds")
            with open(test_report_path, "w") as f:
                f.write(f"Tested {n.filename} in {time.time() - start:.2f} seconds\n")
        else:
            print(
                f"Skipping {n.filename} as test report already exists at {test_report_path}"
            )
            with open(test_report_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    print(line)
    # start = time.time()
    # notebooks
