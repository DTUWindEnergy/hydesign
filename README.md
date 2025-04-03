[![pipeline status](https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign/badges/main/pipeline.svg)](https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign/-/commits/main)
[![coverage report](https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign/badges/main/coverage.svg)](https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign/commits/main)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7573040.svg)](https://doi.org/10.5281/zenodo.7573040)
[![PyPi](https://img.shields.io/pypi/v/hydesign)](https://pypi.org/project/hydesign/)
[![License](https://img.shields.io/pypi/l/hydesign)](https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign/blob/main/LICENSE)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fgitlab.windenergy.dtu.dk%2FTOPFARM%2FHyDesign.git/main)

## Welcome to Hydesign
The DTU tool for design, control and optimization of utility scale Hybrid Power- and Energy Plants inluding wind, solar, storage and P2X technologies.
- Hydesign is a non-linear model for evaluating HPP projects including: 
    - Wind turbine selection
    - Wake losses that depend on the number of turbines and the installation density of the wind plant
    - Internal energy management system operation of the plant. The operation of the HPP plant is designed in an ideal scenario without degradation and a actual operation that can include degradation of the battery, or PV plant, as well as forecast errors in wind, solar and spot prices.
    - Non-linear battery degradation
    - Detailed cost models for wind, solar and Li-ion battery
    - Detailed financial model based on weighted costs of capital per technology
    - P2X capability with power to Hydrogen

## How to cite
If you make use of HyDesign please cite this paper, which inludes a detailed description of the methodology applied in the framework:
```
@Article{wes-9-759-2024,
AUTHOR = {Murcia Leon, J. P. and Habbou, H. and Friis-M{\o}ller, M. and Gupta, M. and Zhu, R. and Das, K.},
TITLE = {HyDesign: a tool for sizing optimization of grid-connected hybrid power plants including wind, solar photovoltaic, and lithium-ion batteries},
JOURNAL = {Wind Energy Science},
VOLUME = {9},
YEAR = {2024},
NUMBER = {4},
PAGES = {759--776},
URL = {https://wes.copernicus.org/articles/9/759/2024/},
DOI = {10.5194/wes-9-759-2024}
}
```

## Source code repository (and issue tracker):
[https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign](https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign)

## License:
[MIT](https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign/blob/main/LICENSE)

## Documentation, installation, etc:
[https://topfarm.pages.windenergy.dtu.dk/hydesign](https://topfarm.pages.windenergy.dtu.dk/hydesign).


