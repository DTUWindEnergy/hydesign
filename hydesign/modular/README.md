# HyDesign Modular Namespace

This folder is the isolated namespace for the modular implementation.

Use these import boundaries:

- `hydesign.*` outside this folder: legacy/reference implementation
- `hydesign.modular.*`: modular builder, modular source modules, and generated modular projects

Current layout:

- `hydesign/modular/project_builder/`: explicit build entrypoints and manifest logic
- `hydesign/modular/modules/`: reusable modular source contributions
- `hydesign/modular/projects/`: generated user-facing modular projects
- `hydesign/modular/tests/`: verification for the modular namespace only

Current public entrypoints:

```python
from hydesign.modular import build_project, ProjectBuildSpec
```

Current generated-project namespace:

```python
from hydesign.modular.projects.wind_solar_battery_cplex import assembly
```
