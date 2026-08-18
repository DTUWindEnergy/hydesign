.. thesis_009:

Sizing optimization of hybrid power plants using heuristic methods
==========================================================================================================================

**Abstract**

The ongoing expansion of renewable energy has driven the development of Hybrid Power
Plants (HPP), which co-locate wind, solar PV, and battery storage to mitigate variability
and improve grid services. Sizing these systems optimally is, however, a challenging
optimization problem that limits the applicability of standard optimization methods and
surrogate based methods, which suffer from poor scalability as problem complexity grows.
This thesis investigates heuristic optimization methods as an alternative for HPP sizing,
with the objective of comparing their solution quality and computational efficiency against
surrogate based approaches. The HPP sizing tool HyDesign, developed at DTU Wind, is
extended to incorporate heuristic algorithms, which are evaluated in terms of convergence
behavior and robustness. The thesis further extends the framework to a multi-objective
formulation, simultaneously optimizing the carbon emission offsets and revenue, to explore
the trade-offs between economic performance and environmental impact.
For the single objective problem, both the Genetic Algorithm and Particle Swarm Optimization
find higher quality solutions than the surrogate based model within a comparable
computational budget. They also converge toward similar HPP designs across independent
runs, demonstrating their ability to locate near global optima reliably. For the
multi-objective problem, both algorithms produce well spread Pareto fronts spanning the
full trade-off between profitability and carbon offset, with NSGA-II achieving more uniform
coverage and reaching more extreme carbon emission offset values than VEPSO due
to its native mechanism. The trade-off is found to be primarily driven by battery sizing,
which emerges as the key design lever between economic and environmental performance.
These results underline the potential of heuristic methods for HPP design, offering developers
a practical tool to identify critical sizing decisions and navigate trade-offs between
profitability and environmental impact.

**Cite this**

Cortanze, Paul Roero de, 2026

**Link**

Download `here
<https://findit.dtu.dk/catalog/6a727ffdcaeb6518a9994815>`_.

.. tags:: Thesis, Multi Objective Optimization, Optimization Algorithms