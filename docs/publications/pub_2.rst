.. pub_2:

HyDesign: a tool for sizing optimization for grid-connected hybrid power plants including wind, solar photovoltaic, and Li-ion batteries.
==========================================================================================================================================

**Abstract**

Hybrid renewable power plants consisting of collocated wind, solar photovoltaic (PV), and lithium-ion battery storage connected behind a single grid connection can provide additional value to the owners and society in comparison to individual technology plants, such as those that are only wind or only PV. The hybrid power plants considered in this article are connected to the grid and share electrical infrastructure costs across different generation and storing technologies. In this article, we propose a methodology for sizing hybrid power plants as a nested-optimization problem: with an outer sizing optimization and an internal operation optimization. The outer sizing optimization maximizes the net present values over capital expenditures and compares it with standard designs that minimize the levelized cost of energy. The sizing problem formulation includes turbine selection (in terms of rated power, specific power, and hub height), a wind plant wake loss surrogate, simplified wind and PV degradation models, battery degradation, and operation optimization of an internal energy management system. The problem of outer sizing optimization is solved using a new parallel "efficient global optimization"algorithm. This new algorithm is a surrogate-based optimization method that ensures a minimal number of model evaluations but ensures a global scope in the optimization. The methodology presented in this article is available in an open-source tool called HyDesign. The hybrid sizing algorithm is applied for a peak power plant use case at different locations in India where renewable energy auctions impose a monetary penalty when energy is not supplied at peak hours. We compare the hybrid power plant sizing results when using two different objective functions: The levelized cost of energy (LCoE) or the relative net present value with respect to the total capital expenditure costs (NPV/CH). Battery storage is installed only on NPV/CH-based designs, while the hybrid design, including wind, solar, and battery, only occurs on the site with good wind resources. Wind turbine selection on this site prioritizes cheaper turbines with a lower hub height and lower rated power. The number of batteries replaced changes at the different sites, ranging between two or three units over the lifetime. A significant oversizing of the generation in comparison to the grid connection occurs on all NPV/CH-based designs. As expected LCoE-based designs are a single technology with no batteries.

**Cite this**

Murcia Leon, JP, Habbou, H, Friis-Møller, M, Gupta, M, Zhu, R, and Das, Kaushik. HyDesign: a tool for sizing optimization for grid-connected hybrid power plants including wind, solar photovoltaic, and Li-ion batteries. Wind Energy Science Discussions;2023:1–22. DOI: 10.5194/wes-9-759-2024

**Link**

Download `here
<https://backend.orbit.dtu.dk/ws/portalfiles/portal/358061438/wes-9-759-2024.pdf>`_.

.. tags:: Sizing, HyDesign, Hybrid Power Plants, Optimization, Surrogate based modeling