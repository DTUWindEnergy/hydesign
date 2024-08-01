from abc import ABC, abstractmethod
import openmdao.api as om

class OptimizationProblem(ABC):
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.constraints = []
        self.objective = None

    @abstractmethod
    def add_inputs(self, inputs):
        pass

    @abstractmethod
    def add_outputs(self, outputs):
        pass

    @abstractmethod
    def add_constraints(self, constraints):
        pass

    @abstractmethod
    def set_objective(self, expression):
        pass

    @abstractmethod
    def collect_data(self):
        pass

class MyOptimizationProblem(OptimizationProblem):
    def add_inputs(self, inputs):
        for name, value in inputs:
            self.inputs.append((name, value))

    def add_outputs(self, outputs):
        for name, value in outputs:
            self.outputs.append((name, value))

    def add_constraints(self, constraints):
        for expression in constraints:
            self.constraints.append(expression)

    def set_objective(self, expression):
        self.objective = expression

    def collect_data(self):
        return {
            'inputs': self.inputs,
            'outputs': self.outputs,
            'constraints': self.constraints,
            'objective': self.objective
        }

# Example instances
opt_problem1 = MyOptimizationProblem()
opt_problem1.add_inputs([('x1', 1.0), ('x2', 2.0)])
opt_problem1.add_outputs([('y1', 0.0)])
opt_problem1.add_constraints(['x1 >= 0', 'x2 <= 10', 'x1 + x2 <= 15'])
opt_problem1.set_objective('minimize y1')

opt_problem2 = MyOptimizationProblem()
opt_problem2.add_inputs([('x3', 0.0), ('x4', 0.0), ('x5', 0.0)])
opt_problem2.add_outputs([('y2', 0.0)])
opt_problem2.add_constraints(['x3 + 2*x4 + 3*x5 <= 10', '2*x3 + x4 + x5 >= 5', 'x3 >= 0', 'x4 >= 0', 'x5 >= 0'])
opt_problem2.set_objective('minimize y2')

class CombinedOptimizationSolver:
    def __init__(self, problem_selectors):
        self.problem_selectors = problem_selectors
        self.problems = [opt_problem1, opt_problem2]

    def solve(self):
        combined_data = {
            'inputs': [],
            'outputs': [],
            'constraints': [],
            'objective': []
        }

        for selector in self.problem_selectors:
            problem = self.problems[selector - 1]
            data = problem.collect_data()
            combined_data['inputs'].extend(data['inputs'])
            combined_data['outputs'].extend(data['outputs'])
            combined_data['constraints'].extend(data['constraints'])
            combined_data['objective'].append(data['objective'])

        # OpenMDAO model setup
        model = om.Group()
        indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])

        # Add inputs
        for name, value in combined_data['inputs']:
            indeps.add_output(name, value)

        # Add outputs and appropriate equations
        for output in combined_data['outputs']:
            if 'y1' in output:
                model.add_subsystem('comp1', om.ExecComp('y1 = x1**2 + x2**2'), promotes=['*'])
            if 'y2' in output:
                model.add_subsystem('comp2', om.ExecComp('y2 = x3 + x4 + x5'), promotes=['*'])

        # Add constraints
        for constraint in combined_data['constraints']:
            model.add_constraint(constraint)

        # Combine objectives (here we simply sum the objectives, but you can customize this)
        combined_objective = ' + '.join(combined_data['objective'])
        model.add_objective(combined_objective)

        # Set up the problem
        prob = om.Problem(model)
        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'Cplex'

        # Setup and run the optimization
        prob.setup()
        prob.run_driver()

        # Get results
        results = {name: prob.get_val(name) for name, _ in combined_data['inputs']}
        results.update({name: prob.get_val(name) for name, _ in combined_data['outputs']})
        return results

# Example usage
solver = CombinedOptimizationSolver(problem_selectors=[1, 2])
results = solver.solve()
print("Results for Combined Problem:")
print(results)
