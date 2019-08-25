"""Farms multiobjective optimisation for salamander"""

import numpy as np

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

from farms_bullet.simulations.simulation_options import SimulationOptions
from farms_bullet.experiments.salamander.simulation import SalamanderSimulation
from farms_bullet.animats.amphibious.animat_options import AmphibiousOptions
from farms_bullet.evolution.evolution import run_evolution


class SalamanderEvolution(FloatProblem):
    """Benchmark problem"""

    def __init__(self):
        super(SalamanderEvolution, self).__init__()
        self.number_of_variables = 18
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["Distance (negative)", "Consumption"]

        self.lower_bound = [-np.inf for _ in range(self.number_of_variables)]
        self.upper_bound = [+np.inf for _ in range(self.number_of_variables)]

        # Body stand amplitude
        self.lower_bound[0], self.upper_bound[0] = 0, 2*np.pi/11
        # Legs amplitudes
        self.lower_bound[1], self.upper_bound[1] = 0, +np.pi/2
        self.lower_bound[2], self.upper_bound[2] = 0, +np.pi/4
        self.lower_bound[3], self.upper_bound[3] = 0, +np.pi
        self.lower_bound[4], self.upper_bound[4] = 0, +np.pi/4
        # Legs offsets
        self.lower_bound[5], self.upper_bound[5] = -np.pi/4, +np.pi/4
        self.lower_bound[6], self.upper_bound[6] = 0, +np.pi/4
        self.lower_bound[7], self.upper_bound[7] = -np.pi/4, +np.pi/4
        self.lower_bound[8], self.upper_bound[8] = 0, +np.pi/2
        # Connectivity
        self.lower_bound[9], self.upper_bound[9] = 0, 1e3
        self.lower_bound[10], self.upper_bound[10] = 0, 1e3
        self.lower_bound[11], self.upper_bound[11] = 0, 1e3
        self.lower_bound[12], self.upper_bound[12] = 0, 1e3
        self.lower_bound[13], self.upper_bound[13] = 0, 1e3
        self.lower_bound[14], self.upper_bound[14] = -1e3, 0
        self.lower_bound[15], self.upper_bound[15] = 0, 1e3
        self.lower_bound[16], self.upper_bound[16] = 0, 1e3
        self.lower_bound[17], self.upper_bound[17] = 0, 1e3

        # Initial solutions
        animat_options = AmphibiousOptions(
            # collect_gps=True,
            scale=1
        )
        network = animat_options.control.network
        legs_amplitudes = network.oscillators.get_legs_amplitudes()
        legs_offsets = network.joints.get_legs_offsets()
        self.initial_solutions = [
            [
                network.oscillators.get_body_stand_amplitude(),
                legs_amplitudes[0],
                legs_amplitudes[1],
                legs_amplitudes[2],
                legs_amplitudes[3],
                legs_offsets[0],
                legs_offsets[1],
                legs_offsets[2],
                legs_offsets[3],
                network.connectivity.weight_osc_body,
                network.connectivity.weight_osc_legs_internal,
                network.connectivity.weight_osc_legs_opposite,
                network.connectivity.weight_osc_legs_following,
                network.connectivity.weight_osc_legs2body,
                network.connectivity.weight_sens_contact_i,
                network.connectivity.weight_sens_contact_e,
                network.connectivity.weight_sens_hydro_freq,
                network.connectivity.weight_sens_hydro_amp
            ]
        ]
        self._initial_solutions = self.initial_solutions.copy()

    @staticmethod
    def get_name():
        """Name"""
        return "Salamander evolution"

    def create_solution(self):
        new_solution = FloatSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints
        )
        new_solution.variables = np.random.uniform(
            self.lower_bound,
            self.upper_bound,
            self.number_of_variables
        ) if not self._initial_solutions else self._initial_solutions.pop()
        return new_solution

    @staticmethod
    def evaluate(solution, evolution=True):
        """Evaluate"""
        # Animat options
        animat_options = AmphibiousOptions(
            # collect_gps=True,
            scale=1
        )
        network = animat_options.control.network
        network.oscillators.body_head_amplitude = 0
        network.oscillators.body_tail_amplitude = 0
        network.oscillators.set_body_stand_amplitude(
            solution.variables[0]
        )
        network.oscillators.set_legs_amplitudes([
            solution.variables[1],
            solution.variables[2],
            solution.variables[3],
            solution.variables[4]
        ])
        network.joints.set_legs_offsets([
            solution.variables[5],
            solution.variables[6],
            solution.variables[7],
            solution.variables[8]
        ])
        # Connectivity
        network.connectivity.weight_osc_body = solution.variables[9]
        network.connectivity.weight_osc_legs_internal = solution.variables[10]
        network.connectivity.weight_osc_legs_opposite = solution.variables[11]
        network.connectivity.weight_osc_legs_following = solution.variables[12]
        network.connectivity.weight_osc_legs2body = solution.variables[13]
        network.connectivity.weight_sens_contact_i = solution.variables[14]
        network.connectivity.weight_sens_contact_e = solution.variables[15]
        network.connectivity.weight_sens_hydro_freq = solution.variables[16]
        network.connectivity.weight_sens_hydro_amp = solution.variables[17]
        # network.oscillators.body_stand_shift = np.pi/4
        # animat_options.control.drives.forward = 4

        # Simulation options
        simulation_options = SimulationOptions.with_clargs()
        simulation_options.headless = evolution
        simulation_options.fast = evolution
        simulation_options.timestep = 5e-3
        simulation_options.duration = 10
        simulation_options.units.meters = 1
        simulation_options.units.seconds = 1
        simulation_options.units.kilograms = 1
        simulation_options.plot = False

        # Setup simulation
        sim = SalamanderSimulation(
            simulation_options=simulation_options,
            animat_options=animat_options
        )

        # Run simulation
        sim.run()

        # Extract fitness
        power = np.sum(
            np.asarray(
                sim.elements.animat.data.sensors.proprioception.motor_torques()
            )*np.asarray(
                sim.elements.animat.data.sensors.proprioception.velocities_all()
            )
        )*simulation_options.timestep/simulation_options.duration
        position = np.array(
            sim.elements.animat.data.sensors.gps.urdf_position(
                iteration=sim.iteration-1,
                link_i=0
            )
        )
        distance = np.linalg.norm(position[:2])
        # Penalty
        if (not 1 < -position[0] < 5) or (not -3 < position[1] < 3):
            distance -= 1e3
            power += 1e3
        # Objectives
        solution.objectives[0] = -distance  # Distance along x axis
        solution.objectives[1] = power  # Energy

        # Terminate simulation
        sim.end()

        return solution


def main():
    """Main"""
    run_evolution(
        problem=SalamanderEvolution(),
        n_pop=40,
        n_gen=5
    )


if __name__ == '__main__':
    main()
