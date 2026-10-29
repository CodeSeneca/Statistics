"""Main module"""

import sys
import numpy as np
from copy import deepcopy
from data import SimulationBox
from input_output import InputOutput
from optimize import Optimize

class MonteCarlo:
    """Class MonteCarlo: Driver class to run the Monte Carlo Simulation"""

    def __init__(self):
        # Object to handle IO related issues
        self._io = InputOutput(sys.argv)

        # Object for the actual Monte Carlo Simulation
        self._opt = Optimize(path = "/home/bechtel/progs/MLIP_vib_benchmarking_project/MACE_models/mace-mh-1.model",
                            head = "omol",
                            dispersion = False,
                            dtype = "float64",
                            device = "cpu",
                            cueq = False,
                            fmax = 0.20)

        # Simulation box of the surface, of the adsorbate and the combined system
        self._surface = None
        self._adsorbate = None
        self._combined_system = None

        # Array of booleans representing the occupation pattern
        self._occupation_pattern = None

        # Total number of Monte Carlo steps, default: 10
        self._num_steps = 10

        # Filename of the output file
        self._out_filename = "mc.log"

    def _make_random_occupation(self, num_adsorbed_places) -> None:
        """Generate a random occupation pattern with num_adsorbed_places
        adsorbed places"""

        # Reset the occupation pattern
        self._occupation_pattern[:] = 0

        # List of random indeces
        random_indeces = []

        # Generate num_adsorbed_places random indeces and set
        # the corresponding occupation to 1
        for i in range(num_adsorbed_places):
            # Generate a random number as long as the random index was already
            # chosen
            while(True):
              random_index = np.random.randint(
                      low=0, high=len(self._occupation_pattern))
              if random_index not in random_indeces:
                  self._occupation_pattern[random_index] = 1
                  random_indeces.append(random_index)
                  break

    def _place_jump(self) -> None:
        """Let one adsorbate jump randomically by changing two places 0 and 1"""
        pass

    def _set_adsorbate(self, anchor_index: int, place_index: int, offset: float) -> None:
        """Set an adsorbate onto the surface"""

        # Anchor atom that links the adsorbate with the surface
        anchor_atom = self._adsorbate.get_atoms()[anchor_index]
        # Place on the surface where to adsorb to
        place_atom = self._surface.get_atoms()[place_index]
        # Distance vector between anchor_atom and place_atom offsetted by offset
        offset_vector = np.array([0.0, 0.0, offset])
        dist_vector = anchor_atom.distance_vector_to(place_atom) + offset_vector

        # Shift all atoms within the adsorbate according to dist_vector
        # and add the shifted atoms to self._surface
        for atom in self._adsorbate.get_atoms():
            shifted_atom = atom.translate(dist_vector)
            self._combined_system.add_atom(shifted_atom)

    def _mc_step(self) -> float:
        """Perform a Monte Carlo step and return the energy for it"""

        # Reset self._combined_system to the clean surface
        self._combined_system = deepcopy(self._surface)

        # Change the adsorption pattern in self._occupation_pattern
        self._make_random_occupation(self._io.get_num_adsorbed_places())

        # Set the adsorbates onto the surface according to self._occupation_pattern
        for i, occ in enumerate(self._occupation_pattern):
          if occ == 1:
            self._set_adsorbate(anchor_index=self._io.get_anchor_index(), place_index=i, offset=self._io.get_offset())

        # Write the updated surface to a POSCAR file
        InputOutput.write_poscar(box=self._combined_system, filename="POSCAR")

        # Perform a geometry optimization
        self._opt.read_box("POSCAR")
        self._opt.set_pes()
        energy = self._opt.optimize()

        return energy

    def run(self) -> None:
        """Run the Monte Carlo Simulation"""

        # Old and new energy within each Monte Carlo step
        old_energy = None
        new_energy = None
        # Energy change within each Monte Carlo step between
        # new_energy and old_energy
        delta_energy = None

        # Print the help message to the console
        InputOutput.print_help()
        # Process the command line arguments
        self._io.process_arguments()

        # Read in the surface and the adsorbate
        self._surface = self._io.read_surface()
        self._adsorbate = self._io.read_adsorbate()
        num_places = self._surface.get_num_atoms()
        num_adsorbed_places = self._io.get_num_adsorbed_places()
        occupied_fraction = (num_adsorbed_places/num_places) * 100
        print(f"Number of adsorbed places: {num_adsorbed_places}")
        print(f"Number of free places:     {num_places - num_adsorbed_places}")
        print("------------------------------------")
        print(f"Total:                     {num_places}")
        print(f"Occupied fraction:         {occupied_fraction:.2f} %\n")
        print(f"The adsorbate adsorbs via atom no. {self._io.get_anchor_index()} with an offset of {self._io.get_offset()} (in direct coordinates)\n")

        # Generate a random occupation pattern according to
        # num_adsorbed_places
        self._occupation_pattern = np.zeros(num_places, dtype=int)
        self._make_random_occupation(num_adsorbed_places)

        # Perform self._num_steps Monte Carlo steps
        print("Entering Monte Carlo loop ...\n")

        with open(self._out_filename, 'w') as out_file:
          old_energy = self._mc_step()
          out_file.write(f"{0} {old_energy:15.10f}\n")
          out_file.flush()

          for i in range(1, self._num_steps):
            new_energy = self._mc_step()
            delta_energy = new_energy - old_energy

            out_file.write(f"{i} {new_energy:15.10f} {delta_energy:15.10f}\n")
            out_file.flush()

            old_energy = new_energy

def main() -> None:
    """Main function to drive everything"""

    mc = MonteCarlo()
    mc.run()

if __name__ == "__main__":
    main()
