"""Main module"""

import sys
import numpy as np
from data import Atom
from input_output import InputOutput
from monte_carlo import MonteCarlo

class Statistics:
    """Class Statistics: Driver class to run the simulation"""

    def __init__(self):
        # Object to handle IO related issues
        self._io = InputOutput(sys.argv)

        # Object for the actual Monte Carlo Simulation
        self._mc = MonteCarlo(path = "/home/bechtel/progs/MLIP_vib_benchmarking_project/MACE_models/mace-mh-1.model",
                              head = "omol",
                              dispersion = False,
                              dtype = "float64",
                              device = "cpu",
                              cueq = False,
                              fmax = 0.02
                              )

        # Simulation box of the surface and of the adsorbate
        self._surface = None
        self._adsorbate = None

        # Array of booleans representing the occupation pattern
        self._occupation_pattern = None

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
            self._surface.add_atom(shifted_atom)

    def run(self) -> None:
        """Run the Statistics Simulation"""

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

        # Generate the random occupation pattern according to
        # num_adsorbed_places
        self._occupation_pattern = np.zeros(num_places, dtype=int)
        self._make_random_occupation(num_adsorbed_places)

        # Set the adsorbates onto the surface according to self._occupation_pattern
        print("Setting the adsorbates onto the surface ...\n")
        for i, occ in enumerate(self._occupation_pattern):
          if occ == 1:
            self._set_adsorbate(anchor_index=self._io.get_anchor_index(), place_index=i, offset=self._io.get_offset())

        # Write the updated surface to a POSCAR file
        InputOutput.write_poscar(box=self._surface, filename="POSCAR")

        # Initialize the Monte Carlo Simulation
        self._mc.read_box("POSCAR")
        self._mc.set_pes()
        new_energy = self._mc.optimize()

def main() -> None:
    """Main function to drive everything"""

    statistics = Statistics()
    statistics.run()

if __name__ == "__main__":
    main()
