"""Main module"""

import sys
import numpy as np
from input_output import InputOutput

class Statistics:
    """Class Statistics: Driver class to run the simulation"""

    def __init__(self):
        # Object to handle IO related issues
        self._io = InputOutput(sys.argv)

        # Simulation box of the surface
        self._surface = None

        # Array of booleans representing the occupation pattern
        self._occupation_pattern = None

    def make_random_occupation(self, num_adsorbed_places) -> None:
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

    def set_adsorbate() -> None:
        """Set an adsorbate onto the surface"""
        pass

    def run(self) -> None:
        """Run the Statistics Simulation"""

        # Print the help message to the console
        InputOutput.print_help()
        # Process the command line arguments
        self._io.process_arguments()

        # Read in the surface
        self._surface = self._io.read_surface()
        print(self._surface)
        num_places = self._surface.get_num_atoms()
        num_adsorbed_places = self._io.get_num_adsorbed_places()
        print(f"Number of adsorbed places: {num_adsorbed_places}")
        print(f"Number of free places:     {num_places - num_adsorbed_places}")
        print("--------------------------------")
        print(f"Total:                     {num_places}\n")

        # Generate the random occupation pattern according to
        # num_adsorbed_places
        self._occupation_pattern = np.zeros(num_places, dtype=int)
        self.make_random_occupation(num_adsorbed_places)
        print(self._occupation_pattern)

def main() -> None:
    """Main function to drive everything"""

    statistics = Statistics()
    statistics.run()

if __name__ == "__main__":
    main()
