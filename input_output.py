from data import UnitCell, Atom, SimulationBox

class InputOutput:
    """Class InputOutput: Handling IO related issues"""

    def __init__(self, arguments):
        # Command line arguments
        self._arguments = arguments
        # Number of command line arguments
        self._arg_number = len(self._arguments)

        # Filename of the surface -> default: POSCAR_surf
        self._surface_filename = "POSCAR_surf"
        # Filename of the adsorbate -> default: POSCAR_ads
        self._adsorbate_filename = "POSCAR_ads"

        # Number of adsorbed places -> default: 1
        self._num_adsorbed_places = 1

        # Index of the anchor atom -> default: 0
        self._anchor_index = 0

    @staticmethod
    def print_help() -> None:
        """Print a help message to the console"""

        print("\nProgram Statistics")
        print("******************\n")
        print("Author: Maximilian Bechtel, <maxi.bechtel@fau.de>")
        print("Lehrstuhl fuer Theoretische Chemie, Friedrich-Alexander-Universitaet Erlangen-Nuernberg\n")
        print("Usage: python main.py [options]\n")
        print("--surface=[string]:          Filename of the clean surface, default: POSCAR_surf")
        print("--adsorbate=[string]:        Filename of the adsorbate,     default: POSCAR_ads")
        print("--num_adsorbed_places=[int]: Number of adsorbed places:     default: 1\n")

    @staticmethod
    def read_poscar(filename: str) -> SimulationBox:
        """Read in a molecular structure in the VASP POSCAR format"""

        with open(filename, 'r') as poscar:
            # Read in the comment
            poscar.readline()
            # Read in the lattice constant and the lattice vectors
            lattice_constant = poscar.readline()
            a1 = [ float(x) for x in poscar.readline().split() ]
            a2 = [ float(x) for x in poscar.readline().split() ]
            a3 = [ float(x) for x in poscar.readline().split() ]
            # Generate the UnitCell object
            unit_cell = UnitCell(lattice_constant, a1, a2, a3)

            # Read in the element symbols with corresponding numbers
            elements = poscar.readline().split()
            numbers = [ int(x) for x in poscar.readline().split() ]
            # Read in whether we have direct or cartesian coordinates
            unit = poscar.readline()

            # Read in all atomic coordinates
            atoms = []
            for i, elem in enumerate(elements):
                for j in range(numbers[i]):
                    coords = [ float(x) for x in poscar.readline().split() ]
                    atom = Atom(elem, coords)
                    atoms.append(atom)

            # Generate the SimulationBox object
            box = SimulationBox(unit_cell, atoms)

        return box

    def get_num_adsorbed_places(self):
        """Return the value of num_adsorbed_places"""

        return self._num_adsorbed_places

    def get_anchor_index(self):
        """Return the anchor index"""

        return self._anchor_index

    def process_arguments(self) -> None:
        """Process command line arguments"""

        for arg in self._arguments:
            arg_split = arg.split("=")
            if arg_split[0] == "--surface":
                self._surface_filename = arg_split[1]
            elif arg_split[0] == "--adsorbate":
                self._adsorbate_filename = arg_split[1]
            elif arg_split[0] == "--num_adsorbed_places":
                self._num_adsorbed_places = int(arg_split[1])
            elif arg_split[0] == "--anchor_index":
                self._anchor_index = int(arg_split[1])

    def read_surface(self) -> SimulationBox:
        """Read in the surface"""

        print(f"Reading in the clean surface from {self._surface_filename} ...")
        box = InputOutput.read_poscar(self._surface_filename)
        return box

    def read_adsorbate(self) -> SimulationBox:
        """Read in the adsorbate"""

        print(f"Reading in the adsorbate from {self._adsorbate_filename} ...\n")
        box = InputOutput.read_poscar(self._adsorbate_filename)
        return box
