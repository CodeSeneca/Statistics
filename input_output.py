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
        print("--surface=[string]:          Filename of the clean surface,                                      default: POSCAR_surf")
        print("--adsorbate=[string]:        Filename of the adsorbate,                                          default: POSCAR_ads")
        print("--anchor_index=[int]:        Index of the atom with whom the adsorbate adsorbs onto the surface, default: 0")
        print("--num_adsorbed_places=[int]: Number of adsorbed places:                                          default: 1\n")

    @staticmethod
    def read_poscar(filename: str) -> SimulationBox:
        """Read in a molecular structure in the VASP POSCAR format"""

        with open(filename, 'r') as poscar:
            # Read in the comment
            poscar.readline()
            # Read in the lattice constant and the lattice vectors
            lattice_constant = float( poscar.readline() )
            a1 = [ float(x) for x in poscar.readline().split() ]
            a2 = [ float(x) for x in poscar.readline().split() ]
            a3 = [ float(x) for x in poscar.readline().split() ]
            # Generate the UnitCell object
            unit_cell = UnitCell(lattice_constant, a1, a2, a3)

            # Read in the element symbols with corresponding numbers
            elements = poscar.readline().split()
            numbers = [ int(x) for x in poscar.readline().split() ]
            # Read in whether we have direct or cartesian coordinates
            unit = poscar.readline().strip()

            # Read in all atomic coordinates
            atoms = []
            for i, elem in enumerate(elements):
                for j in range(numbers[i]):
                    coords = [ float(x) for x in poscar.readline().split() ]
                    atom = Atom(elem, coords)
                    atoms.append(atom)

            # Generate the SimulationBox object
            if unit == "Direct":
                box = SimulationBox(unit_cell=unit_cell, atoms=atoms, direct=True)
            else:
                box = SimulationBox(unit_cell=unit_cell, atoms=atoms, direct=False)

        return box

    @staticmethod
    def write_poscar(box: SimulationBox, filename: str) -> None:
        """Write a Simulation box to the POSCAR file format"""

        unit_cell = box.get_unit_cell()
        atoms = box.get_atoms()

        with open(filename, 'w') as poscar:
            # Write the comment
            poscar.write("POSCAR written by program Statistics\n")

            # Write the unit cell
            poscar.write(f"{unit_cell.lattice_constant}\n")
            poscar.write(f"{unit_cell.a1[0]:15.10f} {unit_cell.a1[1]:15.10f} {unit_cell.a1[2]:15.10f}\n")
            poscar.write(f"{unit_cell.a2[0]:15.10f} {unit_cell.a2[1]:15.10f} {unit_cell.a2[2]:15.10f}\n")
            poscar.write(f"{unit_cell.a3[0]:15.10f} {unit_cell.a3[1]:15.10f} {unit_cell.a3[2]:15.10f}\n")

            # Write the element symbols and the corresponding number of atoms
            symbols = []
            numbers = []

            for atom in atoms:
                symbol = atom.symbol
                if symbol not in symbols:
                    symbols.append(symbol)

            for i, symbol in enumerate(symbols):
                if i == len(symbols) - 1:
                    poscar.write(f"{symbol}\n")
                else:
                    poscar.write(f"{symbol} ")

            for symbol in symbols:
                count = 0
                for atom in atoms:
                    if atom.symbol == symbol:
                        count += 1
                numbers.append(count)

            for i, number in enumerate(numbers):
                if i == len(numbers) - 1:
                    poscar.write(f"{number}\n")
                else:
                    poscar.write(f"{number} ")

            # Write all atoms
            if box.is_direct():
                poscar.write("Direct\n")
            else:
                poscar.write("Cartesian\n")
            for symbol in symbols:
                for atom in atoms:
                    if atom.symbol == symbol:
                        poscar.write(f"{atom.coords[0]:15.10f} {atom.coords[1]:15.10f} {atom.coords[2]:15.10f}\n")

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
