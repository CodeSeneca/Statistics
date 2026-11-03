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

        # Offset of the adsorbate from the surface (in direct coordinates) -> default: 0.05
        self._offset = 0.05

        # Path to the MACE model used within all calculations
        self.path = None
        # Head of the MACE model
        self.head = "omol"
        # If the D3 dispersion correction should be added via torch-dftd
        self.dispersion = False
        # Float type
        self.dtype = "float64"
        # Cuda or CPU
        self.device = "cpu"
        # Whether the CuEq acceleration should be used
        self.cueq = False

        # Convergence criterion for the geometry optimizations, default: 0.02
        self.fmax = 0.02

        # Total number of Monte Carlo steps, default: 10
        self.num_steps = 10

    @staticmethod
    def print_help() -> None:
        """Print a help message to the console"""

        print("\nProgram Statistics")
        print("******************\n")
        print("Author: Maximilian Bechtel, <maxi.bechtel@fau.de>")
        print("Lehrstuhl fuer Theoretische Chemie, Friedrich-Alexander-Universitaet Erlangen-Nuernberg\n")
        print("Usage: python main.py [options]\n")
        print("--surface=[string]:            Filename of the clean surface,                                                           default: POSCAR_surf")
        print("--adsorbate=[string]:          Filename of the adsorbate,                                                               default: POSCAR_ads")
        print("--anchor_index=[int]:          Index of the atom with whom the adsorbate adsorbs onto the surface,                      default: 0")
        print("--num_adsorbed_places=[int]:   Number of adsorbed places:                                                               default: 1")
        print("--offset=[float]:              Distance of the adsorbate to the surface (in direct coordinates),                        default: 0.05")
        print("--path=[String]:               Path to the MACE model")
        print("--head=[String]:               Head of the MACE model for the MH-1 model,                                               default: omol")
        print("--dispersion=:                 Add a dispersion correction to the MACE energy,                                          default: not activated")
        print("--dtype=[float32 or float64]:  Float64 is slower but more accurate,                                                     default: float64")
        print("--device=[cuda or cpu]                                                                                                  default: cpu")
        print("--cueq:                        Activate the cuEquivariance acceleration, only possible for --device=cuda,               default: not activated")
        print("--fmax=[float]:                Convergence criterion for the geometry optimizations,                                    default: 0.02")
        print("--num_steps=[int]:             Total number of Monte Carlo steps from 0 to num_steps:                                   default: 10")

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

    def get_offset(self):
        """Return the offset"""

        return self._offset

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
            elif arg_split[0] == "--offset":
                self._offset = float(arg_split[1])
            elif arg_split[0] == "--path":
                self.path = arg_split[1]
            elif arg_split[0] == "--head":
                self.head = arg_split[1]
            elif arg_split[0] == "--dispersion":
                self.dispersion = True
            elif arg_split[0] == "--dtype":
                self.dtype = arg_split[1]
            elif arg_split[0] == "--device":
                self.device = arg_split[1]
            elif arg_split[0] == "--cueq":
                self.cueq = True
            elif arg_split[0] == "--fmax":
                self.fmax = float(arg_split[1])
            elif arg_split[0] == "--num_steps":
                self.num_steps = int(arg_split[1])

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
