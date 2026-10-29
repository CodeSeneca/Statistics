import numpy as np

class UnitCell:
    """Class UnitCell: The simulation box"""

    def __init__(self, lattice_constant, a1: list, a2: list, a3: list):
        # Lattice constant
        self.lattice_constant = lattice_constant
        # Lattice vectors
        self.a1 = np.array(a1)
        self.a2 = np.array(a2)
        self.a3 = np.array(a3)

    def __str__(self):
        """String representation of the Unit Cell"""

        out_str = f"{self.lattice_constant}"
        out_str += f"{self.a1[0]} {self.a1[1]} {self.a1[2]}\n"
        out_str += f"{self.a2[0]} {self.a2[1]} {self.a2[2]}\n"
        out_str += f"{self.a3[0]} {self.a3[1]} {self.a3[2]}\n\n"

        return out_str

class Atom:
    """Class Atom: An atom in the sense of chemistry"""

    def __init__(self, symbol: str, coords: list):
        # Atomic symbol according to the PSE
        self.symbol = symbol
        # Coordinates (x, y, z)
        self.coords = np.array(coords)

    def distance_vector_to(self, other) -> np.array:
        """Return the distance vector from this to another Atom"""

        dist_vector = (self.coords[0] - other.coords[0])**2 + \
                      (self.coords[1] - other.coords[1])**2 + \
                      (self.coords[2] - other.coords[2])**2

        return dist_vector

    def distance_to(self, other) -> float:
        """Return the distance between this and another Atom"""

        dist = (self.coords[0] - other.coords[0])**2 + \
               (self.coords[1] - other.coords[1])**2 + \
               (self.coords[2] - other.coords[2])**2

        dist = np.sqrt(dist)

        return

    def translate(self, vec: np.array) -> None:
        """Shift the Atom according to vec"""

        self.coords = self.coords + vec

    def __str__(self):
        """String representation of an Atom"""

        return f"{self.symbol} {self.coords[0]} {self.coords[1]} {self.coords[2]}"

class SimulationBox:
    """Class SimulationBox: Box of the simulation with atoms in an unit cell"""

    def __init__(self, unit_cell: UnitCell, atoms: list):
        # The Unit Cell
        self._unit_cell = unit_cell
        # List of all atoms
        self._atoms = atoms
        # Total number of atoms, i.e., total number of free places to adsorb
        # on the surface
        self._num_atoms = len(self._atoms)

    def get_unit_cell(self):
        """Return the unit cell"""

        return self._unit_cell

    def get_atoms(self):
        """Return the list of all atoms"""

        return self._atoms

    def get_num_atoms(self) -> int:
        """Get the total number of atoms in the Simulation Box"""

        return self._num_atoms

    def translate_atoms(self, vec):
        """Shift all atoms according to vec"""

        for atom in self._atoms:
            atom.translate(vec)

    def __str__(self):
        """String representation of the Simulation Box"""

        out_str = "Simulation Box\n"
        out_str += "===============\n"
        out_str += self._unit_cell.__str__()
        for atom in self._atoms:
            out_str += atom.__str__() + "\n"

        return out_str
