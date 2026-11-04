import os
from ase.io import read, write
from ase.optimize import BFGS
from mace.calculators import mace_mp

class Optimize:
    """Class Optimize: Class to perform geometry optimizations"""

    def __init__(self, path, head, dispersion, dtype, device, cueq):
        # Delete an old XDATCAR file
        if os.path.exists("XDATCAR"):
            os.remove("XDATCAR")

        # Atoms object for ASE
        self._atoms = None

        # Dynamics object for ASE to perform geometry optimizations
        self._dyn = None

        # Initialize the MACE calculator
        self._calc = mace_mp(model=path, head=head, dispersion=dispersion, default_dtype=dtype, device=device, enable_cueq=cueq)

    def read_box(self, filename: str) -> None:
        """Read in the current Simulation Box from filename"""

        self._atoms = read(filename)

    def set_pes(self) -> None:
        """Set the Calculator used for all calculations"""

        self._atoms.calc = self._calc

    def optimize(self, fmax) -> None:
        """Perform a geometry optimization for the current self._atoms object"""

        self._dyn = BFGS(atoms=self._atoms, logfile="-", trajectory=None)
        self._dyn.run(fmax=fmax)

        write("CONTCAR", self._atoms, format="vasp")
        write("XDATCAR", self._atoms, format="vasp-xdatcar", append=True)
        min_energy = self._atoms.get_potential_energy()

        return min_energy
