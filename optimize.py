import os
from ase.io import read, write
from ase.optimize import BFGS
from mace.calculators import mace_mp

class Optimize:
    """Class Optimize: Class to perform geometry optimizations"""

    def __init__(self, path, head, dispersion, dtype, device, cueq):
        # Delete an old XDATCAR and XDATCAR_accepted file
        if os.path.exists("XDATCAR"):
            os.remove("XDATCAR")

        if os.path.exists("XDATCAR_accepted"):
            os.remove("XDATCAR_accepted")

        # Atoms object for ASE
        self._atoms = None

        # Dynamics object for ASE to perform geometry optimizations
        self._dyn = None

        # Current minimum energy
        self._min_energy = None

        # Initialize the MACE calculator
        self._calc = mace_mp(model=path, head=head, dispersion=dispersion, default_dtype=dtype, device=device, enable_cueq=cueq)

    def read_box(self, filename: str) -> None:
        """Read in the current Simulation Box from filename"""

        self._atoms = read(filename)

    def write_structure(self, filename: str, fmt: str, append:bool) -> None:
        """Write the current content of self._atoms to filename via ASE"""

        write(filename, self._atoms, format=fmt, append=append)

    def get_min_energy(self) -> float:
        """Get the current minum energy"""

        return self._min_energy

    def set_calculator(self) -> None:
        """Set the Calculator used for all geometry optimizations"""

        self._atoms.calc = self._calc

    def optimize(self, fmax) -> None:
        """Perform a geometry optimization for the current self._atoms object"""

        self._dyn = BFGS(atoms=self._atoms, logfile="-", trajectory=None)
        self._dyn.run(fmax=fmax)

        print("Writing the optimized structure to CONTCAR and to XDATCAR ...")
        self.write_structure(filename="CONTCAR", fmt="vasp", append=False)
        self.write_structure(filename="XDATCAR", fmt="vasp-xdatcar", append=True)

        # Return the final minimum energy
        self._min_energy = self._atoms.get_potential_energy()

        return self._min_energy
