from ase.io import read, write
from ase.optimize import BFGS
from mace.calculators import mace_mp

class MonteCarlo:
    """Class MonteCarlo: Class to perform geometry optimizations"""

    def __init__(self, path, head, dispersion, dtype, device, cueq, fmax):
        # Atoms object for ASE
        self._atoms = None

        # Dynamics object for ASE to perform geometry optimizations
        self._dyn = None

        # Path to the MACE model used within all calculations
        self._path = path
        # Head of the MACE model
        self._head = head
        # If the D3 dispersion correction should be added via torch-dftd
        self._dispersion = dispersion
        # Float type
        self._dtype = dtype
        # Cuda or CPU
        self._device = device
        # Whether the CuEq acceleration should be used
        self._cueq = cueq
        # Convergence criterion for the geometry optimization
        self._fmax = fmax

    def read_box(self, filename: str) -> None:
        """Read in the current Simulation Box from filename"""

        self._atoms = read(filename)

    def set_pes(self) -> None:
        """Set the Calculator used for all calculations"""

        calc = mace_mp(model=self._path, head=self._head, dispersion=self._dispersion, default_dtype=self._dtype, device=self._device, enable_cueq=self._cueq)
        self._atoms.calc = calc

    def optimize(self) -> None:
        """Perform a geometry optimization for the current self._atoms object"""

        self._dyn = BFGS(atoms=self._atoms)
        self._dyn.run(fmax=self._fmax)

        write("CONTCAR", self._atoms, format="vasp")
        min_energy = self._atoms.get_potential_energy()

        return min_energy