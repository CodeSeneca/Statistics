"""Module Monte Carlo"""

import sys
import random
import numpy as np
from copy import deepcopy

from ase.data import atomic_numbers, covalent_radii
from ase.geometry import get_distances

from input_output import InputOutput
from optimize import Optimize
from data import Atom

class MonteCarlo:
    """Class MonteCarlo: Driver class to run the Monte Carlo Simulation"""

    # Boltzmann factor
    kB = 8.617333262e-5    # in eV/K

    def __init__(self):
        # Object to handle IO related issues
        self._io = InputOutput(sys.argv)

        # Object for the actual Monte Carlo Simulation
        self._opt = None

        # Simulation box of the surface, of the adsorbate and the combined system
        self._surface = None
        self._adsorbate = None
        self._combined_system = None

        # Old and new array of booleans representing the occupation pattern
        self._occ1 = None
        self._occ2 = None

        # Filename of the output file
        self._out_filename = "mc.log"

        # Temperature of the simulation
        self._temperature = 293.15

    @staticmethod
    def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
        """Rotation matrix for a rotation about a unit vector axis according to Rodrigues"""

        ux, uy, uz = axis
        cross = np.array([[0.0, -uz, uy],
                          [uz, 0.0, -ux],
                          [-uy, ux, 0.0]])
        return (np.cos(angle) * np.eye(3)
                + np.sin(angle) * cross
                + (1.0 - np.cos(angle)) * np.outer(axis, axis))

    def _make_random_occupation(self, num_adsorbed_places) -> None:
        """Generate a random occupation pattern with num_adsorbed_places
        adsorbed places"""

        # Reset the occupation pattern
        self._occ1 = np.zeros(self._surface.get_num_atoms(), dtype=int)

        # List of random indeces
        random_indeces = []

        # Generate num_adsorbed_places random indeces and set
        # the corresponding occupation to 1
        for i in range(num_adsorbed_places):
            # Generate a random number as long as the random index was already
            # chosen
            while(True):
              random_index = np.random.randint(
                      low=0, high=len(self._occ1))
              if random_index not in random_indeces:
                  self._occ1[random_index] = 1
                  random_indeces.append(random_index)
                  break

    def _place_jump(self) -> None:
        """Let one adsorbate jump randomically by changing two places 0 and 1"""

        # List of all indeces with 0 and list of all indices with 1
        zeros = []
        ones = []

        for i, occ in enumerate(self._occ1):
           if occ == 0:
              zeros.append(i)
           elif occ == 1:
              ones.append(i)

        # Generate a random index from zeros and a random index from ones
        random_zero = random.choice(zeros)
        random_one = random.choice(ones)

        # Update self._occ2 by swapping random_zero and random_one
        self._occ2 = self._occ1.copy()
        self._occ2[random_zero] = 1
        self._occ2[random_one] = 0

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

    def _set_adsorbate(self, anchor_index: int, place_index: int,
                       offset: float, n_angles: int = 72,
                       tilt_max: float = 90.0, n_tilts: int = 7,
                       n_spin: int = 1, x_ok: float = 0.9,
                       z_tol: float = 0.5) -> float:

        """Set an adsorbate onto the surface. The adsorbate is rotated about
        the surface normal (through the anchor atom) and the orientation with
        the smallest overlap with all atoms already present is chosen.

        Each orientation is ranked by the score function

        sum( (dist/d_ref)**(-12) )

        with dist = the distance between each pair of atoms i,j
             dref = the reference distance being the sum of covalence radii for i,j
             **(-12) being inspired by the repulsive term in the Lennard-Jones potential

        Assumes: surface and adsorbate share the same unit cell, both Direct."""

        # Cell matrix of the clean surface and the adsorbate
        cell = self._surface.get_cell_matrix()
        cell_inv = np.linalg.inv(cell)

        ads_atoms = self._adsorbate.get_atoms()
        site = self._surface.get_atoms()[place_index]

        # Anchor position in direct coordinates (offset along a3)
        anchor_frac = site.coords + np.array([0.0, 0.0, offset])
        anchor_pos = anchor_frac @ cell

        # Adsorbate atoms relative to the anchor, in cartesian coordinates
        ads_frac = np.array([a.coords for a in ads_atoms])
        rel = (ads_frac - ads_frac[anchor_index]) @ cell
        # Covalence radii of all adsorbate atoms
        ads_rc = covalent_radii[[atomic_numbers[atom.symbol] for atom in ads_atoms]]

        # All atoms already there: surface + previously set adsorbates
        present = self._combined_system.get_atoms()
        present_pos = np.array([a.coords for a in present]) @ cell
        present_rc = covalent_radii[[atomic_numbers[a.symbol] for a in present]]

        # Atoms that move under the rotation (everything but the anchor)
        moving = np.array([i for i in range(len(ads_atoms)) if i != anchor_index])
        
        # Skip the following rotation algorithm if the adsorbate consists only of the anchor atom
        best_rel = rel
        if len(moving) > 0:
            # Surface normal
            normal = np.cross(cell[0], cell[1])
            normal /= np.linalg.norm(normal)

            # Tilt axis: any in-plane axis works, because the rotation about the
            # normal afterwards fixes the direction in which the molecule leans
            tilt_axis = cell[0] - np.dot(cell[0], normal) * normal
            tilt_axis /= np.linalg.norm(tilt_axis)

            # Reference distances for all pairs (moving atom, present atom)
            d_ref = ads_rc[moving][:, None] + present_rc[None, :]

            # Lowest height (along the normal) of the original orientation
            z_min_ref = np.min(rel @ normal)

            tilts = np.radians(np.linspace(0.0, tilt_max, n_tilts))
            spins = np.linspace(0.0, 2.0 * np.pi, n_spin, endpoint=False)
            azimuths = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)

            best_score = np.inf

            # Outer loop: tilt angle, in ascending order
            for tilt in tilts:
                r_tilt = self._rotation_matrix(tilt_axis, tilt)

                tilt_score = np.inf
                tilt_rel = None
                tilt_xmin = 0.0

                # Spin about the molecule's own axis (only meaningful if tilted)
                for spin in (spins if tilt > 0.0 else spins[:1]):
                    spun = rel @ self._rotation_matrix(normal, spin).T
                    tilted = spun @ r_tilt.T

                    # Reject orientations in which atoms sink below the anchor
                    # plane (heights do not change under the azimuthal rotation)
                    if np.min(tilted[moving] @ normal) < z_min_ref - z_tol:
                        continue

                    # Inner loop: azimuthal rotation about the surface normal
                    for phi in azimuths:
                        trial = tilted @ self._rotation_matrix(normal, phi).T

                        _, dist = get_distances(anchor_pos + trial[moving], present_pos,
                                                cell=cell, pbc=True)
                        ratio = dist / d_ref
                        score = np.sum(ratio ** -12)
                        if score < tilt_score:
                            tilt_score = score
                            tilt_rel = trial
                            tilt_xmin = ratio.min()

                if tilt_rel is None:
                    continue

                # Smallest tilt without any close contact: take it and stop
                if tilt_xmin >= x_ok:
                    best_rel = tilt_rel
                    break

                if tilt_score < best_score:
                    best_score = tilt_score
                    best_rel = tilt_rel

        # Back to direct coordinates (unverändert)
        new_frac = anchor_frac + best_rel @ cell_inv
        for atom, coords in zip(ads_atoms, new_frac):
            self._combined_system.add_atom(Atom(symbol=atom.symbol, coords=coords))

        return tilt_xmin

    def _evaluate(self, occ_pattern: np.ndarray) -> float:
        """Set the N adsorbates onto the surface according to occ_pattern
        and calculate the optimized energy for it"""

        # Reset self._combined_system to the clean surface
        self._combined_system = deepcopy(self._surface)

        # Set the adsorbates onto the surface according to self._occ1
        for i, occ in enumerate(occ_pattern):
          if occ == 1:
            min_dist = self._set_adsorbate_improved(anchor_index=self._io.get_anchor_index(), place_index=i, offset=self._io.get_offset())

        # Write the updated surface to a POSCAR file
        InputOutput.write_poscar(box=self._combined_system, filename="POSCAR")

        # Perform a geometry optimization
        self._opt.read_box("POSCAR")
        self._opt.set_pes()
        energy = self._opt.optimize(fmax=self._io.fmax)

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
        # Initialize the self._opt argument
        self._opt = Optimize(path = self._io.path,
                             head = self._io.head,
                             dispersion = self._io.dispersion,
                             dtype = self._io.dtype,
                             device = self._io.device,
                             cueq = self._io.cueq)

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
        self._make_random_occupation(num_adsorbed_places)

        # Perform self._num_steps Monte Carlo steps
        print("Entering Monte Carlo loop ...\n")

        with open(self._out_filename, 'w') as out_file:

            old_energy = self._evaluate(self._occ1)
            out_file.write(f"0    {old_energy:15.10f}\n")
            out_file.flush()
            
            for i in range(1, self._io.num_steps):
                self._place_jump()
                new_energy = self._evaluate(self._occ2)
                delta_energy = new_energy - old_energy

                out_file.write(f"{i}    {new_energy:15.10f} {delta_energy:15.10f}")
                out_file.flush()

                # The Metropolis condition
                if delta_energy <= 0 or random.random() < np.exp( -delta_energy/(MonteCarlo.kB * self._temperature) ):
                    out_file.write("    accepted \n")
                    self._occ1 = self._occ2.copy()
                    old_energy = new_energy
                else:
                    out_file.write("    not accepted \n")
