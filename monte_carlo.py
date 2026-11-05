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

    def _set_adsorbate_improved(self, anchor_index: int, place_index: int,
                       offset: float, n_angles: int,
                       tilt_max: float, n_tilts,
                       n_spin, x_ok,
                       z_tol, shift_max: float,
                       n_shift_rings: int, n_shift_dirs: int,
                       score_tol: float = 1.0) -> float:

        """Set an adsorbate onto the surface. The orientation is searched over

            lateral shift (parallel to the surface, ascending length)
              tilt angle (ascending)
                spin about the molecule's axis
                  azimuth about the surface normal

        Each orientation is ranked by the score function

        sum( (dist/d_ref)**(-12) )

        with dist = the distance between each pair of atoms i,j
             dref = the reference distance being the sum of covalence radii for i,j

        All adsorbate atoms (including the anchor) enter the score. The search
        stops at the first shift/tilt without a close adsorbate-adsorbate contact
        (min ratio >= x_ok). Otherwise the smallest shift/tilt is taken whose score
        is at most score_tol times the best score.

        Every tested candidate is printed right after its test, the final
        choice is reported at the end.

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
        ads_rc = covalent_radii[[atomic_numbers[atom.symbol] for atom in ads_atoms]]

        # All atoms already there: surface + previously set adsorbates
        present = self._combined_system.get_atoms()
        present_pos = np.array([a.coords for a in present]) @ cell
        present_rc = covalent_radii[[atomic_numbers[a.symbol] for a in present]]
        n_surf = self._surface.get_num_atoms()

        # Atoms that move under rotation and tilt (everything but the anchor)
        moving = np.array([i for i in range(len(ads_atoms)) if i != anchor_index])
        single = len(moving) == 0     # adsorbate consists only of the anchor

        # Surface normal and two in-plane unit vectors
        normal = np.cross(cell[0], cell[1])
        normal /= np.linalg.norm(normal)
        tilt_axis = cell[0] - np.dot(cell[0], normal) * normal
        tilt_axis /= np.linalg.norm(tilt_axis)
        in_plane_2 = np.cross(normal, tilt_axis)

        # Lateral shifts parallel to the surface: (0, 0) first, then rings of
        # increasing radius, so small shifts are always preferred
        shifts = [np.zeros(3)]
        radii = (np.linspace(shift_max / n_shift_rings, shift_max, n_shift_rings)
                 if n_shift_rings > 0 else [])
        for r in radii:
            for theta in np.linspace(0.0, 2.0 * np.pi, n_shift_dirs, endpoint=False):
                shifts.append(r * (np.cos(theta) * tilt_axis + np.sin(theta) * in_plane_2))

        # Angle grids (a single atom needs no rotation)
        if single:
            tilts = np.array([0.0])
            spins = np.array([0.0])
            azimuths = np.array([0.0])
        else:
            tilts = np.radians(np.linspace(0.0, tilt_max, n_tilts))
            spins = np.linspace(0.0, 2.0 * np.pi, n_spin, endpoint=False)
            azimuths = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)

        # Rotation matrices about the normal for all azimuths (built once)
        rots_az = np.array([self._rotation_matrix(normal, phi) for phi in azimuths])

        # Reference distances for all pairs (adsorbate atom, present atom);
        # the anchor is included, because it moves with the shift
        d_ref = ads_rc[:, None] + present_rc[None, :]
        n_present = present_pos.shape[0]

        # Lowest height (along the normal) of the original orientation
        z_min_ref = np.min(rel @ normal)

        best_rel = rel
        best_shift = np.zeros(3)
        best_xmin = np.inf

        candidates = []          # (score, xmin, rel, shift, row number), ascending shift/tilt
        n_rows = 0               # number of printed table rows
        chosen_idx = None        # row number of the chosen candidate
        threshold = None         # score_tol * s_min (only if no early stop)
        stopped_early = False

        # Header of the table (rows are printed during the search)
        print(f"\nPlace {place_index}: placement candidates", flush=True)
        print(f"{'No.':>4} {'shift/Å':>8} {'tilt/°':>7} {'score':>10} {'min ratio':>10}", flush=True)

        # Outermost loop: lateral shift
        for shift in shifts:
            base = anchor_pos + shift

            # Loop: tilt angle, in ascending order
            for tilt in tilts:
                r_tilt = self._rotation_matrix(tilt_axis, tilt)

                tilt_score = np.inf
                tilt_rel = None
                tilt_xmin = 0.0

                # Spin about the molecule's own axis (only meaningful if tilted)
                for spin in (spins if tilt > 0.0 else spins[:1]):
                    spun = rel @ self._rotation_matrix(normal, spin).T
                    tilted = spun @ r_tilt.T

                    # Reject orientations in which atoms sink below the anchor plane
                    if not single and np.min(tilted[moving] @ normal) < z_min_ref - z_tol:
                        continue

                    # All azimuths at once: trials has the shape (n_az, n_ads, 3)
                    trials = np.einsum("aij,nj->ani", rots_az, tilted)
                    pos = (base + trials).reshape(-1, 3)

                    _, dist = get_distances(pos, present_pos, cell=cell, pbc=True)
                    ratio = dist.reshape(len(azimuths), len(ads_atoms), n_present) / d_ref
                    scores = np.sum(ratio ** -12, axis=(1, 2))

                    a = int(np.argmin(scores))
                    if scores[a] < tilt_score:
                        tilt_score = scores[a]
                        tilt_rel = trials[a]

                        # Only contacts to other adsorbates count for the stop criterion
                        ratio_ads = ratio[a][:, n_surf:]
                        tilt_xmin = ratio_ads.min() if ratio_ads.size > 0 else np.inf

                if tilt_rel is None:
                    continue

                # Print the row immediately after the test of this candidate
                n_rows += 1
                idx = n_rows
                print(f"{idx:4d} {np.linalg.norm(shift):8.2f} {np.degrees(tilt):7.1f} "
                      f"{tilt_score:10.3g} {tilt_xmin:10.2f}", flush=True)

                # Smallest shift/tilt without any close contact: take it and stop
                if tilt_xmin >= x_ok:
                    best_rel, best_shift, best_xmin = tilt_rel, shift, tilt_xmin
                    chosen_idx = idx
                    stopped_early = True
                    break

                candidates.append((tilt_score, tilt_xmin, tilt_rel, shift, idx))

            if stopped_early:
                break

        # Nothing was good enough: the smallest shift/tilt that is almost as good as the best
        if not stopped_early and candidates:
            s_min = min(c[0] for c in candidates)
            threshold = score_tol * s_min
            for sc, xm, rl, sh, idx in candidates:
                if sc <= threshold:
                    best_rel, best_shift, best_xmin = rl, sh, xm
                    chosen_idx = idx
                    break

        # Report the choice at the end
        if stopped_early:
            print(f"-> chosen: No. {chosen_idx} (stopped early, min ratio >= x_ok = {x_ok})\n",
                  flush=True)
        elif chosen_idx is not None:
            print(f"-> chosen: No. {chosen_idx} (score {sc:.3g} <= threshold "
                  f"score_tol * s_min = {threshold:.3g})\n", flush=True)
        else:
            print("-> no valid candidate, adsorbate placed in its original orientation\n",
                  flush=True)

        # Back to direct coordinates: the anchor sits at anchor_frac + shift
        new_frac = anchor_frac + (best_shift + best_rel) @ cell_inv
        for atom, coords in zip(ads_atoms, new_frac):
            self._combined_system.add_atom(Atom(symbol=atom.symbol, coords=coords))

        return best_xmin

    def _evaluate(self, occ_pattern: np.ndarray) -> float:
        """Set the N adsorbates onto the surface according to occ_pattern
        and calculate the optimized energy for it"""

        # Reset self._combined_system to the clean surface
        self._combined_system = deepcopy(self._surface)

        # Set the adsorbates onto the surface according to self._occ1
        for i, occ in enumerate(occ_pattern):
          if occ == 1:
            min_dist = self._set_adsorbate_improved(
                anchor_index=self._io.get_anchor_index(),
                place_index=i, offset=self._io.get_offset(),
                n_tilts=self._io.num_tilt_angles,
                tilt_max=self._io.max_tilt_angle,
                n_angles=self._io.num_azimuth_angles,
                n_spin=self._io.num_spin_angles,
                x_ok=self._io.min_dist_break,
                z_tol=self._io.z_tol,
                shift_max=self._io.shift_max,
                n_shift_rings=self._io.num_shift_rings,
                n_shift_dirs=self._io.num_shift_dirs)

        # Write the updated surface to a POSCAR file
        print("The starting structure is written to POSCAR ...")
        InputOutput.write_poscar(box=self._combined_system, filename="POSCAR")

        # Perform a geometry optimization
        self._opt.read_box("POSCAR")
        self._opt.set_calculator()
        energy = self._opt.optimize(fmax=self._io.fmax)

        return energy, min_dist

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
        print("Entering Monte Carlo loop ...")
        print("-------------------------------------------")
        print(f"Temperature: {self._io.temp} K")
        print(f"Total number of MC steps: {self._io.num_steps}\n")

        with open(self._out_filename, 'w') as out_file:

            print("Step no. 0")
            print("----------------------------------------------")
            print("Setting the adsorbates onto the clean surface ...")
            old_energy, dist_min_start = self._evaluate(self._occ1)
            out_file.write(f"0    {old_energy:15.10f}\n")
            out_file.flush()
            print("----------------------------------------------")
            
            for i in range(1, self._io.num_steps):
                print(f"Step no. {i}")
                self._place_jump()
                print("----------------------------------------------")
                print("Setting the adsorbates onto the clean surface ...")
                new_energy, dist_min_start = self._evaluate(self._occ2)
                delta_energy = new_energy - old_energy
                print("----------------------------------------------\n")

                # The Metropolis condition
                if delta_energy <= 0 or random.random() < np.exp( -delta_energy/(MonteCarlo.kB * self._io.temp) ):
                    self._occ1 = self._occ2.copy()
                    old_energy = new_energy
                    out_file.write(f"{i}    {new_energy:15.10f} {delta_energy:15.10f} {old_energy:15.10f} {dist_min_start:5.2f} accepted\n")
                    out_file.flush()
                    # Write only the accepted structures to XDATCAR_accepted
                    self._opt.write_structure(filename="XDATCAR_accepted", fmt="vasp-xdatcar", append=True)
                else:
                    out_file.write(f"{i}    {new_energy:15.10f} {delta_energy:15.10f} {old_energy:15.10f} {dist_min_start:5.2f} not accepted\n")
                    out_file.flush()
