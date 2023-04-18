#%%

import sys
import ase


from AAmatcher.matching import *

from AAmatcher.xyz2res.deploy import xyz2res
from pathlib import Path

#%%


'''
Class for storing a dataset (positions, energies, forces/gradients) for a molecule that can be described by a pdb file.
Can be initialized from positions and elements alone, the residues and membership of constituent atoms are deduced in this case.
Given a sequence and assuming that the atoms are already ordered by residue, can also be initialised more cheaply from a gaussian log file.

This class can also be used to create PDB files from xyz coordinates and elements alone. See examples/from_xyz.py for a tutorial.

Note that the order of the stored element/xyz/gradient array may deviate from the element/xyz/gradient array used for initialization to correspond to the order used in the pdb file. To add data manually after initialization, apply the permutation stored by the class to your data beforehand.
...

Attributes
----------
xyz: np.ndarray
    Array of shape (N_conf x N_atoms x 3) containing the atom positions. The order along the N_atoms axis is determined by the order of the elements member variable. This ordering can deviate from the ordering in the initialization arguments.

energies: np.ndarray
    Array of shape (N_conf) containing the atom positions.

gradients: np.ndarray
    Array of shape (N_conf x N_atoms x 3) containing the gradient of the energy wrt the positions (i.e. the negative forces). The order along the N_atoms axis is determined by the order of the elements member variable. This ordering can deviate from the ordering in the initialization arguments.

elements: np.ndarray
    Array of shape (N_atoms) containing the atomic numbers of the elements. The order can deviate from the ordering in the initialization arguments but corresponds to the stored values for xyz and the gradients.

pdb: list
    List of strings representing the pdb-file of some conformation.
    This stores information on connectivity and atom types of the molecule, not the on the conformations.

sequence : list
    List of strings describing the sequence

residues: np.ndarray
    List of strings containing the residues by atom in the order of the element member variable, only set if initialized by xyz.

residues_numbers: np.ndarray
    List of integers starting at zero and ending at len(sequence), grouping atoms to residues, only set if initialized by xyz.

permutation: np.ndarray
    Array of integers describing an atom permutation representing the change of order of the input atom order (which is done to be consistent with the pdb). This can be used to add own xyz or gradient data after initialization.


Methods
-------
__init__():
    Sets every member to None. Use the from_... classmethods for initialization.

PDBMolecule.from_gaussian_log(logfile, cap:bool=True, rtp_path=DEFAULT_RTP, sequence:list=None):
    Use a gaussian logfile for initialization.

PDBMolecule.from_xyz(cls, xyz:np.ndarray, elements:np.ndarray, energies:np.ndarray=None, gradients:np.ndarray=None, rtp_path=DEFAULT_RTP, residues:list=None, res_numbers:list=None):
    Use an xyz array of shape (N_confsxN_atomsx3) and an element array of shape (N_atoms) for initialization. The atom order in which xyz and element are stored may differ from that of those used for initilization (See description of the xyz member).
    Currently only works for peptides that do not contain HIS or GLU
'''

class PDBMolecule:
    DEFAULT_RTP = Path(__file__).parent.parent/Path("example/amber99sb-star-ildnp.ff/aminoacids.rtp")

    # use the classmethods to construct the object!
    def __init__(self):
        self.xyz = None
        self.energies = None
        self.gradients = None
        self.elements = None

        self.pdb = None
        self.sequence = None
        self.residues = None
        self.residue_numbers = None

        self.permutation = None

    """
    Creates a pdb file at the given path.
    """
    def write_pdb(self, pdb_path="my_pdb.pdb"):
        with open(pdb_path, "w") as f:
            f.writelines(self.pdb)
        
    """
    For the state indexed by idx, returns a tuple of (xyz, energy, gradients), where xyz and gradient are arrays of shape (N_atoms, 3) and energy is a scalar value.
    """
    def __getitem__(self, idx):
        if (not self.energies is None) and (not self.gradients is None):
            return self.xyz[idx], self.energies[idx], self.gradients[idx]
        
        if self.energies is None and not self.gradients is None:
            return self.xyz[idx], None, self.gradients[idx]

        if not self.energies is None and self.gradients is None:
            return self.xyz[idx], self.energies[idx], None

        return self.xyz[idx], None, None
    
    """
    Returns the number of conformations.
    """
    def __len__(self):
        return len(self.xyz)

    """
    Internal helper function
    """
    def to_dict(self):
        arrays = {}
        arrays["elements"] = self.elements
        arrays["xyz"] = self.xyz
        arrays["pdb"] = np.array(self.pdb)
        if not self.energies is None:
            arrays["energies"] = self.energies
        if not self.gradients is None:
            arrays["gradients"] = self.gradients
        arrays["permutation"] = self.permutation
        arrays["sequence"] = np.array(self.sequence)
        return arrays

    """
    Internal helper function
    """
    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.elements = d["elements"]
        obj.xyz = d["xyz"]
        obj.pdb = d["pdb"].tolist()
        if "energies" in d.keys():
            obj.energies = d["energies"]
        if "gradients" in d.keys():
            obj.gradients = d["gradients"]
        obj.permutation = d["permutation"]
        obj.sequence = d["sequence"].tolist()
        return obj
    
    """
    Creates an npz file with the uncompressed dataset.
    """
    def save(self, filename):
        with open(filename, "bw") as f:
            np.savez(f, **self.to_dict())

    """
    Creates an npz file with the compressed dataset.
    """
    def compress(self, filename):
        with open(filename, "bw") as f:
            np.savez_compressed(f, **self.to_dict())

    """
    Initializes the object from an npz file.
    """
    @classmethod
    def load(cls, filename):
        with open(filename, "br") as f:
            return cls.from_dict(np.load(f))

    """
    Returns an ASE trajectory of the states given by idxs. Default is to return a trajectory containing all states.
    """
    def to_ase(self, idxs=None):
        pass

    """
    Returns a heterogeneous dgl graph containing the connectivity of the pdb and positions, energies and gradients stored in the respective node types of the states given by idxs. Default is to store all states.
    """
    def to_dgl(self, idx=None):
        pass


    """
    Use a gaussian logfile for initialization. Returns the initialized object.
    Parameters
    ----------
    logfile: str/pathlib.Path
        Path to the gaussian log file
    
        
    """
    @classmethod
    def from_gaussian_log(cls, logfile, cap:bool=True, rtp_path=None, sequence:list=None, logging:bool=False):
        if rtp_path is None:
            rtp_path = PDBMolecule.DEFAULT_RTP

        if type(rtp_path) == str:
            rtp_path = Path(rtp_path)

        obj = cls()

        AAs_reference = read_rtp(rtp_path)

        if type(logfile) == str:
            logfile = Path(logfile)

        if sequence is None:
            sequence = seq_from_filename(logfile, AAs_reference, cap)
        obj.sequence = sequence

        mol, trajectory = read_g09(logfile, obj.sequence, AAs_reference, log=logging)

        atom_order = match_mol(mol, AAs_reference, obj.sequence, log=logging)
        obj.permutation = np.array(atom_order)

        # write single pdb
        conf = trajectory[0]
        obj.pdb, elements = PDBMolecule.pdb_from_ase(ase_conformation=conf, sequence=obj.sequence, AAs_reference=AAs_reference,atom_order=atom_order)

        obj.elements = np.array(elements)

        # write xyz and energies
        energies = []
        positions = []
        gradients = []

        # NOTE: take higher precision for energies since we are interested in differences that are tiny compared to the absolute magnitude
        for step in trajectory:
            try:
                energy = step.get_total_energy()
            except PropertyNotImplementedError:
                energy = float("nan")
            energies.append(energy)

            # NOTE: implement forces

            pos = step.positions[atom_order] # array of shape n_atoms x 3 in the correct order
            positions.append(pos)
        
        obj.energies = np.array(energies)
        obj.xyz = np.array(positions)
        return obj

    @classmethod
    def from_xyz(cls, xyz:np.ndarray, elements:np.ndarray, energies:np.ndarray=None, gradients:np.ndarray=None, rtp_path=None, residues:list=None, res_numbers:list=None, logging:bool=False):

        for l, to_be_checked, name in [(1,energies, "energies"), (3,gradients,"gradients")]:
            if not to_be_checked is None:
                if xyz.shape[0] != to_be_checked.shape[0]:
                    raise RuntimeError(f"Number of conformations must be consistent between xyz and {name}, shapes are {xyz.shape} and {to_be_checked.shape}")
                if len(to_be_checked.shape) != l:
                    raise RuntimeError(f"{name} must be a {l}-dimensional array but is of shape {to_be_checked.shape}")
                
        if not gradients is None:
            if xyz.shape != gradients.shape:
                raise RuntimeError(f"Gradients and positions must have the same shape. Shapes are {gradients.shape} and {xyz.shape}.")
            
        

        if rtp_path is None:
            rtp_path = PDBMolecule.DEFAULT_RTP
        if type(rtp_path) == str:
            rtp_path = Path(rtp_path)

        obj = cls()
        
        # get the residues:
        if residues is None or res_numbers is None:
            residues, res_numbers = xyz2res(xyz[0], elements)

        # find a permutation that orders the residues:
        l = [(res_numbers[i], i) for i in range(len(res_numbers))]
        l.sort()
        perm = [l[i][1] for i in range(len(l))]

        xyz = xyz[:,perm]
        elements = elements[perm]

        if not energies is None:
            obj.energies = energies

        # make the sequence:
        residues = np.array(residues)[perm]
        res_numbers = [entry[0] for entry in l]

        # match_mol will leave this invariant
        obj.res_numbers = np.array(res_numbers)
        obj.residues = residues
        
        obj.sequence = []
        num_before = -1
        for j,i in enumerate(res_numbers):
            if i != num_before:
                num_before = i
                obj.sequence.append(residues[j])

        # preparation for getting the pdb file
        AAs_reference = read_rtp(rtp_path)

        # generate ase molecule
        n = xyz[0].shape[0]
        pos = xyz[0]
        ase_mol = ase.Atoms(f"N{n}")
        ase_mol.set_positions(pos)
        ase_mol.set_atomic_numbers(elements)

        mol, _ = read_g09(None, obj.sequence, AAs_reference, trajectory_in=[ase_mol], log=logging)

        atom_order = match_mol(mol, AAs_reference, obj.sequence, log=logging)

        # concatenate the permutations:
        obj.permutation = np.array(perm)[atom_order]

        obj.pdb, _ = PDBMolecule.pdb_from_ase(ase_conformation=ase_mol, sequence=obj.sequence, AAs_reference=AAs_reference, atom_order=atom_order)

        obj.elements = elements[atom_order]
        obj.xyz = xyz[:,atom_order]

        # NOTE set residue and res numbers in correct order
        
        if not gradients is None:
            obj.gradients = gradients[:,atom_order]

        return obj


    """
    Helper method for internal use.
    """
    @staticmethod
    def pdb_from_ase(ase_conformation, sequence, AAs_reference, atom_order):
        pdb_lines = []
        elements = []

        # copied from AAmatcher.write_trjtopdb
        AA_pos = 0
        elements = ase_conformation.get_atomic_numbers()[atom_order]
        res_len = len(AAs_reference[sequence[AA_pos]]['atoms'])
        for j,k in enumerate(atom_order):
            if j >= res_len:
                AA_pos += 1
                res_len += len(AAs_reference[sequence[AA_pos]]['atoms'])
            atomname = AAs_reference[sequence[AA_pos]]['atoms'][j-res_len][0]

            pdb_lines.append('{0}  {1:>5d} {2:^4s} {3:<3s} {4:1s}{5:>4d}    {6:8.3f}{7:8.3f}{8:8.3f}{9:6.2f}{10:6.2f}          {11:>2s}\n'.format('ATOM',j,atomname ,sequence[AA_pos].split(sep="_")[0],'A',AA_pos+1,*ase_conformation.positions[k],1.00,0.00,atomname[0]))

        pdb_lines.append("END")
        return pdb_lines, elements

# %%
# some small testing:
if __name__ == "__main__":
    from pathlib import Path

    #%%
    logfile = "./../example/g09_dataset/Ala_nat_opt.log"
    mol_log = PDBMolecule.from_gaussian_log(Path(logfile), logging=True)
    # %%
    xyz = mol_log.xyz
    elements = mol_log.elements

    #%%
    mol_xyz = PDBMolecule.from_xyz(xyz, elements)
    # %%
    # now with permuted atoms
    perm = np.arange(xyz.shape[1])
    np.random.shuffle(perm)
    # %%
    mol_xyz = PDBMolecule.from_xyz(xyz[:,perm], elements[perm])
    mol_xyz.write_pdb()

    #%%
    mol_xyz.save("tmp.npz")
    loaded = PDBMolecule.load("tmp.npz")
    # loading and saving works:
    assert np.all(loaded.xyz == mol_xyz.xyz)
    # keeping track of the permutation works
    assert np.all(loaded.elements == mol_log.elements[perm][mol_xyz.permutation])
    #%%
    mol_xyz.compress("tmp.npz")
    loaded = PDBMolecule.load("tmp.npz")
    assert np.all(loaded.xyz == mol_xyz.xyz)

# %%
