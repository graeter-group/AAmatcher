#%%
import os
import ase

from PDBData.xyz2res.deploy import xyz2res
from pathlib import Path
import tempfile
from openmm.app import PDBFile, ForceField
from typing import List, Union, Tuple
from dgl import graph, add_reverse_edges
from openmm.unit import angstrom
import ase
import numpy as np
import PDBData.matching
import torch
from PDBData.utils import utilities, utils, draw_mol
from PDBData import parametrize

# NOTE: BUILD FILTER FOR RESIDUES THAT ARE NOT IN RES LIST OF THE XYZ MATCHING
#%%


class PDBMolecule:
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
        Array of shape (N_conf) containing the energies in kcal/mol.

    gradients: np.ndarray
        Array of shape (N_conf x N_atoms x 3) containing the gradient of the energy wrt the positions (i.e. the negative forces). The order along the N_atoms axis is determined by the order of the elements member variable. This ordering can deviate from the ordering in the initialization arguments. Unit is kcal/mol/Angstrom.

    elements: np.ndarray
        Array of shape (N_atoms) containing the atomic numbers of the elements. The order can deviate from the ordering in the initialization arguments but corresponds to the stored values for xyz and the gradients.

    pdb: list
        List of strings representing the pdb-file of some conformation.
        This stores information on connectivity and atom types of the molecule, not the on the conformations.

    sequence: str
        String of threeletter amino acid codes representing the sequence of the molecule. Separated by '-', capitalized and including caps.

    residues: np.ndarray
        List of strings containing the residues by atom in the order of the element member variable, only set if initialized by xyz.

    residues_numbers: np.ndarray
        List of integers starting at zero and ending at len(sequence), grouping atoms to residues, only set if initialized by xyz.

    name: str
        Name of the molecule, by default, this is the sequence. However this is only unique up to charges and HIS tautomers.

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
        The positions must be given in angstrom.
        Currently only works for:
        peptides that do not contain TYR, PHE.
        and R,K: positive charge, D,E: negative charge, H: neutral - HIE


    Usage
    -----
    >>> from PDBData.PDBMolecule import PDBMolecule
    mol = PDBMolecule.from_pdb("my_pdbpath.pdb", xyz=xyz, energies=energies, gradients=gradients)
    mol.bond_check()
    mol.parametrize()
    g = mol.to_dgl()
    '''

    DEFAULT_RTP = Path(__file__).parent.parent/Path("amber99sb-star-ildnp.ff/aminoacids.rtp")

    
    def __init__(self)->None:
        """
        Initializes everything to None. Use the classmethods to construct the object!
        """
        self.xyz = None
        self.energies = None
        self.gradients = None
        self.elements = None

        self.pdb = None
        self.sequence = None
        self.residues = None
        self.residue_numbers = None
        self.name = None # optional

        self.permutation = None

        self.graph_data = {} # dictionary holding all of the data in the graph


    def write_pdb(self, pdb_path:Union[str, Path]="my_pdb.pdb")->None:
        """
        Creates a pdb file at the given path.
        """
        with open(pdb_path, "w") as f:
            f.writelines(self.pdb)


    def __getitem__(self, idx:int)->Tuple[np.ndarray, Union[np.ndarray, None], Union[np.ndarray, None]]:        
        """
        For the state indexed by idx, returns a tuple of (xyz, energy, gradients), where xyz and gradient are arrays of shape (N_atoms, 3) and energy is a scalar value.
        """
        if (not self.energies is None) and (not self.gradients is None):
            return self.xyz[idx], self.energies[idx], self.gradients[idx]
        
        if self.energies is None and not self.gradients is None:
            return self.xyz[idx], None, self.gradients[idx]

        if not self.energies is None and self.gradients is None:
            return self.xyz[idx], self.energies[idx], None

        return self.xyz[idx], None, None
    

    def __len__(self)->int:
        """
        Returns the number of conformations.
        """
        return len(self.xyz)


    def to_dict(self)->dict:
        """
        Create a dictionary holding arrays of the internal data.
        """
        arrays = {}
        arrays["elements"] = self.elements
        arrays["xyz"] = self.xyz
        arrays["pdb"] = np.array(self.pdb)
        if not self.energies is None:
            arrays["energies"] = self.energies
        if not self.gradients is None:
            arrays["gradients"] = self.gradients
        arrays["permutation"] = self.permutation
        arrays["sequence"] = np.array((self.sequence))

        for level in self.graph_data.keys():
            for feat in self.graph_data[level].keys():
                key = level + " " + feat
                arrays[key] = self.graph_data[level][feat]

        return arrays
    

    @classmethod
    def from_dict(cls, d):
        """
        Internal helper function
        """
        obj = cls()
        obj.elements = d["elements"]
        obj.xyz = d["xyz"]
        obj.pdb = d["pdb"].tolist()
        if "energies" in d.keys():
            obj.energies = d["energies"]
        if "gradients" in d.keys():
            obj.gradients = d["gradients"]
        obj.permutation = d["permutation"]
        obj.sequence = str(d["sequence"]) # assume that the entry is given as np array with one element

        for key in d.keys():
            if " " in key:
                level, feat = key.split(" ")
                if not level in obj.graph_data.keys():
                    obj.graph_data[level] = {}
                obj.graph_data[level][feat] = d[key]

        return obj
    

    def save(self, filename)->None:
        """
        Creates an npz file with the uncompressed dataset.
        """
        with open(filename, "bw") as f:
            np.savez(f, **self.to_dict())


    def compress(self, filename)->None:
        """
        Creates an npz file with the compressed dataset.
        """
        with open(filename, "bw") as f:
            np.savez_compressed(f, **self.to_dict())


    @classmethod
    def load(cls, filename):
        """
        Initializes the object from an npz file.
        """
        with open(filename, "br") as f:
            return cls.from_dict(np.load(f))


    def to_ase(self, idxs=None)->List[ase.Atoms]:
        """
        Returns an ASE trajectory of the states given by idxs. Default is to return a trajectory containing all states.
        """
        if idxs is None:
            idxs = np.arange(len(self))

        def get_ase(xyz, elements):
            ase_mol = ase.Atoms(f"N{len(elements)}")
            ase_mol.set_atomic_numbers(elements)
            ase_mol.set_positions(xyz)
            return ase_mol
        
        traj = [get_ase(self.xyz[idx], self.elements) for idx in idxs]
        return traj


    def get_bonds(self, from_pdb:bool=True)->List[Tuple[int, int]]:
        """
        Returns a list of tuples describing the bonds between the atoms where the indices correspond to the order of self.elements, xyz, etc. .
        """
        if from_pdb:
            openmm_mol = self.to_openmm()
            bonds = [(b[0].index, b[1].index) for b in openmm_mol.topology.bonds()]
            min_bond = np.array(bonds).flatten().min()
            if min_bond != 0:
                bonds = [(b[0]-min_bond, b[1]-min_bond) for b in bonds]
        else:
            bonds = self.get_ase_bonds(idxs=[0])[0]
        return bonds

    def to_graph(self, from_pdb:bool=True)->graph:
        """
        Returns a homogeneous dgl graph with connectivity induced by the pdb.
        """
        bonds = self.get_bonds(from_pdb=from_pdb)
        # reshape the list from n_bx2 to 2xn_b:
        bonds = [[b[i] for b in bonds] for i in [0,1] ]
        # create an undirected graph:
        g = graph(tuple(bonds))
        g = add_reverse_edges(g)
        # torch onehot encoding:
        g.ndata["atomic_number"] = torch.nn.functional.one_hot(torch.tensor(self.elements))
        return g

    def draw(self):
        """
        Draws the molecular graph using networx.
        """
        g = self.to_graph()
        draw_mol(g, show=True)

    def to_dgl(self, graph_data:bool=True)->graph:
        """
        Returns a heterogeneous dgl graph of n-body tuples for forcefield parameter prediction.
        The data stored in the class is keyed by: ("n1","xyz"), ("g","u_qm") ("n1","grad_qm")
        """
        with tempfile.TemporaryDirectory() as tmp:
            pdbpath = os.path.join(tmp, 'pep.pdb')
            with open(pdbpath, "w") as pdb_file:
                pdb_file.writelines([line for line in self.pdb])
            g = utils.pdb2dgl(pdbpath)

        # write data in the graph:
        if graph_data:
            for level in self.graph_data.keys():
                for feat in self.graph_data[level].keys():
                    g.nodes[level].data[feat] = torch.tensor(self.graph_data[level][feat])

            g.nodes["n1"].data["xyz"] = torch.tensor(self.xyz, dtype=torch.float32).transpose(0,1)
            if not self.gradients is None:
                g.nodes["n1"].data["grad_qm"] = torch.tensor(self.gradients, dtype=torch.float32).transpose(0,1)
            if not self.energies is None:
                g.nodes["g"].data["u_qm"] = torch.tensor(self.energies, dtype=torch.float32).unsqueeze(0)
        return g


    def to_openff(self):
        """
        Returns an openff molecule.
        """
        with tempfile.TemporaryDirectory() as tmp:
            pdbpath = os.path.join(tmp, 'pep.pdb')
            with open(pdbpath, "w") as pdb_file:
                pdb_file.writelines([line for line in self.pdb])
            mol = utils.pdb2openff(pdbpath)
        return mol
    

    def to_openff_graph(self):
        """
        Returns an openff molecule for representing the graph structure of the molecule, without chemical details such as bond order, formal charge and stereochemistry.
        """
        openmm_mol = self.to_openmm()
        mol = utils.openmm2openff_graph(openmm_mol.topology)
        return mol


    def to_openmm(self)->PDBFile:
        """
        Returns an openmm molecule containing the pdb member variable.
        """
        with tempfile.TemporaryDirectory() as tmp:
            pdbpath = os.path.join(tmp, 'pep.pdb')
            with open(pdbpath, "w") as pdb_file:
                pdb_file.writelines([line for line in self.pdb])
            openmm_pdb = PDBFile(pdbpath)
        openmm_pdb = utils.replace_h23_to_h12(openmm_pdb)
        return openmm_pdb


    def parametrize(self, forcefield:ForceField=ForceField('amber99sbildn.xml'), suffix:str="_amber99sbildn", get_charges=None, charge_suffix:str=None, ref_suffix:str="_ref", openff_charge_flag=False)->None:
        """
        Stores the forcefield parameters and energies/gradients in the graph_data dictionary.
        get_charges is a function that takes a topology and returns a list of charges as openmm Quantities in the order of the atoms in topology.
        Also writes reference data (such as the energy/gradients minus nonbonded and which torsion coefficients are zero) to the graph.
        if not openffmol is None, get_charge can also take an openffmolecule instead.
        If the charge model taket an openff molecule as input, set openff_charge_flag to True.
        """
        g = self.to_dgl(graph_data=False)
        g.nodes["n1"].data["xyz"] = torch.tensor(self.xyz, dtype=torch.float32,).transpose(0,1)

        openffmol = None
        if openff_charge_flag:
            openffmol = self.to_openff()

        g = parametrize.parametrize_amber(g, forcefield=forcefield, suffix=suffix, get_charges=get_charges, charge_suffix=charge_suffix, topology=self.to_openmm().topology, openffmol=openffmol)


        torch_energies = None if self.energies is None else torch.tensor(self.energies, dtype=torch.float32,).unsqueeze(dim=0)
        torch_gradients = None if self.gradients is None else torch.tensor(self.gradients, dtype=torch.float32,).transpose(0,1)

        g = utilities.write_reference_data(g, energies=torch_energies, gradients=torch_gradients, class_ff=suffix, ref_suffix=ref_suffix)

        for level in g.ntypes:
            if not level in self.graph_data.keys():
                self.graph_data[level] = {}
            for feat in g.nodes[level].data.keys():
                self.graph_data[level][feat] = g.nodes[level].data[feat].detach().numpy()

        
    def get_ase_bonds(self, idxs:List[int]=[0])->List[List[Tuple[int, int]]]:
        """
        Returns a list of shape len(idxs)*n_bonds of 2-tuples describing the bonds between the atoms where the indices correspond to the order of self.elements, xyz, etc., inferred by ase from the positions and elements only.
        """
        from ase.geometry.analysis import Analysis
        traj = self.to_ase(idxs)
        try:
            connectivities = [Analysis(traj[id]).nl[0].get_connectivity_matrix() for id in idxs]
        except IndexError:
            raise RuntimeError(f"Index out of bounds for get_ase_bonds. Max idx is {max(idxs)} but there are only {len(self)} conformations.")
        bonds = [[(n1,n2) for (n1,n2) in c.keys() if n1!=n2] for c in connectivities]
        return bonds
        

    def validate_confs(self, forcefield=ForceField("amber99sbildn.xml"))->Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Returns the rmse between the classical force field in angstrom and kcal/mol and the stored data, and the standard deviation of the data itself.
        """
        e_class, grad_class = self.get_ff_data(forcefield)
        e_class -= e_class.mean()
        en = self.energies - self.energies.mean(axis=-1)
        diffs_e = en - e_class
        if not self.gradients is None:
            diffs_g = self.gradients - grad_class
        
        if not self.gradients is None:
            return (np.sqrt(np.mean(diffs_e**2)), np.std(en)), (np.sqrt(np.mean(diffs_g**2)), np.std(self.gradients))
        else:
            return (np.sqrt(np.mean(diffs_e**2)), np.std(en)), (None, None)
        

    def conf_check(self, forcefield=ForceField("amber99sbildn.xml"), sigmas:Tuple[float,float]=(1.,1.))->bool:
        """
        Checks if the stored energies and gradients are consistent with the forcefield. Returns True if the energies and gradients are within var_param[0] and var_param[1] standard deviations of the stored energies/forces.
        If sigmas are 1, this corresponds to demanding that the forcefield data is better than simply always guessing the mean.
        """
        (rmse_e, std_e), (rmse_g, std_g) = self.validate_confs(forcefield)
        if not self.gradients is None:
            return rmse_e < sigmas[0]*std_e and rmse_g < sigmas[1]*std_g
        else:
            return rmse_e < sigmas[0]*std_e


    def filter_confs(self, max_energy:float=60., max_force:float=None)->bool:
        """
        Filters out conformations with energies or forces that are over 60 kcal/mol away from the minimum of the dataset (not the actual minimum). Apply this before parametrizing or re-apply the parametrization after filtering. Units are kcal/mol and kcal/mol/angstrom.
        """
        if (not max_energy is None) and (not self.energies is None):
            energies = self.energies - self.energies.min()
            energy_mask = energies < max_energy
        else:
            energy_mask = np.ones(len(self), dtype=bool)

        if (not max_force is None) and (not self.gradients is None):
            forces = np.max(np.max(np.abs(self.gradients), axis=-1), axis=-1)
            force_mask = forces < max_force
        else:   
            force_mask = np.ones(len(self), dtype=bool)

        mask = energy_mask & force_mask

        if not self.energies is None:
            self.energies = self.energies[mask]
        if not self.gradients is None:
            self.gradients = self.gradients[mask]
        self.xyz = self.xyz[mask]
        
        return self.xyz.shape[0] > 1



    def bond_check(self, idxs:List[int]=None, perm:bool=False, seed:int=0)->None:
        """
        For each configuration, creates a list of tuples describing the bonds between the atoms where the indices correspond to the order of self.elements, xyz, etc. and compares it to the list of bonds inferred by ase from the positions and elements only. Throws an error if the lists are not identical.
        If perm is True, also creates a list of bonds for a permutation of the atoms and checks if the lists are identical after permutating back.
        """

        # NOTE: what about indistinguishable atoms? (e.g. hydrogens)
        # -> should be fine since they have the same bonds if they are indistuinguishable

        if idxs is None:
            idxs = np.arange(len(self))
        # sort the bonds by the smaller index:
        pdb_bonds = [( min(b[0],b[1]), max(b[0], b[1]) ) for b in self.get_bonds()]
        ase_bonds = [[( min(b[0],b[1]), max(b[0], b[1]) ) for b in id_bonds] for id_bonds in self.get_ase_bonds(idxs)]

        atoms = [a.name for a in self.to_openmm().topology.atoms()]

        for id in idxs:
            ona = set(pdb_bonds) - set(ase_bonds[id])
            ano = set(ase_bonds[id]) - set(pdb_bonds)
            if len(ona) > 0 or len(ano) > 0:
                atoms = [a.name for a in self.to_openmm().topology.atoms()]

                errstr = f"Encountered dissimilar bonds during PDBMolecule.bond_check for config {id}.\nIn openmm but not ASE: {ona}, atoms {[(atoms[i], atoms[j]) for (i,j) in ona]}.\nIn ASE but not openmm: {ano}, atoms {[(atoms[i], atoms[j]) for (i,j) in ano]}. (The indices are starting at zero not at one!)"
                raise RuntimeError(errstr)

        if perm:

            # check if the bonds are the same after permuting the atoms:

            # get the permutation:
            np.random.seed(seed)
            perm = np.arange(len(self.elements))
            perm = np.random.permutation(perm)

            # permute the atoms:
            new_xyz = self.xyz[:,perm]
            new_elements = self.elements[perm]
            # create a new molecule:
            new_mol = PDBMolecule.from_xyz(xyz=new_xyz, elements=new_elements)
            # the new mol performed a permutation of the input, so we need to chain the two permutations:
            perm = perm[new_mol.permutation]
            
            # get the bonds:
            # [print(b[0], b[1]) for b in new_mol.to_openmm().topology.bonds()]

            new_bonds = new_mol.get_bonds(from_pdb=True)

            # permute indices back:
            perm_map = {i:np.arange(len(self.elements))[perm][i] for i in range(len(perm))}

            assert np.all([self.xyz[:,perm_map[i]]==new_mol.xyz[:,i] for i in range(len(self.elements))])

            # sort the bonds by the smaller index:
            new_bonds = [(perm_map[b[0]], perm_map[b[1]]) for b in new_bonds]
            new_bonds = [( min(b[0], b[1]), max(b[0], b[1]) ) for b in new_bonds]

            # check if the bonds are the same after permutating back:

            if set(pdb_bonds) != set(new_bonds):
                # find the bonds that are in one but not the other:
                nmo = set(new_bonds) - set(pdb_bonds)
                omn = set(pdb_bonds) - set(new_bonds)
                atoms = [a.name for a in self.to_openmm().topology.atoms()]

                errstr = f"Encountered dissimilar bonds during PDBMolecule.bond_check for permutated version.\nIn permuted but not original: {nmo}, atoms {[(atoms[i], atoms[j]) for (i,j) in nmo]}.\noriginal but not permuted {omn}, atoms {[(atoms[i], atoms[j]) for (i,j) in omn]}. (The indices are starting at zero not at one and belong to the original version!)"
                raise RuntimeError(errstr)


    @classmethod
    def from_pdb(cls, pdbpath:Union[str,Path], xyz:np.ndarray=None, energies:np.ndarray=None, gradients:np.ndarray=None, sequence:str=None):
        """
        Initializes the object from a pdb file and an xyz array of shape (N_confsxN_atomsx3). The atom order is the same as in the pdb file.
        """

        if type(pdbpath) == Path:
            pdbpath = str(pdbpath)

        for l, to_be_checked, name in [(1,energies, "energies"), (3,gradients,"gradients")]:
            if not to_be_checked is None:
                if xyz.shape[0] != to_be_checked.shape[0]:
                    raise RuntimeError(f"Number of conformations must be consistent between xyz and {name}, shapes are {xyz.shape} and {to_be_checked.shape}")
                if len(to_be_checked.shape) != l:
                    raise RuntimeError(f"{name} must be a {l}-dimensional array but is of shape {to_be_checked.shape}")
                
        if not gradients is None:
            if xyz.shape != gradients.shape:
                raise RuntimeError(f"Gradients and positions must have the same shape. Shapes are {gradients.shape} and {xyz.shape}.")
            
        obj = cls()
        obj.sequence = sequence
        pdbmol = PDBFile(pdbpath)
        obj.elements = np.array([a.element.atomic_number for a in pdbmol.topology.atoms()])
        if xyz is None:
            obj.xyz = pdbmol.getPositions(asNumpy=True).value_in_unit(angstrom).reshape(1,-1,3)
        else:
            obj.xyz = xyz

        with open(pdbpath, "r") as f:
            obj.pdb = f.readlines()

        obj.energies = energies
        obj.gradients = gradients
        obj.permutation = np.arange(len(obj.elements))
        
        # NOTE: implement a way to get the sequence from the pdb file?

        return obj

    @classmethod
    def from_gaussian_log(cls, logfile:Union[str,Path], cap:bool=True, rtp_path=None, sequence:list=None, logging:bool=False):
        """
        Use a gaussian logfile for initialization. Returns the initialized object.
        Parameters
        ----------
        logfile: str/pathlib.Path
            Path to the gaussian log file
        """
        if rtp_path is None:
            rtp_path = PDBMolecule.DEFAULT_RTP

        if type(rtp_path) == str:
            rtp_path = Path(rtp_path)

        obj = cls()

        AAs_reference = PDBData.matching.read_rtp(rtp_path)

        if type(logfile) == str:
            logfile = Path(logfile)

        if sequence is None:
            sequence = PDBData.matching.seq_from_filename(logfile, AAs_reference, cap)

        obj.sequence = ''
        for aa in sequence:
            obj.sequence += aa
            obj.sequence += '-'
        obj.sequence = obj.sequence[:-1]


        mol, trajectory = PDBData.matching.read_g09(logfile, obj.sequence, AAs_reference, log=logging)

        atom_order = PDBData.matching.match_mol(mol, AAs_reference, obj.sequence, log=logging)
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
            except ase.calculators.calculator.PropertyNotImplementedError:
                energy = float("nan")
            energies.append(energy)

            # NOTE: implement forces

            pos = step.positions[atom_order] # array of shape n_atoms x 3 in the correct order
            positions.append(pos)
        
        obj.energies = np.array(energies)
        obj.xyz = np.array(positions)
        return obj

    @classmethod
    def from_xyz(cls, xyz:np.ndarray, elements:np.ndarray, energies:np.ndarray=None, gradients:np.ndarray=None, rtp_path=None, residues:list=None, res_numbers:list=None, logging:bool=False, debug:bool=False):
        """
        Use an xyz array of shape (N_confsxN_atomsx3) and an element array of shape (N_atoms) for initialization. The atom order in which xyz and element are stored may differ from that of those used for initilization (See description of the xyz member).
        The positions must be given in angstrom.
        Currently only works for:
        peptides that do not contain TYR, PHE.
        and R,K: positive charge, D,E: negative charge, H: neutral - HIE
        """
        for l, to_be_checked, name in [(1,energies, "energies"), (3,gradients,"gradients")]:
            if not to_be_checked is None:
                if xyz.shape[0] != to_be_checked.shape[0]:
                    raise RuntimeError(f"Number of conformations must be consistent between xyz and {name}, shapes are {xyz.shape} and {to_be_checked.shape}")
                if len(to_be_checked.shape) != l:
                    raise RuntimeError(f"{name} must be a {l}-dimensional array but is of shape {to_be_checked.shape}")
                
        if not gradients is None:
            if xyz.shape != gradients.shape:
                raise RuntimeError(f"Gradients and positions must have the same shape. Shapes are {gradients.shape} and {xyz.shape}.")
            
        if not elements.shape[0] == xyz.shape[1]:
            raise RuntimeError(f"Number of atoms in xyz and elements must be the same. Shapes are {xyz.shape} and {elements.shape}.")
        

        if rtp_path is None:
            rtp_path = PDBMolecule.DEFAULT_RTP
        if type(rtp_path) == str:
            rtp_path = Path(rtp_path)

        obj = cls()
        
        # get the residues:
        if residues is None or res_numbers is None:
            residues, res_numbers = xyz2res(xyz[0], elements, debug=debug)


        # order such that residues belong toghether 
        l = [(res_numbers[i], i) for i in range(len(res_numbers))]
        if not all(l[i] <= l[i+1] for i in range(len(l) - 1)):
            l.sort()

        perm = [l[i][1] for i in range(len(l))]

        xyz = xyz[:,perm]
        elements = elements[perm]
        if not gradients is None:
            gradients = gradients[:,perm]

        if not energies is None:
            obj.energies = energies

        # make the sequence: first order the residues according to the ordering above
        residues = np.array(residues)[perm]
        res_numbers = np.array(res_numbers)[perm]

        # match_mol will leave this invariant, therefore we can set it already
        obj.res_numbers = np.array(res_numbers)
        obj.residues = residues
        
        seq = []
        num_before = -1
        for j,i in enumerate(res_numbers):
            if i != num_before:
                num_before = i
                seq.append(residues[j])

        # preparation for getting the pdb file
        AAs_reference = PDBData.matching.read_rtp(rtp_path)

        # generate ase molecule from permuted xyz and elements:
        n = xyz[0].shape[0]
        pos = xyz[0]
        ase_mol = ase.Atoms(f"N{n}")
        ase_mol.set_positions(pos)

        ase_mol.set_atomic_numbers(elements)

        mol, _ = PDBData.matching.read_g09(None, seq, AAs_reference, trajectory_in=[ase_mol], log=logging)

        atom_order = PDBData.matching.match_mol(mol, AAs_reference, seq, log=logging)

        # concatenate the permutations:
        obj.permutation = np.array(perm)[atom_order]

        pdb, _ = PDBMolecule.pdb_from_ase(ase_conformation=ase_mol, sequence=seq, AAs_reference=AAs_reference, atom_order=atom_order)

        # NOTE convert to openmm standard:
        # with tempfile.NamedTemporaryFile(mode='r+') as f:
        # with open("t.pdb",mode='r+') as f:
        #     f.writelines(pdb)
        #     openmm_pdb = PDBFile(f.name)
        #     PDBFile.writeModel(topology=openmm_pdb.getTopology(), positions=xyz[0], file=f)
        #     pdb = f.readlines()
        obj.pdb = pdb


        obj.elements = elements[atom_order]
        obj.xyz = xyz[:,atom_order]

        obj.sequence = ''
        for aa in seq:
            obj.sequence += aa
            obj.sequence += '-'
        obj.sequence = obj.sequence[:-1]

        # NOTE set residue and res numbers in correct order
        
        if not gradients is None:
            obj.gradients = gradients[:,atom_order]

        return obj


    @staticmethod
    def pdb_from_ase(ase_conformation, sequence, AAs_reference, atom_order):
        """
        Helper method for internal use. Generates a pdb file from an ase molecule, permuting the atoms in the ase mol according to atom_order.
        """
        pdb_lines = []
        elements = []

        # copied from PDBMolecule.write_trjtopdb
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
    

    # def get_min_energy(self, forcefield:ForceField=ForceField('amber99sbildn.xml')):
    #     from openmm.app import ForceField, PDBFile, topology, Simulation, PDBReporter, PME, HBonds, NoCutoff, CutoffNonPeriodic
    #     from openmm import LangevinMiddleIntegrator, Vec3
    #     from openmm.unit import Quantity, picosecond, kelvin, kilocalorie_per_mole, picoseconds, angstrom, nanometer, femtoseconds, newton

    #     assert not self.xyz is None
    #     assert not self.elements is None
    #     assert not self.pdb is None

    #     pdb_ = self.to_openmm()

    #     integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.5*femtoseconds)

    #     system = forcefield.createSystem(pdb_.topology)

    #     simulation = Simulation(
    #         pdb_.topology, system=system, integrator=integrator
    #     )

    #     simulation.context.setPositions(pdb_.positions)
    #     simulation.minimizeEnergy(maxIterations=int(10**3))

    #     state = simulation.context.getState(
    #                 getEnergy=True,
    #             )
    #     e = state.getPotentialEnergy()
    #     e = e.value_in_unit(kilocalorie_per_mole)
    #     return e
        


    def get_ff_data(self, forcefield:ForceField=ForceField('amber99sbildn.xml'))->Tuple[np.ndarray, np.ndarray]:
        """
        Returns energies, gradients that are calculated by the forcefield for all states xyz. Tha state axis is the zeroth axis.
        """
        from openmm.app import ForceField, PDBFile, topology, Simulation, PDBReporter, PME, HBonds, NoCutoff, CutoffNonPeriodic
        from openmm import LangevinMiddleIntegrator, Vec3
        from openmm.unit import Quantity, picosecond, kelvin, kilocalorie_per_mole, picoseconds, angstrom, nanometer, femtoseconds, newton

        assert not self.xyz is None
        assert not self.elements is None
        assert not self.pdb is None

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'pep.pdb')
            self.write_pdb(path)
            gen_pdb = PDBFile(path)
        gen_pdb = self.to_openmm()

        integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.5*femtoseconds)
        system = forcefield.createSystem(gen_pdb.topology)
        simulation = Simulation(
            gen_pdb.topology, system=system, integrator=integrator
        )

        # compare the datasets that should match
        ff_forces = []
        ff_energies = []
        for pos in self.xyz:
            simulation.context.setPositions(pos/10) # go to nanometer
            state = simulation.context.getState(
                    getEnergy=True,
                    getForces=True,
                )
            e = state.getPotentialEnergy()
            e = e.value_in_unit(kilocalorie_per_mole)
            f = state.getForces(True)
            f = f.value_in_unit(kilocalorie_per_mole/angstrom)
            f = np.array(f)
            ff_energies.append(e)
            ff_forces.append(f)

        ff_forces = np.array(ff_forces)
        ff_energies = np.array(ff_energies)
        return ff_energies, -ff_forces

    @staticmethod
    def do_energy_checks(path:Path, seed:int=0, permute:bool=True, n_conf:int=5, forcefield=None, accuracy:List[float]=[0.1, 1.], verbose:bool=True, fig:bool=True)->bool:
        """
        For all .pdb files in the path, create a set of configurations using openmm, then shuffle positions, elements and forces, create a PDBMolecule and use openmm to calculate the energies and forces belonging to the (re-ordered) positions of the mol, and compare these elementwise. Returns false if the energies deviate strongly.
        accuracy: the maximum allowed deviations in kcal/mol and kcal/mol/angstrom
        """
        counter = 0
        crashed = []

        en_diffs = []

        if verbose:
            print("doing energy checks...")
    
        for p in path.rglob('*.pdb'):
            if verbose:
                print(counter, end='\r')
            counter += 1
            pdb = PDBFile(str(p))
            l = []
            if not PDBMolecule.energy_check(pdb, seed=seed, permute=permute, n_conf=n_conf, forcefield=forcefield, accuracy=accuracy, store_energies_ptr=l):
                # get residue names:
                residues = list(set([a.residue.name for a in pdb.topology.atoms()]))
                crashed.append(residues)
                en_diffs.append(np.mean(np.abs(l[0]-l[1])))

        crashcount = {}

        if len(crashed) > 0:
            import matplotlib.pyplot as plt
            for l in crashed:
                for res in l:
                    if not res in crashcount.keys():
                        crashcount[res] = 1
                    else:
                        crashcount[res] += 1
            if "ACE" in crashcount.keys():
                crashcount.pop("ACE")
            if "NME" in crashcount.keys():
                crashcount.pop("NME")
            if verbose:
                print()
                print(f"energy check errors occured {len(crashed)} out of {counter} times.\nThe following residues where involved:\n{crashcount}")
                print(f"mean energy difference: {np.mean(np.array(en_diffs))} kcal/mol")
                if fig and len(crashed) > 1:
                    plt.figure(figsize=(7,4))
                    plt.title("Failed energy test occurences")
                    plt.bar(crashcount.keys(), crashcount.values())
                    plt.savefig(f"failed_energy_{path.stem}.png")

                    plt.figure(figsize=(7,4))
                    plt.title("Energy diffs")
                    plt.hist(np.array(en_diffs), bins=20)
                    plt.savefig(f"energy_diffs_{path.stem}.png")
            return False
        else:
            if verbose:
                print()
                print(f"no errors occured for {counter} files")
            return True

                

    @staticmethod
    def energy_check(pdb:PDBFile, seed:int=0, permute:bool=True, n_conf:int=5, forcefield:ForceField=ForceField('amber99sbildn.xml'), accuracy:List[float]=[0.1, 1.], store_energies_ptr:list=None)->bool:
        """
        Create a set of configurations using openmm, then shuffle positions, elements and forces, create a PDBMolecule and use openmm to calculate the energies and forces belonging to the (re-ordered) positions of the mol, and compare these elementwise. Returns false if the energies deviate strongly.
        accuracy: the maximum allowed deviations in kcal/mol and kcal/mol/angstrom
        """
        from openmm.app import ForceField, PDBFile, topology, Simulation, PDBReporter, PME, HBonds, NoCutoff, CutoffNonPeriodic
        from openmm import LangevinMiddleIntegrator, Vec3
        from openmm.unit import Quantity, picosecond, kelvin, kilocalorie_per_mole, picoseconds, angstrom, nanometer, femtoseconds, newton


        # generate a dataset

        integrator = LangevinMiddleIntegrator(500*kelvin, 1/picosecond, 0.5*femtoseconds)

        forcefield=ForceField('amber99sbildn.xml')
        system = forcefield.createSystem(pdb.topology)

        simulation = Simulation(
            pdb.topology, system=system, integrator=integrator
        )

        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy(maxIterations=int(10**3), tolerance=accuracy[0]*kilocalorie_per_mole)
        simulation.step(100)

        elements = np.array([a.element.atomic_number for a in pdb.topology.atoms()])

        amber_e = []
        amber_f = []
        xyz = []
        for t in range(n_conf):
            simulation.step(100)
            state = simulation.context.getState(
                    getEnergy=True,
                    getPositions=True,
                    getForces=True,
                )
            e = state.getPotentialEnergy()
            e = e.value_in_unit(kilocalorie_per_mole)
            f = state.getForces(True)
            f = f.value_in_unit(kilocalorie_per_mole/angstrom)
            f = np.array(f)
            pos = state.getPositions().value_in_unit(angstrom)

            xyz.append(np.array(pos))
            amber_e.append(e)
            amber_f.append(f)

        amber_e = np.array(amber_e)
        amber_f = np.array(amber_f)
        xyz = np.array(xyz)

        # shuffle the atoms
        perm = np.arange(xyz.shape[1])
        np.random.seed(seed=seed)
        if permute:
            np.random.shuffle(perm)
        xyz = xyz[:,perm]
        elements = elements[perm]
        amber_f = amber_f[:,perm]

     # generate a pdb molecule with our package
        mol = PDBMolecule.from_xyz(xyz=xyz, elements=elements, energies=amber_e, gradients=-amber_f)

        new_energies, new_grad = mol.get_ff_data(forcefield=forcefield)

        emse, fmse = np.sum((new_energies - mol.energies)**2)/len(new_energies), np.sum((new_grad - mol.gradients)**2)/len(new_grad)

        if not store_energies_ptr is None:
            store_energies_ptr.extend([new_energies, mol.energies, new_grad, mol.gradients])

        if emse > accuracy[0] or fmse > accuracy[1]:
            return False
        else:
            return True


# %%
# some small testing:
if __name__ == "__main__":
    DATASET_PATH = Path(__file__).parent.parent/Path("PDBData/xyz2res/scripts/data/pdbs/pep1")

    #%%
    m = PDBMolecule.from_pdb(pdbpath=str(DATASET_PATH/"E/pep.pdb"))
    #%%
    m.parametrize()
    g = m.to_dgl()
    #%%
    g.nodes['n1'].data.keys()
    #%%
    from openmm.unit import bohr, Quantity, hartree, mole
    import numpy as np

    from PDBData.units import DISTANCE_UNIT, ENERGY_UNIT, FORCE_UNIT

    PARTICLE = mole.create_unit(
        6.02214076e23 ** -1,
        "particle",
        "particle",
    )

    HARTREE_PER_PARTICLE = hartree / PARTICLE

    SPICE_DISTANCE = bohr
    SPICE_ENERGY = HARTREE_PER_PARTICLE
    SPICE_FORCE = SPICE_ENERGY / SPICE_DISTANCE

    spicepath = "/hits/fast/mbm/seutelf/data/datasets/SPICE-1.1.2.hdf5"
    name = "hie-hie"
    import h5py
    with h5py.File(spicepath, "r") as f:
        elements = f[name]["atomic_numbers"]
        energies = f[name]["dft_total_energy"]
        xyz = f[name]["conformations"] # in bohr
        gradients = f[name]["dft_total_gradient"]
        # convert to kcal/mol and kcal/mol/angstrom
        elements = np.array(elements, dtype=np.int64)
        xyz = Quantity(np.array(xyz), SPICE_DISTANCE).value_in_unit(DISTANCE_UNIT)
        gradients = Quantity(np.array(gradients), SPICE_FORCE).value_in_unit(FORCE_UNIT)
        energies = Quantity(np.array(energies)-np.array(energies).mean(axis=-1), SPICE_ENERGY).value_in_unit(ENERGY_UNIT)

    m = PDBMolecule.from_xyz(xyz, elements, energies, gradients)
    #%%
    #%%
    e = g.nodes['g'].data['u_qm'].detach().clone().numpy()
    a = g.nodes['g'].data['u_total_amber99sbildn'].numpy()

    e-=e.min()
    a-=a.min()
    import matplotlib.pyplot as plt
    plt.scatter(e, a)

    #%%
    m.compress("tmp.npz")
    loaded = PDBMolecule.load("tmp.npz")
    # loading and saving works:
    assert np.all(loaded.xyz == m.xyz)
    assert np.all(loaded.energies == m.energies)
    assert torch.allclose(g.nodes['g'].data['u_qm'], loaded.to_dgl().nodes['g'].data['u_qm']), f"{g.nodes['g'].data['u_qm']}\n{loaded.to_dgl().nodes['g'].data['u_qm']}"
    assert torch.allclose(g.nodes['n1'].data['grad_total_amber99sbildn'], loaded.to_dgl().nodes['n1'].data['grad_total_amber99sbildn'])
    assert "u_ref" in loaded.to_dgl().nodes['g'].data.keys(), loaded.to_dgl().nodes['g'].data.keys()
    #%%
    t = m.xyz[0,0,0]
    m.xyz[0,0,0] = 20
    print(m.conf_check())
    m.xyz[0,0,0] = t
    print(m.conf_check())

    #%%

    m.filter_confs(max_energy=50)
    m.parametrize()
    g = m.to_dgl()
    e = g.nodes['g'].data['u_qm'].detach().clone().numpy()
    a = g.nodes['g'].data['u_total_amber99sbildn'].numpy()

    e-=e.min()
    a-=a.min()
    import matplotlib.pyplot as plt
    plt.scatter(e, a)


    #%%
















    #%%
    PDBMolecule.do_energy_checks(DATASET_PATH, accuracy=[0.1, 1.])
    #%%
    m = PDBMolecule.from_pdb(pdbpath=str(DATASET_PATH/"E/pep.pdb"))
    #m = PDBMolecule.from_xyz(xyz=xyz, elements=elements)
    m.bond_check(perm=True, seed=0)
    #%%
    l=[]
    PDBMolecule.energy_check(PDBFile(str(DATASET_PATH/Path("E/pep.pdb"))), store_energies_ptr=l, permute=True, seed=0)
    #%%
    PDBMolecule.do_energy_checks(Path(DATASET_PATH).parent/Path("pep2"))
    #%%
    PDBMolecule.do_energy_checks(Path(DATASET_PATH).parent/Path("pep5"))
    #%%
    from openmm.unit import bohr, Quantity, hartree, mole
    import numpy as np

    from PDBData.units import DISTANCE_UNIT, ENERGY_UNIT, FORCE_UNIT

    PARTICLE = mole.create_unit(
        6.02214076e23 ** -1,
        "particle",
        "particle",
    )

    HARTREE_PER_PARTICLE = hartree / PARTICLE

    SPICE_DISTANCE = bohr
    SPICE_ENERGY = HARTREE_PER_PARTICLE
    SPICE_FORCE = SPICE_ENERGY / SPICE_DISTANCE

    spicepath = "/hits/fast/mbm/seutelf/data/datasets/SPICE-1.1.2.hdf5"
    name = "hie-hie"
    import h5py
    with h5py.File(spicepath, "r") as f:
        elements = f[name]["atomic_numbers"]
        energies = f[name]["dft_total_energy"]
        xyz = f[name]["conformations"] # in bohr
        gradients = f[name]["dft_total_gradient"]
        # convert to kcal/mol and kcal/mol/angstrom
        elements = np.array(elements, dtype=np.int64)
        xyz = Quantity(np.array(xyz), SPICE_DISTANCE).value_in_unit(DISTANCE_UNIT)
        gradients = Quantity(np.array(gradients), SPICE_FORCE).value_in_unit(FORCE_UNIT)
        energies = Quantity(np.array(energies)-np.array(energies).mean(axis=-1), SPICE_ENERGY).value_in_unit(ENERGY_UNIT)

    m = PDBMolecule.from_xyz(xyz, elements, energies, gradients)
    #%%
    g = m.to_dgl()
    g.nodes["n2"].data["idxs"]
    #%%
    print(m.validate_confs())  
    #%%
    import matplotlib.pyplot as plt
    e_class = m.get_ff_data()[0]
    plt.scatter(e_class-e_class.mean(), m.energies-m.energies.mean())
    #%%
    g_class = m.get_ff_data()[1]
    plt.scatter(g_class, m.gradients)
    #%%




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
