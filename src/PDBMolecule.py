from AAmatcher import *

DEFAULT_RTP = Path(__file__).parent.parent/Path("example/amber99sb-star-ildnp.ff/aminoacids.rtp")

# class for storing a dataset for a molecule that can be described by a pdb file from given coordinates and elements
# given a sequence, can also be initialised from a gaussian log file

class PDBMolecule:
    # use the classmethods to construct the object!
    def __init__(self):
        self.sequence = None # list of strings describing the sequence
        self.mol = None # list of AtomLists characterizing the molecule structure

        # numpy array containing the positions. shape: n_conf x n_atoms x 3
        # the atom order corresponds to that of the pdb file.
        self.xyz = None

        # array of relative energies
        self.energies = None

        # store the negative forces in an array
        self.gradients = None

        # array containing the elements numbers. shape: n_atoms:
        # the atom order corresponds to that of the pdb file.
        self.elements = None

        # store a list of strings representing the pdb-file of some conformation.
        # this stores information on connectivity of the molecule
        self.pdb = None

    def write_pdb(self, pdb_path="my_pdb.pdb"):
        with open(pdb_path, "w") as f:
            f.writelines(self.pdb)

    def get_openmm_pdb(self):
        pass

    # return an array containing the 3-format residue string for each atom
    def get_residues(self):
        pass

    def __getitem__(self, idx):
        return self.xyz[idx], self.energies[idx], self.gradients[idx]

    def __len__(self):
        return len(self.xyz)



    @classmethod
    def from_gaussian_log(cls, logfile, cap=True, rtp_path=DEFAULT_RTP):
        obj = cls()

        AAs_reference = read_rtp(rtp_path)
        cls.sequence = seq_from_filename(logfile, AAs_reference, cap)
        cls.mol, trajectory = read_g09(logfile, cls.sequence, AAs_reference)
        atom_order = match_mol(cls.mol, AAs_reference, cls.sequence)

        # write single pdb
        conf = trajectory[0]
        obj.pdb, elements = PDBMolecule.pdb_from_ase(ase_conformation=conf, sequence=cls.sequence, AAs_reference=AAs_reference,atom_order=atom_order)

        obj.elements = np.array(elements)

        # write xyz and energies
        energies = []
        positions = []
        forces = []

        # NOTE: take higher precision for energies since we are interested in differences that are tiny compared to the absolute magnitude
        for step in trajectory:
            try:
                energy = step.get_total_energy()
            except PropertyNotImplementedError:
                energy = float("nan")
            energies.append(energy)

            #NOTE: implement forces

            pos = step.positions # array of shape n_atoms x 3
            positions.append(pos)
        
        obj.energies = np.array(energies)
        obj.positions = np.array(positions)
        return obj

    @classmethod
    def from_xyz(cls, xyz, elements):
        # NOTE: implement this
        cls.xyz = xyz
        cls.elements = elements
        assert False


    # helper method
    @staticmethod
    def pdb_from_ase(ase_conformation, sequence, AAs_reference, atom_order):
        pdb_lines = []
        elements = []

        # copied from AAmatcher.write_trjtopdb
        AA_pos = 0
        res_len = len(AAs_reference[sequence[AA_pos]]['atoms'])
        for j,k in enumerate(atom_order):
            if j >= res_len:
                AA_pos += 1
                res_len += len(AAs_reference[sequence[AA_pos]]['atoms'])
            atomname = AAs_reference[sequence[AA_pos]]['atoms'][j-res_len][0]

            pdb_lines.append('{0}  {1:>5d} {2:^4s} {3:<3s} {4:1s}{5:>4d}    {6:8.3f}{7:8.3f}{8:8.3f}{9:6.2f}{10:6.2f}          {11:>2s}\n'.format('ATOM',j,atomname ,sequence[AA_pos].split(sep="_")[0],'A',AA_pos+1,*ase_conformation.positions[k],1.00,0.00,atomname[0]))

            # NOTE: this is not the atom number at the moment
            elements.append(atomname)

        pdb_lines.append("END")
        return pdb_lines, elements
