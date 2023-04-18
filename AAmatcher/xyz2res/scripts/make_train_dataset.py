#%%
import torch
import dgl
import ase
from openmm.app import PDBFile
from openmm.unit import angstrom
import numpy as np
from ase.geometry.analysis import Analysis
from pathlib import Path
import json

from AAmatcher.xyz2res.constants import MAX_ELEMENT, RESIDUES

POS_UNIT = angstrom


# in: graph with elements
# out: graph with elements and residue

# next model: instance segmentation on this

class BondMismatch(Exception):
    pass

class NodeMismatch(Exception):
    pass

if __name__=="__main__":

    DS = "pep2"

    graphs = []
    labels = []

    # %%
    counter = 0
    errors = []
    err_types = []
    err_res = dict.fromkeys(RESIDUES)
    for k in err_res.keys():
        err_res[k] = 0

    for filename in Path(DS).rglob('*.pdb'):
        counter += 1
        try:
            labels.append(str(filename))
            mol = PDBFile(str(filename))
            atom_numbers = []
            residues = []
            res_indices = []
            atom_names = []
            res_names = []
            for a in mol.topology.atoms():
                atom_numbers.append(a.element.atomic_number)
                residues.append(RESIDUES.index(a.residue.name))
                res_indices.append(a.residue.index)
                atom_names.append(a.name)
                res_names.append(a.residue.name)

            res_names = set(res_names)
            
            # for now, gly is problematic due to the 3HA
            if "GLY" in res_names:
                continue

            bonds = []
            for b in mol.topology.bonds():
                a0, a1 = b[0].index, b[1].index
                if a0 < a1:
                    bonds.append((a0,a1))
                else:
                    bonds.append((a1,a0))

            n = mol.topology.getNumAtoms()
            pos = mol.positions
            pos = pos.value_in_unit(POS_UNIT)
            pos = np.array([np.array(v) for v in pos])

            ase_mol = ase.Atoms(f"N{n}")
            ase_mol.set_positions(pos)
            ase_mol.set_atomic_numbers(atom_numbers)

            ana = Analysis(ase_mol)
            connectivity = ana.nl[0].get_connectivity_matrix()

            node_pairs = [(n1,n2) for (n1,n2) in connectivity.keys() if n1!=n2]

            # if no ring is included, check the bonds
            if not any([ring in res_names for ring in ["HIS", "PHE", "TRP", "TYR"]]):
        
                if set(bonds) != set(node_pairs):
                    raise BondMismatch(f"After {len(graphs)} runs, encountered dissimilar bonds at {str(filename)}. diff: {(set(bonds) - set(node_pairs)).union(set(node_pairs) - set(bonds))}")


            node_tensors = [ torch.tensor([node_pair[i] for node_pair in node_pairs], dtype=torch.int32) for i in [0,1] ]

            g = dgl.graph(tuple(node_tensors))
            
            g.ndata["atomic_number"] = torch.nn.functional.one_hot(torch.tensor(atom_numbers), num_classes=MAX_ELEMENT)*1.
            g.ndata["residue"] = torch.tensor(residues)
            g.ndata["res_index"] = torch.tensor(res_indices)

            g.ndata["c_alpha"] = torch.tensor([name == "CA" for name in atom_names])

            g = dgl.add_reverse_edges(g)

            graphs.append(g)
        
        except BondMismatch as e:
            errors.append(str(filename))
            err_types.append(type(e))
            for res in set([RESIDUES[i] for i in residues]):
                err_res[res] += 1
            continue

        except Exception as e:
            raise e

    #%%
    print("errors at: ")
    print(errors)
    print()
    print(err_types)
    print()
    print("errors occured n times for ")
    print(err_res)
    print(f"successful for {len(graphs)} out of {counter} molecules.")
    #%%

    dspath = str(Path('tmp')/Path(DS+'.bin'))
    dgl.data.utils.save_graphs(dspath, graphs)

    with open(str(Path('tmp')/Path(DS+'_files.json')), "w") as f:
        json.dump(labels, f)
    # %%
