# inspired from espaloma:


# MIT License

# Copyright (c) 2020 Yuanqing Wang @ choderalab // MSKCC

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


""" Build simple graph from RDKit molecule object.

"""
# =============================================================================
# IMPORTS
# =============================================================================
import torch

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def fp_rdkit(atom):
    from rdkit import Chem

    HYBRIDIZATION_RDKIT = {
        Chem.rdchem.HybridizationType.SP: torch.tensor(
            [1, 0, 0, 0, 0],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.SP2: torch.tensor(
            [0, 1, 0, 0, 0],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.SP3: torch.tensor(
            [0, 0, 1, 0, 0],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.SP3D: torch.tensor(
            [0, 0, 0, 1, 0],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.SP3D2: torch.tensor(
            [0, 0, 0, 0, 1],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.S: torch.tensor(
            [0, 0, 0, 0, 0],
            dtype=torch.get_default_dtype(),
        ),
    }
    return torch.cat(
        [
            torch.tensor(
                [
                    atom.GetTotalDegree(),
                    atom.GetTotalValence(),
                    atom.GetExplicitValence(),
                    atom.GetFormalCharge(),
                    atom.GetIsAromatic() * 1.0,
                    atom.GetMass(),
                    atom.IsInRingSize(3) * 1.0,
                    atom.IsInRingSize(4) * 1.0,
                    atom.IsInRingSize(5) * 1.0,
                    atom.IsInRingSize(6) * 1.0,
                    atom.IsInRingSize(7) * 1.0,
                    atom.IsInRingSize(8) * 1.0,
                ],
                dtype=torch.get_default_dtype(),
            ),
            HYBRIDIZATION_RDKIT[atom.GetHybridization()],
        ],
        dim=0,
    )


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def from_openff_toolkit_mol(mol, use_fp=True):
    import dgl

    # enter bonds
    bonds = list(mol.bonds)
    bonds_begin_idxs = [bond.atom1_index for bond in bonds]
    bonds_end_idxs = [bond.atom2_index for bond in bonds]
    bonds_types = [bond.bond_order for bond in bonds]

    # initialize graph
    g = dgl.graph((bonds_begin_idxs, bonds_end_idxs))

    g = dgl.add_reverse_edges(g)

    n_atoms = mol.n_atoms
    assert n_atoms == g.num_nodes() , f"error initializing the homogeneous graph: inferred {g.num_nodes()} atoms from the given edges but the molecule has {n_atoms}"

    g.ndata["type"] = torch.Tensor(
        [[atom.atomic_number] for atom in mol.atoms]
    )
    
    h_v = torch.zeros(
        g.ndata["type"].shape[0], 100, dtype=torch.get_default_dtype()
    )

    h_v[
        torch.arange(g.ndata["type"].shape[0]),
        torch.squeeze(g.ndata["type"]).long(),
    ] = 1.0

    h_v_fp = torch.stack(
        [fp_rdkit(atom) for atom in mol.to_rdkit().GetAtoms()], axis=0
    )

    if use_fp == True:
        h_v = torch.cat([h_v, h_v_fp], dim=-1)  # (n_atoms, 117)

    g.ndata["h0"] = h_v

    # g.edata["type"] = torch.Tensor(bonds_types)[:, None].repeat(2, 1)

    return g
