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


""" 
Build simple graph from RDKit molecule object, containing only geometric data, no chemical information such as bond order, formal charge, etc.
"""
# =============================================================================
# IMPORTS
# =============================================================================
import torch

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def fp_rdkit(atom):
    """
    Returns a 10-dimensional feature vector for a given atom.
    """
    return torch.tensor(
                [   
                    atom.IsInRing() * 1.0,
                    atom.GetTotalDegree(),
                    atom.GetFormalCharge(), # NOTE: currently, this has no effect
                    atom.GetMass(),
                    atom.IsInRingSize(3) * 1.0,
                    atom.IsInRingSize(4) * 1.0,
                    atom.IsInRingSize(5) * 1.0,
                    atom.IsInRingSize(6) * 1.0,
                    atom.IsInRingSize(7) * 1.0,
                    atom.IsInRingSize(8) * 1.0,
                ],
                dtype=torch.float32,
            )

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def from_openff_toolkit_mol(mol, use_fp=True, max_element=26):
    """
    Creates a homogenous dgl graph containing the one-hot encoded elements and, if use_fp, the 10 dimensional rdkit features. Stored in the feature type 'h0'. To get the elements only, use the feature type 'h0' and slice the first max_element elements.
    """
    import dgl

    # enter bonds
    bonds = list(mol.bonds)
    bonds_begin_idxs = [bond.atom1_index for bond in bonds]
    bonds_end_idxs = [bond.atom2_index for bond in bonds]

    # initialize graph
    g = dgl.graph((bonds_begin_idxs, bonds_end_idxs))

    g = dgl.add_reverse_edges(g)

    n_atoms = mol.n_atoms
    assert n_atoms == g.num_nodes() , f"error initializing the homogeneous graph: inferred {g.num_nodes()} atoms from the given edges but the molecule has {n_atoms}"

    atomic_numbers = torch.Tensor(
        [atom.atomic_number for atom in mol.atoms]
    )

    atomic_numbers = torch.nn.functional.one_hot(atomic_numbers.long(), num_classes=max_element)

    h_v = atomic_numbers.float()

    if use_fp:
        h_v_fp = torch.stack(
            [fp_rdkit(atom) for atom in mol.to_rdkit().GetAtoms()], dim=0
        )
        h_v = torch.cat([h_v, h_v_fp], dim=-1)  # (n_atoms, 10+max_element)


    g.ndata["h0"] = h_v

    return g
