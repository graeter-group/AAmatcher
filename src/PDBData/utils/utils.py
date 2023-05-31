#%%
from openmm.app import PDBFile

# supress openff warning:
import logging
logging.getLogger("openff").setLevel(logging.ERROR)
from openff.toolkit.topology import Molecule

from PDBData.create_graph.read_heterogeneous_graph import from_homogeneous_and_mol
from PDBData.create_graph.read_homogeneous_graph import from_openff_toolkit_mol

import dgl
import h5py

import numpy as np
import tempfile
import os.path
import torch
import openff.toolkit.topology
import openmm.app
from typing import Union
from pathlib import Path

import copy

def is_energy(ftype):
    return ("energy" in ftype) or ("energ" in ftype) or ("u_" in ftype)

def is_n1_conf(ftype):
    return ("grad" in ftype) or ("force" in ftype) or ("xyz" in ftype)

def is_n1(ftype):
    return is_n1_conf(ftype) or ("elem" in ftype) or ("h" in ftype)

def get_pdbfile(hdf5:str, idx:int):
    with h5py.File(hdf5, "r") as f:
        for i, name in enumerate(f.keys()):
            if i==idx:
                pdb = f[name]["pdb"]
                with tempfile.TemporaryDirectory() as tmp:
                    pdbpath = os.path.join(tmp, 'pep.pdb')
                    with open(pdbpath, "w") as pdb_file:
                        pdb_file.writelines([line.decode('UTF-8') for line in pdb])
                    return PDBFile(pdbpath)


#%%
def pdb2dgl(pdbpath:Union[str,Path], chemical_properties=False) -> dgl.graph:
    if chemical_properties:
        openff_mol = pdb2openff(str(pdbpath))
    else:
        openff_mol = pdb2openff_graph(str(pdbpath))

    return openff2dgl(openff_mol)


def openff2dgl(openff_mol:openff.toolkit.topology.Molecule) -> dgl.graph:
    homgraph = from_openff_toolkit_mol(openff_mol)
    g = from_homogeneous_and_mol(homgraph, openff_mol)
    return g


def replace_h23_to_h12(pdb:PDBFile):
    """
    for every residue, remap Hs (2,3) -> (2,3), eg HB2, HB3 -> HB2, HB1.
    (in this order to avoid double remapping)
    """

    for atom in pdb.topology.atoms():
        if atom.residue.name in replace_h23_to_h12.d.keys():
            for level in replace_h23_to_h12.d[atom.residue.name]:
                for i in ["3"]:
                    if atom.name == f"H{level}{i}":
                        atom.name = f"H{level}{int(i)-2}"
                        continue
        elif atom.residue.name == "ILE":
            for i in ["3"]:
                if atom.name == f"HG1{i}":
                    atom.name = f"HG1{int(i)-2}"
                    continue
            for i in ["1","2","3"]:
                if atom.name == f"HD1{i}":
                    atom.name = f"HD{i}"
            if atom.name == f"CD1":
                atom.name = f"CD"
                continue

    return pdb

replace_h23_to_h12.d = {"CYS":["B"], "ASP":["B"], "GLU":["B","G"], "PHE":["B"], "GLY":["A"], "HIS":["B"], "LYS":["B","G","D","E"], "LEU":["B"], "MET":["B","G"], "ASN":["B"], "PRO":["B","G","D"], "GLN":["B", "G"], "ARG":["B","G","D"], "SER":["B"], "TRP":["B"], "TYR":["B"]}



def pdb2openff(pdbpath:str) -> Molecule:
    openff_mol = Molecule.from_polymer_pdb(str(pdbpath))
    return openff_mol


def pdb2openff_graph(pdbpath:Union[str,Path])->openff.toolkit.topology.Molecule:
    """
    Returns an openff molecule for representing the graph structure of the molecule, without chemical details such as bond order, formal charge and stereochemistry.
    """
    openmm_top = openmm.app.PDBFile(pdbpath).topology
    mol = openmm2openff_graph(openmm_top)
    return mol


def openmm2openff_graph(openmm_top:openmm.app.topology.Topology)->openff.toolkit.topology.Molecule:
    """
    Returns an openff molecule for representing the graph structure of the molecule, without chemical details such as bond order, formal charge and stereochemistry.
    To assign formal charges, the topology must contain residues.
    """
    mol = Molecule()
    # zero charge used for all atoms, regardless what charge they actually have
    # NOTE: later, find a way to infer charge from pdb, by now we only have standard versions of AAs
    idx_lookup = {} # maps from openmm atom index to openff atom index (usually they agree)
    for i, atom in enumerate(openmm_top.atoms()):
        charge = assign_standard_charge(atom_name=atom.name, residue=atom.residue.name)
        mol.add_atom(atom.element.atomic_number, formal_charge=charge, is_aromatic=False, stereochemistry=None)
        idx_lookup[atom.index] = i

    # bond_order 1 used for all bonds, regardless what type they are
    for bond in openmm_top.bonds():
        mol.add_bond(idx_lookup[bond.atom1.index], idx_lookup[bond.atom2.index], bond_order=1, is_aromatic=False, stereochemistry=None)

    return mol



def assign_standard_charge(atom_name:str, residue:str):
    """
    Assigns standard charges to atoms based on their name and residue.
    """

    if residue in assign_standard_charge.atom_dict.keys():
        if atom_name == assign_standard_charge.atom_dict[residue]:
            return assign_standard_charge.charge_dict[residue]
    # else, the atom is not charged
    return 0
    
assign_standard_charge.atom_dict = {
    "GLU":"OE1", 
    "ASP":"OD1",
    "LYS":"NZ",
    "ARG":"NH1",
    "HIS":"ND1", # choose HIE as the standard protonation state
    "HIE":"ND1"
    }

assign_standard_charge.charge_dict = {
    "GLU":-1,
    "ASP":-1,
    "LYS":1,
    "ARG":1,
    "HIS":1,
    "HIE":1
    }
