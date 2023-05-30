#%%
from openmm.app import PDBFile
from openff.toolkit.topology import Molecule

from PDBData.read_heterogeneous_graph import from_homogeneous_and_mol
from PDBData.read_homogeneous_graph import from_openff_toolkit_mol

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
def pdb2dgl(pdbpath:str) -> dgl.graph:
    openff_mol = Molecule.from_polymer_pdb(str(pdbpath))
    homgraph = from_openff_toolkit_mol(openff_mol)
    g = from_homogeneous_and_mol(homgraph, openff_mol)
    return g


def replace_h23_to_h12(pdb:PDBFile):
    """
    for every residue, remap Hs (2,3) -> (2,3), eg HB2, HB3 -> HB2, HB1.
    (in this order to avoid double remapping)
    """
    d = {"CYS":["B"], "ASP":["B"], "GLU":["B","G"], "PHE":["B"], "GLY":["A"], "HIS":["B"], "LYS":["B","G","D","E"], "LEU":["B"], "MET":["B","G"], "ASN":["B"], "PRO":["B","G","D"], "GLN":["B", "G"], "ARG":["B","G","D"], "SER":["B"], "TRP":["B"], "TYR":["B"]}
    for atom in pdb.topology.atoms():
        if atom.residue.name in d.keys():
            for level in d[atom.residue.name]:
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


def pdb2openff(pdbpath:str) -> Molecule:
    openff_mol = Molecule.from_polymer_pdb(str(pdbpath))
    return openff_mol

def get_graphs(hdf5:str, in_ram:bool=True, verbose:bool=False, n_max:int=None) -> list:
    graphs = []
    with h5py.File(hdf5, "r") as f:
        for idx, name in enumerate(f.keys()):
            if not n_max is None:
                if idx >= n_max:
                    break
            if verbose:
                print(f"generating graphs, idx={idx}, name={name[:6]}...", end="\r")
            pdb = f[name]["pdb"]
            with tempfile.TemporaryDirectory() as tmp:
                pdbpath = os.path.join(tmp, 'pep.pdb')
                with open(pdbpath, "w") as pdb_file:
                    pdb_file.writelines([line.decode('UTF-8') for line in pdb])
                g = pdb2dgl(pdbpath)
            if in_ram:
                g = write_in_graph(g, dictlike=f[name])
            graphs.append(g)

    if verbose:
        print()
    return graphs


def write_data_to_graphs(graphs:list, hdf5:str, verbose:bool=False) -> list:
    with h5py.File(hdf5, "r") as f:
        for idx, name in enumerate(f.keys()):
            if idx>=len(graphs):
                break
            if verbose:
                print(f"writing data to graph, idx={idx}, name={name[:6]}...", end="\r")
            graphs[idx] = write_in_graph(g=graphs[idx], dictlike=f[name])
    if verbose:
        print()
    return graphs

def remove_conformational_data(graphs:list) -> list:
    graphs_ = copy.deepcopy(graphs)
    for idx in range(len(graphs_)):
        for ntype in ["g", "n1"]:
            ftps = list(graphs[idx].nodes[ntype].data.keys())
            for feat_type in ftps:
                is_en = is_energy(feat_type)
                is_nfeat = (ntype=="n1" and is_n1_conf(feat_type))
                if is_en or is_nfeat:
                    graphs_[idx].nodes[ntype].data.pop(feat_type)
    return graphs_
                    

def write_in_graph(g:dgl.graph, dictlike) -> dgl.graph:
    for key in dictlike.keys():
        if key=="pdb":
            continue
        value = torch.tensor(np.array(dictlike[key])).float()
        if is_energy(key):
            try:
                g.nodes["g"].data[key] = value
            except Exception as e:
                print(key)
                raise e
        elif is_n1(key):
            try:
                g.nodes["n1"].data[key] = value
            except Exception as e:
                print(key)
                raise e
        else:
            raise RuntimeError(f"could not assign key value {key} to n1 or g level")
    return g



def openmm2openff_graph(openmm_top:openmm.app.topology.Topology)->openff.toolkit.topology.Molecule:
    """
    Returns an openff molecule for representing the graph structure of the molecule, without chemical details such as bond order, formal charge and stereochemistry.
    """
    mol = Molecule()
    # zero charge used for all atoms, regardless what charge they actually have
    # NOTE: later, find a way to infer charge from pdb, by now we only have standard versions of AAs
    idx_lookup = {} # maps from openmm atom index to openff atom index (usually they agree)
    for i, atom in enumerate(openmm_top.atoms()):
        mol.add_atom(atom.element.atomic_number, formal_charge=0, is_aromatic=False, stereochemistry=None)
        idx_lookup[atom.index] = i

    # bond_order 1 used for all bonds, regardless what type they are
    for bond in openmm_top.bonds():
        mol.add_bond(idx_lookup[bond.atom1.index], idx_lookup[bond.atom2.index], bond_order=1, is_aromatic=False, stereochemistry=None)

    return mol

def pdb2openff_graph(pdbpath:Union[str,Path])->openff.toolkit.topology.Molecule:
    """
    Returns an openff molecule for representing the graph structure of the molecule, without chemical details such as bond order, formal charge and stereochemistry.
    """
    openmm_top = openmm.app.PDBFile(pdbpath).topology
    mol = openmm2openff_graph(openmm_top)
    return mol