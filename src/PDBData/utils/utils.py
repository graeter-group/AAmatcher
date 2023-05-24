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



