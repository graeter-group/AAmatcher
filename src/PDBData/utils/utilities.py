import numpy as np
import dgl
import torch

def invert_permutation(p):
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s


def write_reference_data(g:dgl.graph, energies=None, gradients=None, class_ff:str="_amber99sbildn", ref_suffix:str="_ref")->dgl.graph:

        if not energies is None:
            g.nodes["g"].data["u"+ref_suffix] = energies - g.nodes["g"].data[f"u_nonbonded{class_ff}"]

        if not gradients is None:
            g.nodes["n1"].data["grad"+ref_suffix] = gradients - g.nodes["n1"].data[f"grad_nonbonded{class_ff}"]

        g.nodes["n4"].data["use_k"] = torch.where(torch.abs(g.nodes["n4"].data[f"k{class_ff}"]) > 0, 1., 0.)
        g.nodes["n4_improper"].data["use_k"] = torch.where(torch.abs(g.nodes["n4_improper"].data[f"k{class_ff}"]) > 0, 1., 0.)
        
        return g


def get_oneletter(threeletter):
    return get_oneletter.d[threeletter]

get_oneletter.d = {
        "ALA":"A", "ARG":"R", "ASN":"N", "ASP":"D", "CYS":"C", "GLN":"Q", "GLU":"E", "GLY":"G", "HIS":"H", "ILE":"I", "LEU":"L", "LYS":"K", "MET":"M", "PHE":"F", "PRO":"P", "SER":"S", "THR":"T", "TRP":"W", "TYR":"Y", "VAL":"V"
    }