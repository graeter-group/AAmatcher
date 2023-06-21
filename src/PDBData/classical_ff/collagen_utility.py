from openmm.app.topology import Topology


def add_bonds(top:Topology):
    """
    For HYP and DOP, adds bonds to the topology if they are not already present.
    Assumes that external bonds are always from N to C!
    Note that OH1 and OH2 is false, one of them is supposed to be OZ. This is an error in our naming convention, but we keep it for backwards compatibility.
    """
    for r in top.residues():
        if r.name in _add_bonds_for_res.bond_dict.keys():
            name = r.name
            _add_bonds_for_res(r,top,name)
    return top


def add_external_bonds(r, top):
        # check whether external bonds are present
        is_start = r.index == 0
        is_end = r.index == top.getNumResidues() - 1
        ext_bonds = list(r.external_bonds())
        search_ext_bonds = False
        two_needed = not (is_start or is_end)
        one_needed = (not (is_start and is_end)) and not two_needed

        if two_needed and len(ext_bonds)<2:
            search_ext_bonds = True

        elif one_needed and len(ext_bonds)<1:
            search_ext_bonds = True
        
        prev_res_needed = is_start or two_needed
        next_res_needed = is_end or two_needed

        if not search_ext_bonds:
            return
        else:
            prev_res = None
            next_res = None
            for res in top.residues():
                if res.index == r.index - 1:
                    prev_res = res
                if res.index == r.index + 1:
                    next_res = res

                # break condition
                if (not prev_res_needed) or prev_res is not None:
                    if (not next_res_needed) or next_res is not None:
                        break

            # find the N in the residue before
            prev_C = None
            next_N = None
            if prev_res_needed:
                assert prev_res is not None
                for a in prev_res.atoms():
                    if a.name == "C":
                        prev_C = a
                        break
                assert prev_C is not None

            if next_res_needed:
                assert next_res is not None
                for a in next_res.atoms():
                    if a.name == "N":
                        next_N = a
                        break
                assert next_N is not None
            
            existing_ext_bonds = [[b[0].index, b[1].index] for b in ext_bonds]
            # flatten:
            existing_ext_bonds = [i for sublist in existing_ext_bonds for i in sublist]

            # add the external bond to the C in the previous residue:
            if prev_C is not None:
                if not prev_C.index in existing_ext_bonds:
                    # find N in r:
                    r_N = None
                    for a in r.atoms():
                        if a.name == "N":
                            r_N = a
                            break
                    assert r_N is not None
                    top.addBond(prev_C, r_N)

            # same thing for N:
            if next_N is not None:
                if not next_N.index in existing_ext_bonds:
                    # find C in r:
                    r_C = None
                    for a in r.atoms():
                        if a.name == "C":
                            r_C = a
                            break
                    assert r_C is not None
                    top.addBond(r_C, next_N)

#%%
def _add_bonds_for_res(r,top,name):
    bond_list = _add_bonds_for_res.bond_dict[name]

    atom_names = [a.name for a in r.atoms()]

    ref_names = set([a for b in bond_list for a in b])

    if not set(atom_names) == ref_names:
        if name+"2" in _add_bonds_for_res.bond_dict.keys():
            return _add_bonds_for_res(r,top,name+"2")
        else:
            raise RuntimeError(f"Could not match atoms with those of the bond definitions in residue {name}.\nIn bond definitions but not in topology: {(ref_names - set(atom_names))}\nIn topology but not bond definitions: {(set(atom_names)-ref_names)}.\nThis function only supports standard versions of the variants {list(_add_bonds_for_res.bond_dict.keys())}.")
    atoms = [a for a in r.atoms()]
    # existing_bonds are both permutations of existing bonds:
    existing_bonds = [[b[0].name, b[1].name] for b in r.internal_bonds()]
    existing_bonds += [[b[1].name, b[0].name] for b in r.external_bonds()]

    for b in bond_list:
        if b in existing_bonds:
            continue
        try:
            a1 = atoms[atom_names.index(b[0])]
            a2 = atoms[atom_names.index(b[1])]
        except ValueError as e:
            raise RuntimeError(f"Could not find atoms {b[0]} or {b[1]} in residue {r.name}.\nThis function only supports standard versions of the variants {list(_add_bonds_for_res.bond_dict.keys())}.") from e
        top.addBond(a1, a2)

    add_external_bonds(r, top)
    return
#%%

# assumes that external bonds are always from N to C!
# note that OH1 and OH2 is false, one of them is supposed to be OZ. This is an error in pdb convention, but we keep it for backwards compatibility
_add_bonds_for_res.bond_dict = {
    "DOP":[
        ["N","H"],
        ["N","CA"],
        ["CA","HA"],
        ["CA","CB"],
        ["CA","C"],
        ["CB","HB1"],
        ["CB","HB2"],
        ["CB","CG"],
        ["CG","CD1"],
        ["CG","CD2"],
        ["CD1","HD1"],
        ["CD1","CE1"],
        ["CE1","HE1"],
        ["CE1","CZ"],
        ["CZ","OH1"],
        ["CZ","CE2"],
        ["OH1","HH1"],
        ["CE2","OH2"],
        ["CE2","CD2"],
        ["OH2","HH2"],
        ["CD2","HD2"],
        ["C","O"]
    ],
    "DOP2":[
        ["N","H"],
        ["N","CA"],
        ["CA","HA"],
        ["CA","CB"],
        ["CA","C"],
        ["CB","HB1"],
        ["CB","HB2"],
        ["CB","CG"],
        ["CG","CD1"],
        ["CG","CD2"],
        ["CD1","HD1"],
        ["CD1","CE1"],
        ["CE1","OH1"],
        ["OH1","HH1"],
        ["CE1","CZ"],
        ["CZ","OH2"],
        ["CZ","CE2"],
        ["OH2","HH2"],
        ["CE2","HE2"],
        ["CE2","CD2"],
        ["CD2","HD2"],
        ["C","O"]
    ],
    "HYP":[
        ["N","CD2"],
        ["N","CA"],
        ["CD2","HD21"],
        ["CD2","HD22"],
        ["CD2","CG"],
        ["CG","HG"],
        ["CG","OD1"],
        ["CG","CB"],
        ["OD1","HD1"],
        ["CB","HB1"],
        ["CB","HB2"],
        ["CB","CA"],
        ["CA","HA"],
        ["CA","C"],
        ["C","O"]
    ],

}