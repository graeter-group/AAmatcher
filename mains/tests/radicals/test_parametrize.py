#%%
from PDBData.PDBMolecule import PDBMolecule
from openmm.app import PDBFile, ForceField
#%%

rad = PDBMolecule.from_pdb("F_rad.pdb")

topology = rad.to_openmm().topology
# %%
forcefield=ForceField('amber99sbildn.xml')
# sys = forcefield.createSystem(topology=topology)
# %%

def add_residues(forcefield, topology):
    [templates, residues] = forcefield.generateTemplatesForUnmatchedResidues(topology)
    for t_idx, template in enumerate(templates):
        ref_template = forcefield._templates[template.name]
        # the atom names stored in the template of the residue
        ref_names = [a.name for a in ref_template.atoms]

        # delete the H indices since the Hs are not distinguishable anyways
        # in the template, the Hs are named HB2 and HB3 while in some pdb files they are named HB1 and HB2

        for i, n in enumerate(ref_names):
            if n[0] == 'H' and len(n) == 3:
                ref_names[i] = n[:-1]

        for atom in template.atoms:
            name = atom.name
            if name[0] == 'H' and len(name) == 3:
                name = name[:-1]

            # find the atom with that name in the reference template
            atom.type = ref_template.atoms[ref_names.index(name)].type
            print(atom.name, atom.type)

        # create a new template
        template.name = template.name + str(t_idx)
        forcefield.registerResidueTemplate(template)

    return forcefield

#%%
forcefield = add_residues(forcefield, topology)
sys = forcefield.createSystem(topology=topology)

# %%
rad = PDBMolecule.from_pdb("F_rad.pdb")
rad.parametrize(forcefield=ForceField('amber99sbildn.xml'), allow_radicals=True)
# %%
forcefield=ForceField('amber99sbildn.xml')
[templates, residues] = forcefield.generateTemplatesForUnmatchedResidues(topology)
for t_idx, template in enumerate(templates):
    ref_template = forcefield._templates[template.name]
    for a in template.atoms:
        print(a.bondedTo)
        break
# %%
