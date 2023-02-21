import os
from pathlib import Path

from g09topdb import read_rtp, seq_from_filename, generate_radical_reference,read_g09, match_mol, write_trjtopdb

print(os.getcwd())

dataset = Path("sw/g09-to-pdb/example/")
FF_rtp = Path("sw/g09-to-pdb/example/amber99sb-star-ildnp.ff/aminoacids.rtp")
targetdir = Path("workdir/scripts/test_g09topdb")

AAs_reference = read_rtp(FF_rtp)
filenames = dataset.glob("*rad*.log")
for filename in filenames:
    seq = seq_from_filename(filename, AAs_reference)
    print(seq)
    for res in seq:
        ressplit = res.split(sep='_')
        if len(ressplit) > 1:
            if ressplit[1] == 'R':
                AAs_reference = generate_radical_reference(AAs_reference,ressplit[0],'CA')
    print(AAs_reference['GLY'])
    print(AAs_reference["GLY_R"])
    mol, trj = read_g09(filename, seq, AAs_reference)
    print(mol)

    atom_order = match_mol(mol, AAs_reference, seq)
    outfile = targetdir / f"{filename.stem}_test.pdb"
    write_trjtopdb(outfile,trj, atom_order, seq, AAs_reference)
    