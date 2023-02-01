# g09-to-pdb 1.0.0

## General Information
- this is a package to convert structures from g09 logfiles to gromacs convertible pdb files
- ASE is used to read output from g09 logfiles

## Installation
- `git clone https://github.com/hits-mbm-dev/g09-to-pdb.git`
- `cd g09-to-pdb`
- `pip install -e .`

## Usage

### Python
```
from g09topdb import convert_log_pdb

convert_log_pdb("<dataset directory>","<reference FF path>","<output directory>")
```

