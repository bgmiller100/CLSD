#! bin/bash

module load gcc/7.1.0
module load gsl/2.6
module load python3/3.6.1
source env/bin/activate
python3 setupmpi.py install --user
deactivate
mv build/*/*.so .
mv build/*/*.o .
