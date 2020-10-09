#! bin/bash

python3 setup.py install
mv build/*/*.so .
mv build/*/*.o .
