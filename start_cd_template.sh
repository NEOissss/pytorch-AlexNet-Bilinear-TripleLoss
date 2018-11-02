#!/bin/sh

python3 SUN360/create_dataset_v1.py
python3 network/SUN360Dataset.py 1
python3 network/extractFC7.py 1