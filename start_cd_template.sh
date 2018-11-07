#!/bin/sh

python3 SUN360/create_dataset_v1.py
python3 models/SUN360Dataset.py 1
python3 models/extractFC7.py 1