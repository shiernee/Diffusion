#!/bin/bash

python3.7 main_generate_pt.py
python3.7 main_generate_initial_condition.py
python3.7 main_set_parameter.py

sleep 1
cd ../test/
python3.7 Testing.py

