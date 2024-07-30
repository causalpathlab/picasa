#!/bin/bash

python znode/$1/4_attncl_train.py
python znode/$1/4_attncl_analyze.py
python znode/$1/4_attncl_attncontx.py
