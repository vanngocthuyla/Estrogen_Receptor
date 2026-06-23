#!/bin/bash

export SCRIPT='/home/vla/python/ER/scripts/submit_ER_alpha.py'

export OUT_DIR='/home/vla/python/ER/fitting'

export NITERS=5000

export NBURNS=2000

python $SCRIPT --out_dir $OUT_DIR --niters $NITERS --nburn $NBURNS