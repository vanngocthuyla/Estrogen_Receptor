import os
import argparse
import jax
import numpyro

from _main import _main

parser = argparse.ArgumentParser()

parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--niters",                        type=int,               default=1000)
parser.add_argument( "--nburn",                         type=int,               default=200)
parser.add_argument( "--nchains",                       type=int,               default=4)

args = parser.parse_args()

jax.config.update("jax_enable_x64", True)
numpyro.set_host_device_count(4)

_main(dir=args.out_dir, receptors=("ER_alpha",), num_warmup=args.nburn, num_samples=args.niters, num_chains=args.nchains)
