import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument( "--out_dir",                       type=str,               default="")
parser.add_argument( "--niters",                        type=int,               default=1000)
parser.add_argument( "--nburn",                         type=int,               default=200)
parser.add_argument( "--nchains",                       type=int,               default=4)

args = parser.parse_args()

running_script = "/home/vla/python/ER/scripts/run_ER_alpha.py"
file_name = "ER_alpha"

qsub_file = os.path.join(args.out_dir, f'{file_name}.job')
log_file = os.path.join(args.out_dir, f'{file_name}.log')
        
qsub_script = '''#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --ntasks=4
#SBATCH --mem=4096M
#SBATCH -o %s ''' %log_file + '''
#SBATCH -e %s ''' %log_file + '''

module load miniconda/3
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate gci \n
cd ''' + args.out_dir + '''\n''' + \
        '''date\n''' + \
        '''python ''' + running_script + \
        ''' --out_dir ''' + args.out_dir + \
        ''' --niters %d '''%args.niters + \
        ''' --nburn %d '''%args.nburn + \
        ''' --nchains %d ''' %args.nchains + \
        '''\ndate\n'''

print("Writing qsub file", qsub_file)
open(qsub_file, "w").write(qsub_script)
# os.system("sbatch %s"%qsub_file)