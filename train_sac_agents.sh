#!/bin/bash
#set job requirements
#SBATCH --job-name="marl_sac_agents"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=rome
#SBATCH --time=06:00:00
#SBATCH --output=test_case5_sacd_m3_%j.out
#SBATCH --array=0-4

trained_model=

# function to handle the SIGTERM signal
function handle_interrupt {
    echo "Caught SIGTERM signal, copying output directory from scratch to home..."
    cp -r "$TMPDIR"/evds_output_dir/result $HOME/marl/
    exit 1
}

# register the signal handler
trap handle_interrupt TERM

echo "Activate envirnonment"
source activate l2rpn2023_env
export PYTHONPATH=$PYTHONPATH:$PWD

#Create output directory on scratch
mkdir "$TMPDIR"/evds_output_dir
srun cp -r $HOME/marl4powergridtopo/data "$TMPDIR"/evds_output_dir/data
# copy trained model to continue training
if (
    	[[ ! -z $trained_model ]]
)
then
    	echo " * copy trained model to  $TMPDIR/evds_output_dir/$trained_model * "
        mkdir "$TMPDIR"/evds_output_dir/$trained_model
        srun cp -r $HOME/marl/result/$trained_model "$TMPDIR"/evds_output_dir/
else
        echo " * no model to copy * "
fi

i=${SLURM_ARRAY_TASK_ID}
echo "Run code: Task id $i"
time srun python -u test.py -d "$TMPDIR"/evds_output_dir -ns 5_000 -ev 100 -n test_case5 -c 5 -a "sacd" -bs 64 -m 3 -s $i -u 4 -tu 1 -nl 3 -lr 5e-5 -ma 'capa' -g 0.995 --tau 0.001 -te 0.98
echo "Done"

#Copy output directory from scratch to home
echo "copy output to home dir"
srun cp -r "$TMPDIR"/evds_output_dir/result $HOME/marl/
