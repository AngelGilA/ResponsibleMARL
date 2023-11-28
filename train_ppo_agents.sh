#!/bin/bash
#set job requirements
#SBATCH --job-name="marl_ppo_agents"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=rome
#SBATCH --time=03:00:00
#SBATCH --output=test_case5_ppo_m3_%j.out
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
source activate MARL2023paper_env
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
time srun python -u test.py -d "$TMPDIR"/evds_output_dir -ns 5_000 -ev 100 -n test_case5 -c 5 -a "ppo" -bs 32 -m 3 -s $i -u 4 -lr 0.003 -g 0.95 -nl 3 -ma 'capa' -en 0.01 -ep 0.2 -l 0.95
echo "Done"

#Copy output directory from scratch to home
echo "copy output to home dir"
srun cp -r "$TMPDIR"/evds_output_dir/result $HOME/marl/
