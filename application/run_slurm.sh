#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.effing@hhi.fraunhofer.de

#         Output (stdout and stderr) of this job will go into a file named with SLURM_JOB_ID (%j) and the job_name (%x)
#SBATCH --output=%j_%x.out

#         Tell slurm to run the job on a single node. Always use this statement.
#SBATCH --nodes=1

#         Ask slurm to run at most 1 task (slurm task == OS process). A task (process) might launch subprocesses/threads.
#SBATCH --ntasks=1

#         Max number of cpus per process (threads/subprocesses) is 16. Seems reasonable with 4 GPUs on a 64 core machine.
#SBATCH --cpus-per-task=16

#         Request from the generic resources 2 GPU 
#SBATCH --gpus=1

#         Request RAM for your job
#SBATCH --mem=8G

#####################################################################################

# This included file contains the definition for $LOCAL_JOB_DIR to be used locally on the node.
source "/etc/slurm/local_job_dir.sh"

echo "Welcome on node ${hostname}"

# Launch the apptainer image with --nv for nvidia support. Two bind mounts are used: 
# - One for the ImageNet dataset and 
# - One for the results (e.g. checkpoint data that you may store in $LOCAL_JOB_DIR on the node
# - One for the Llama model

apptainer exec --nv \
		--bind $DATAPOOL1/datasets/ImageNet-complete:/mnt/dataset \
		--bind $DATA_CLUSTER/users/effing/Llama-2-7b-hf:/mnt/models/Llama-2-7b-hf \
		--bind ${LOCAL_JOB_DIR}:/mnt/output \
		"./application.sif" \
		bash 

# This command copies all results generated in $LOCAL_JOB_DIR back to the submit folder regarding the job id.
cp -r ${LOCAL_JOB_DIR} ${SLURM_SUBMIT_DIR}/${SLURM_JOB_ID}
