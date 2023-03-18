#!/usr/bin/env python

# Written by GPT-4 in response to the prompt, slightly modified:
# I want to run some hyperparameter sensitivity experiments on a machine learning model. I am planning to read a number of hyperparameter settings from a tsv file. Then I want to run each setting as a SLURM job. I want to wait until all jobs finish. Then I want to read the results of each run and output them as a tsv file. Can you write some python code for this?

import os
import sys
import csv
import time
import subprocess

# Read hyperparameter settings from a TSV file
hyperparameter_settings_file = sys.argv[1]
hyperparameter_settings = []

with open(hyperparameter_settings_file, "r") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        hyperparameter_settings.append(row)

# Submit SLURM jobs for each hyperparameter setting
job_ids = []
slurm_script = "calvin_supervised_training.py"

for i, setting in enumerate(hyperparameter_settings):
    sbatch_options = f"""#!/bin/bash
#SBATCH --job-name=ml_job_{i}
#SBATCH --output=ml_job_output_{i}.txt
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
"""

    sbatch_script = f"job_{i}.sbatch"
    with open(sbatch_script, "w") as f:
        f.write(sbatch_options)
        f.write(f"python {slurm_script} ")
        f.write(' '.join([f'--{key} "{value}"' for key, value in setting.items()]))
        f.write('\n')

    submit_cmd = f"sbatch {sbatch_script}"
    submit_result = subprocess.run(submit_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Get the job ID from the sbatch output
    job_id = int(submit_result.stdout.strip().split()[-1])
    job_ids.append(job_id)

    print(f"Submitted job {job_id} with settings: {setting}")

# Wait for all jobs to complete
print("Waiting for all jobs to complete...")
job_ids_left = job_ids.copy()
while job_ids_left:
    for job_id in job_ids_left:
        status_cmd = f"squeue -j {job_id} -o %t"
        status_result = subprocess.run(status_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in status_result.stdout.splitlines():
            status = line.strip()
        if status not in ("R", "PD", "CG", "S"):
            job_ids_left.remove(job_id)
    time.sleep(10)

print("All jobs completed.")

# Collect and save results as a TSV file
results_file = "results.tsv"

with open(results_file, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(list(hyperparameter_settings[0].keys()))

    for i, setting in enumerate(hyperparameter_settings):
        output_file = f"ml_job_output_{i}.txt"
        result = "N/A"

        if os.path.exists(output_file):
            with open(output_file, "r") as f_out:
                for line in f_out:
                    result = line.strip()

        try:
            setting['val_acc'] = float(result)
        except ValueError:
            setting['val_acc'] = -1
        setting['job_id'] = job_ids[i]
        writer.writerow(list(setting.values()))

print(f"Results saved to {results_file}.")
