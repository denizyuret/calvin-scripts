#!/usr/bin/env python

# Written by GPT-4 in response to the prompt, slightly modified:
# I want to run some hyperparameter sensitivity experiments on a machine learning model. I am planning to read a number of hyperparameter settings from a tsv file. Then I want to run each setting as a SLURM job. I want to wait until all jobs finish. Then I want to read the results of each run and output them as a tsv file. Can you write some python code for this?

import os
import re
import sys
import csv
import time
import shutil
import subprocess
import logging
from logging import info
logging.basicConfig(level=logging.INFO)

def job_status(job_id):
    status_cmd = f"squeue -j {job_id} -o %t"
    status_run = subprocess.run(status_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return status_run.stdout.splitlines()[-1].strip()


def set_val_acc(setting):
    setting['val_acc'] = -1
    setting['step'] = -1
    ckpt_dir = f"lightning_logs/version_{setting['job_id']}/checkpoints"
    if not os.path.isdir(ckpt_dir):
        return
    for f in os.listdir(ckpt_dir):
        m = re.match(r"epoch=(\d+)-step=(\d+)\.ckpt", f)
        if m is None:
            continue
        eval_cmd = f"eval.py {ckpt_dir}/{f} -c {setting['context_length']} -f '{setting['features']}'"
        eval_run = subprocess.run(eval_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        eval_last_line = eval_run.stdout.splitlines()[-1]
        try:
            val_acc = float(eval_last_line)
        except ValueError:
            val_acc = -1
        if val_acc > setting['val_acc']:
            setting['val_acc'] = val_acc
            setting['step'] = m.group(2)
            


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
#SBATCH --job-name={i:03}
#SBATCH --output={i:03}_%j.out
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
"""
    sbatch_script = f"{i:03}.sbatch"
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
    # Make batch file unique to preserve
    shutil.copy(sbatch_script, f"{i:03}_{job_id}.sbatch")

    info(f"Submitted job {job_id} with settings: {setting}")

# Collect and save results as a TSV file
results_file = f"{job_ids[0]}-{job_ids[-1]}.tsv"
job_ids_left = job_ids.copy()

with open(results_file, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(list(hyperparameter_settings[0].keys()))
    info(f"Waiting for {len(job_ids_left)} jobs to finish...")
    while job_ids_left:
        for job_id in job_ids_left:
            status = job_status(job_id)
            if status not in ("R", "PD", "CG", "S"):
                setting_id = job_ids.index(job_id)
                setting = hyperparameter_settings[setting_id]
                setting['job_id'] = job_id
                set_val_acc(setting)
                writer.writerow(list(setting.values()))
                f.flush()
                job_ids_left.remove(job_id)
                info(f"{setting_id:03} finished: job_id={job_id} status={status} val_acc={setting['val_acc']} step={setting['step']}; {len(job_ids_left)} left")
        time.sleep(10)

info(f"Saved results to {results_file}")
