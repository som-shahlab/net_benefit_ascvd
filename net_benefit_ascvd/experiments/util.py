import subprocess
import time
import numpy as np
import random
import itertools
from sklearn.model_selection import ParameterGrid


def run_commands(commands_list, max_concurrent_jobs=4):
    """
    Runs a list of shell jobs concurrently using the subprocess.Popen interface
    Arguments:
        commands_list: a list where each element is a list of args to subprocess.Popen
        max_concurrent_jobs: maximum number of jobs to run at once
    """
    max_iterations = len(commands_list)
    if max_concurrent_jobs > max_iterations:
        max_concurrent_jobs = max_iterations
    i = 0
    current_jobs_running = 0
    completed_jobs = 0

    job_dict = {}
    running_job_ids = []

    while completed_jobs < max_iterations:
        # Start jobs until reaching maximum number of concurrent jobs
        while (current_jobs_running < max_concurrent_jobs) and (i < max_iterations):
            print("Starting job {}".format(i))
            job_dict[i] = subprocess.Popen(commands_list[i])
            running_job_ids.append(i)
            current_jobs_running += 1
            i += 1

        # Check if jobs are done
        time.sleep(5)
        still_running_job_ids = []
        for j in running_job_ids:
            if job_dict[j].poll() is None:
                still_running_job_ids.append(j)
            else:
                job_dict[j].wait()
                print("Job {} complete".format(j))
                del job_dict[j]
                current_jobs_running -= 1
                completed_jobs += 1
        running_job_ids = still_running_job_ids


def flatten_multicolumns(df):
    """
    Converts multi-index columns into single column
    """
    df.columns = [
        "_".join([el for el in col if el != ""]).strip()
        for col in df.columns.values
        if len(col) > 1
    ]
    return df


def generate_grid(
    global_tuning_params_dict,
    model_tuning_params_dict=None,
    experiment_params_dict=None,
    grid_size=None,
    seed=None,
):

    the_grid = list(ParameterGrid(global_tuning_params_dict))
    if model_tuning_params_dict is not None:
        local_grid = []
        for i, pair_of_grids in enumerate(
            itertools.product(the_grid, list(ParameterGrid(model_tuning_params_dict)))
        ):
            local_grid.append({**pair_of_grids[0], **pair_of_grids[1]})
        the_grid = local_grid

    if grid_size is not None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        np.random.shuffle(the_grid)
        the_grid = the_grid[:grid_size]

    if experiment_params_dict is not None:
        outer_grid = list(ParameterGrid(experiment_params_dict))
        final_grid = []
        for i, pair_of_grids in enumerate(itertools.product(outer_grid, the_grid)):
            final_grid.append({**pair_of_grids[0], **pair_of_grids[1]})
        return final_grid
    else:
        return the_grid
