import os

import pandas
import yaml

# generate combinations
import itertools
import pandas as pd

for config_file in os.listdir('explore_run2_scripts'):
    with open('explore_run2_scripts/' + config_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    expl_config = config['explore']
    expl_list = [v for k,v in expl_config.items()]
    combinations = list(itertools.product(*expl_list))

    with open('job.hcl') as f:
        lines = f.read()

    for c in combinations:
        job = lines.format(reuse_factor=c[0], n_filters=c[1], clock_period=c[2], quantization=c[3], precision=c[4],
                           input_data=config['simulation']['input_data'],
                           output_predictions=config['simulation']['output_predictions'])
        f = open('jobs_run2_stop/job_f{}_clk{}_rf{}_q{}_{}.hcl'.format(c[1], c[2], c[0], c[3], c[4]), "w")
        f.write(job)
        f.close()