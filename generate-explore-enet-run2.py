import pandas
import yaml

# generate combinations
import itertools
import pandas as pd

with open(r'explore_run2.yml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

df = pd.read_csv('dataframe.csv')
df = df[df['MaxLatency'] == df['MaxLatency'].min()][['Filters', 'Quantization', 'Precision']].drop_duplicates()

expl_config = config['explore']
expl_list = [v for k,v in expl_config.items()]
combinations = list(itertools.product(*expl_list))

with open('job.hcl') as f:
    lines = f.read()

for _, row in df.iterrows():
    for c in combinations:
        reuse_factor = c[0]
        n_filters = row['Filters']
        clock_period = c[1]
        quantization = row['Quantization']
        precision = 'ap_fixed<{}>'.format(row['Precision'])
        job = lines.format(reuse_factor=reuse_factor, n_filters=n_filters, clock_period=clock_period,
                           quantization=quantization, precision=precision,
                           input_data=config['simulation']['input_data'],
                           output_predictions=config['simulation']['output_predictions'])
        f = open('jobs_run2/job_f{}_clk{}_rf{}_q{}_{}.hcl'.format(n_filters, clock_period, reuse_factor, quantization,
                                                                  precision), "w")
        f.write(job)
        f.close()