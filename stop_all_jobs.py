import os
os.chdir('jobs')
jobs = os.listdir('.')
for job in jobs:
    os.system('nomad job stop -purge -yes \'' + job.replace('job', 'scan').replace('.hcl', '') + '\'' )