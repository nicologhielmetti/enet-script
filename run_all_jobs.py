import os
os.chdir('jobs_run2')
jobs = os.listdir('.')
for job in jobs:
    os.system('nomad job run -detach \'' + job + '\'' )