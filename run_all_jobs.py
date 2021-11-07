import os
os.chdir('jobs')
jobs = os.listdir('.')
for job in jobs:
    os.system('nomad job run -detach ' + job )