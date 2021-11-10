import os
import json


def save_results(dead_job_id):
    dead_job_id_tarname = dead_job_id.replace(',', '-').replace('<', '_').replace('>', '_')
    os.system('nomad alloc fs -job \'{}\' \'synth_scan/local/enet-script/{}.tar.gz\' > '
              '/eos/user/n/nghielme/enet-results/{}.tar.gz '
              .format(dead_job_id, dead_job_id_tarname, dead_job_id_tarname))


os.system('nomad job status | grep \'dead\' | grep -v \'dead (stopped)\' > dead_jobs_new.txt')
dead_jobs_new = list()
dead_jobs_all = json.load(open('dead_jobs.json', ))
with open('dead_jobs_new.txt', 'r') as dead_jobs_file:
    for dead_job in dead_jobs_file.readlines():
        dead_job_id = dead_job.split(' ')[0]
        if dead_job_id not in dead_jobs_all:
            dead_jobs_all.append(dead_job.split(' ')[0])
            save_results(dead_job_id)

with open('dead_jobs.json', 'w') as dead_jobs_all_file:
    json.dump(dead_jobs_all, dead_jobs_all_file, indent=4)
