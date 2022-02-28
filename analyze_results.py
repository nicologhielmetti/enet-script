import hls4ml
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tarfile
import shutil

PARSE = False
data = []
data_path = 'data_pickles/data6.pkl'
saved_dir = os.getcwd()
if PARSE:
    df = pd.read_pickle(data_path)
    os.chdir('/eos/home-n/nghielme/')
    ids = df['ID'].tolist()
    for dir in os.listdir('.'):
        if dir.startswith('enet-results-run'):
            os.chdir(dir)
        else:
            continue
        for model in os.listdir('.'):
            datum = {}
            if model.endswith('.tar.gz') and model[8:-7] not in ids:
                with tarfile.open(model) as tar:
                    subdir_and_files = [
                        tarinfo for tarinfo in tar.getmembers()
                        if tarinfo.name.startswith('hls')
                    ]
                    tar.extractall(members=subdir_and_files)
            else:
                continue

            model = model[8:-7]
            parsed = hls4ml.report.vivado_report.parse_vivado_report(model + '_FIFO_OPT')
            shutil.rmtree(model + '_FIFO_OPT')
            model_info = model.split('_')
            datum['ID'] = model
            datum['Run'] = dir.split('-')[-1]
            datum['Filters'] = int(model_info[1][1:])
            datum['Clock'] = int(model_info[2][3:])
            datum['ReuseFactor'] = int(model_info[3][2:])
            datum['Model'] = 'Clock: ' + str(datum['Clock']) + ' \n RF: ' + str(datum['ReuseFactor'])
            datum['Quantization'] = int(model_info[4][1:])
            datum['Precision'] = model_info[7].replace('-', ',')
            try:
                datum['LUTs%'] = int(round(parsed['ImplementationReport']['TotLUTs%']))
                datum['FFs%'] = int(round(parsed['ImplementationReport']['FFs%']))
                datum['RAM36Bs%'] = int(round(parsed['ImplementationReport']['RAMB36s%']))
                datum['RAM18s%'] = int(round(parsed['ImplementationReport']['RAMB18s%']))
                datum['DSPs%'] = int(round(parsed['ImplementationReport']['DSPs%']))
                datum['WNS'] = parsed['TimingReport']['WNS']
            except KeyError:
                datum['LUTs%'] = 'NA'
                datum['FFs%'] = 'NA'
                datum['RAM36Bs%'] = 'NA'
                datum['RAM18s%'] = 'NA'
                datum['DSPs%'] = 'NA'
                datum['WNS'] = 'NA'
            datum['MaxLatency'] = parsed['CosimReport']['LatencyMax']
            data.append(datum)
        os.chdir('..')
    os.chdir(saved_dir)
    df1 = pd.DataFrame(data)
    list_df = [df, df1]
    res = df.concat(list_df)
    res.to_pickle(data_path)
else:
    df = pd.read_pickle(data_path)

df_na = df[df['LUTs%'] == 'NA']
df_na.to_csv('NA_models.csv')
df = df[df['LUTs%'] != 'NA']

df['Max Latency [ms]'] = df['MaxLatency'] * 1e-5
df['10 x WNS [ns]'] = df['WNS'] * 10
df['Latency Overclock [ms]'] = df['MaxLatency'] * (10 - df['WNS']) * 1e-6
# df.to_csv('dataframe.csv')
ap_fixed_16_6_data = df[df['Precision'] == '16,6']
ap_fixed_8_4_data = df[df['Precision'] == '8,4']

ap_fixed_8_4_data = ap_fixed_8_4_data.sort_values(by=['Clock', 'ReuseFactor'], ascending=True)
ap_fixed_16_6_data = ap_fixed_16_6_data.sort_values(by=['Clock', 'ReuseFactor'], ascending=True)


def print_plot(data, title):
    def pointplot_with_outliers(*args, **kwargs):
        local_data = kwargs.pop('data')
        gt100ms = local_data.copy()
        gt100ms.loc[gt100ms['Max Latency [ms]'] >= 100, 'Max Latency [ms]'] = 100
        gt100ms[['LUTs%', 'FFs%', 'RAM36Bs%', 'RAM18s%', 'DSPs%', '10 x WNS [ns]', 'Latency Overclock [ms]']] = -10
        lt100ms = local_data.copy()
        lt100ms.loc[lt100ms['Max Latency [ms]'] >= 100, 'Max Latency [ms]'] = -10
        gt100ms = gt100ms.melt(id_vars=['Model', 'ReuseFactor', 'Clock', 'Filters', 'Quantization'],
                               value_vars=['LUTs%', 'FFs%', 'RAM36Bs%', 'RAM18s%', 'DSPs%',
                                           'Max Latency [ms]', '10 x WNS [ns]', 'Latency Overclock [ms]'])
        lt100ms = lt100ms.melt(id_vars=['Model', 'ReuseFactor', 'Clock', 'Filters', 'Quantization'],
                               value_vars=['LUTs%', 'FFs%', 'RAM36Bs%', 'RAM18s%', 'DSPs%',
                                           'Max Latency [ms]', '10 x WNS [ns]', 'Latency Overclock [ms]'])
        palette = kwargs['palette']
        if len(gt100ms) > 0:
            kwargs['palette'] = 'dark:brown'
            sns.pointplot(**kwargs, data=gt100ms, markers='x')
            kwargs['palette'] = palette

        sns.pointplot(**kwargs, data=lt100ms)

    sns.set_theme()

    g = sns.FacetGrid(data, col='Filters', row='Quantization', sharex=False, sharey=False, aspect=3.2,
                      ylim=(0, 110))
    g.map_dataframe(pointplot_with_outliers, join=False, x='Model', y='value', hue='variable', palette='tab10')
    g.add_legend()
    g.set_xticklabels(rotation=45)
    g.fig.suptitle(title)
    plt.show()


print_plot(ap_fixed_8_4_data, 'Default Quantization: ap_fixed<8,4>')
print_plot(ap_fixed_16_6_data, 'Default Quantization: ap_fixed<16,6>')
