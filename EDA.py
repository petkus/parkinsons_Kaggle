import numpy as np 
import pandas as pd 
import seaborn as sns
import os
import matplotlib.pyplot as plt
import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

def plot_velocity_data(file_path):
    name = os.path.basename(file_path)
    if name[-4:] == '.csv':
        df = pd.read_csv(file_path)
    else: #parquet
        df = pd.read_parquet(file_path)

    plt.figure(figsize=(9,6))
    plt.title(f"{name}")
    
    alpha = .2

    if 'Walking' in df.columns: 
        first = True
        bounds = _get_bounds(df.loc[df.Walking == True].Time.to_numpy())
        for left, right in bounds:
            if first:
                plt.axvspan(left, right, alpha=alpha, color='green', label='walking')
                first = False
            else:
                plt.axvspan(left, right, alpha=alpha, color='green')

    if 'Turn' in df.columns: 
        first = True
        bounds = _get_bounds(df.loc[df.Turn == True].Time.to_numpy())
        for left, right in bounds:
            if first:
                plt.axvspan(left, right, alpha=alpha, color='blue', label='turning')
                first = False
            else:
                plt.axvspan(left, right, alpha=alpha, color='blue')

    if 'StartHesitation' in df.columns: 
        first = True
        bounds = _get_bounds(df.loc[df.StartHesitation == True].Time.to_numpy())
        for left, right in bounds:
            if first:
                plt.axvspan(left, right, alpha=alpha, color='red', label='start hesitation')
                first = False
            else:
                plt.axvspan(left, right, alpha=alpha, color='red')

    if 'Task' in df.columns: 
        first = True
        bounds = _get_bounds(df.loc[df.Task == False].Time.to_numpy())
        for left, right in bounds:
            if first:
                plt.axvspan(left, right, alpha=alpha, color='gray', label='unannotated')
                first = False
            else:
                plt.axvspan(left, right, alpha=alpha, color='gray')
    
    if 'Valid' in df.columns: 
        first = True
        bounds = _get_bounds(df.loc[df.Valid == False].Time.to_numpy())
        for left, right in bounds:
            if first:
                plt.axvspan(left, right, alpha=alpha, color='black', label='ambiguous')
                first = False
            else:
                plt.axvspan(left, right, alpha=alpha, color='black')
    
    sns.lineplot(df, x='Time', y='AccV', label = 'vertical')
    sns.lineplot(df, x='Time', y='AccML', label = 'mediolateral')
    sns.lineplot(df, x='Time', y='AccAP', label = 'anteroposterior')
    plt.show()

def _get_bounds(indicators):
    output = []
    left = None
    for i in indicators:
        if left == None:
            left = i
            right = i
        elif right + 1 == i:
            right = i
        else:
            output += [(left, right)]
            left = i
            right = i
    if left != None:
        output += [(left, right)]
    return output
        

def generate_aggregate_statistics():
    df_agg = pd.DataFrame()

    path = os.path.join(config['data_path'], 'train', 'defog')
    print('scanning defog')
    for name in os.listdir(path):
        file_path = os.path.join(path, name)
        _df = _generate_statistics(file_path)
        df_agg = pd.concat([df_agg, _df])

    path = os.path.join(config['data_path'], 'train', 'tdcsfog')
    print('scanning tdcsfog')
    for name in os.listdir(path):
        file_path = os.path.join(path, name)
        _df = _generate_statistics(file_path)
        df_agg = pd.concat([df_agg, _df])

    path = os.path.join(config['data_path'], 'train', 'notype')
    print('scanning notype')
    for name in os.listdir(path):
        file_path = os.path.join(path, name)
        _df = _generate_statistics(file_path)
        df_agg = pd.concat([df_agg, _df])
    return df_agg

def _generate_statistics(file_path):
    df = pd.read_csv(file_path)

    df_std = df.std()
    df_mean = df.mean()
    count = df.count().Time

    statistics = {
        'file_name': [os.path.basename(file_path)],
        'AccV_mean': df_mean.AccV,
        'AccML_mean': df_mean.AccML,
        'AccAP_mean': df_mean.AccAP,

        'AccV_std': df_std.AccV,
        'AccML_std': df_std.AccML,
        'AccAP_std': df_std.AccAP,
        
        'count': count,
    }

    if 'StartHesitation' in df.columns:
        statistics['StartHesitation_percentage'] = df.StartHesitation.sum()/count
    else:
        statistics['StartHesitation_percentage'] = np.nan

    if 'Turn' in df.columns:
        statistics['Turn_percentage'] = df.Turn.sum()/count
    else:
        statistics['Turn_percentage'] = np.nan

    if 'Walking' in df.columns:
        statistics['Walking_percentage'] = df.Walking.sum()/count
    else:
        statistics['Walking_percentage'] = np.nan

    if 'Valid' in df.columns:
        statistics['Valid_percentage'] = df.Valid.sum()/count
    else:
        statistics['Valid_percentage'] = np.nan

    if 'Task' in df.columns:
        statistics['Task_percentage'] = df.Task.sum()/count
    else:
        statistics['Task_percentage'] = np.nan
        
    return pd.DataFrame.from_dict(statistics)