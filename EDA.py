import numpy as np 
import pandas as pd 
import seaborn as sns
import os
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

class Meta_Data():
    def __init__(self):
        """
            metadata fields:
            - defog
            - tdcsfog
            - daily
            - subjects
            - tasks
            - events
            - sample_submission
        """

        print("Loading metadata")
        self.daily = pd.read_csv(config['data_path'] + 'daily_metadata.csv')
        self.defog = pd.read_csv(config['data_path'] + 'defog_metadata.csv')
        self.tdcsfog = pd.read_csv(config['data_path'] + 'tdcsfog_metadata.csv')

        self.daily['Source'] = 'daily'
        self.defog['Source'] = 'defog'
        self.tdcsfog['Source'] = 'tdcsfog'

        self.subjects = pd.read_csv(config['data_path'] + 'subjects.csv')
        self.tasks = pd.read_csv(config['data_path'] + 'tasks.csv')

        self.events = pd.read_csv(config['data_path'] + 'events.csv')

        self.sample_submission = pd.read_csv(config['data_path'] + 'sample_submission.csv')

        # Adding Target Totals
        print("Adding totals for subjects")
        self.subjects[['StartHesitation_Total', 'Turn_Total', 'Walking_Total']] = 0

        for subject in tqdm(self.subjects.Subject):

            # Getting all files with subject
            df_subject = pd.concat([
                self.tdcsfog.loc[self.tdcsfog.Subject == subject],
                self.defog.loc[self.defog.Subject == subject],
                self.daily.loc[self.daily.Subject == subject]
            ])

            # Adding up total target values for subject
            for index, row in df_subject.iterrows():
                if row['Source'] != 'daily':
                    path = os.path.join(config['data_path'], 'train', row['Source'], f'{row["Id"]}.csv')
                    if os.path.exists(path):
                        df = pd.read_csv(path)
                        totals = df[['StartHesitation', 'Turn', 'Walking']].sum()
                        self.subjects.loc[self.subjects.Subject == subject, ['StartHesitation_Total', 'Turn_Total', 'Walking_Total']] += totals.values
            self.subjects.loc[self.subjects.Subject == subject, ['StartHesitation_Total', 'Turn_Total', 'Walking_Total']]


def plot_subject_velocity_data(subject, meta_data: Meta_Data):
    df_subject = pd.concat([
        meta_data.tdcsfog.loc[meta_data.tdcsfog.Subject == subject],
        meta_data.defog.loc[meta_data.defog.Subject == subject],
        meta_data.daily.loc[meta_data.daily.Subject == subject]
    ])

    print(f'subject {subject} has {len(df_subject)} entries')

    for index, row in df_subject.iterrows():
        if row['Source'] != 'daily':
            path = os.path.join(config['data_path'], 'train', row['Source'], f'{row["Id"]}.csv')
            if not os.path.exists(path):
                path = path = os.path.join(config['data_path'], 'train', 'notype', f'{row["Id"]}.csv')
            plot_velocity_data(path)


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
        

##################################### EXPERIMENTAL BELOW


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