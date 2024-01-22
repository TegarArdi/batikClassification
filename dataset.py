import pandas as pd

def glcm_avg():
    csv_save = 'csv/ekstraksi_fitur.csv'
    properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy', 'label']
    angles = ['0', '45', '90', '135']

    df = pd.read_csv(csv_save)
    df_avg = pd.DataFrame()

    df_avg['dissiimilarity'] = df.iloc[:, 0:4].mean(axis=1)
    df_avg['correlation'] = df.iloc[:, 4:8].mean(axis=1)
    df_avg['homogeneity'] = df.iloc[:, 8:12].mean(axis=1)
    df_avg['contrast'] = df.iloc[:, 12:16].mean(axis=1)
    df_avg['ASM'] = df.iloc[:, 16:20].mean(axis=1)
    df_avg['energy'] = df.iloc[:, 20:24].mean(axis=1)
    df_avg['label']  = df.iloc[:, 24]

    print(df_avg)

glcm_avg()

