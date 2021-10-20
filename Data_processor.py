from os import sep, truncate
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import sklearn
from sklearn.model_selection import train_test_split
import seaborn as sns

def test_train_split():
    ''' 
    Take a sample of 4000 from the total 20000 of
    hatexplain data set and split it into test and
    train dataset
    '''
    df = pd.read_csv('HateExplain\hateexplain_2_VUA.csv', sep='\t')


    df = df.sample(frac=1)

    test = df.head(n=400)

    train = df.tail(n=3600)

    train.to_csv('trainData.csv', sep='\t')
    
    test.to_csv('testData.csv', sep='\t')


def make_scatter_plot(model, data):
    '''
    Process data and make a scatter plot
    '''

    df1 = pd.read_csv(f'Values/{data}/Values_{data}_{model}.csv')
    df2 = pd.read_csv(f'Values/{data}/Values_{data}_Bert.csv')
    values_1 = ['1']*len(df1['Word'])
    values_2 = ['2']*len(df2['Word'])

    df1['Model'] = values_1
    df2['Model'] = values_2
    l = [df1, df2]
    
    new = pd.concat(l)

    


    # print(df.describe())
    new_df1 = df1.groupby('Word').mean()
    new_df1['count'] = df1.groupby('Word').count()['Value']

    # print(new_df)
    df1 = new_df1.loc[(new_df1['Value'] > 0.10)]

    new_df2 = DataFrame()
    # print(df.describe())
    new_df2 = df2.groupby('Word').mean()
    new_df2['count'] = df2.groupby('Word').count()['Value']

    # print(new_df)
    df2 = new_df2.loc[(new_df2['Value'] > 0.10)]


    
    print(df1)

    df1['label'] = ['1' for i in range(len(df1['Value']))]
    df2['label'] = ['2' for i in range(len(df2['Value']))]

    df = pd.concat([df1, df2])
    
    print(df)
    # new_df = DataFrame()
    # # print(df.describe())
    # new_df = df.groupby('Word').mean()
    # new_df['count'] = df.groupby('Word').count()['Value']

    # # print(new_df)
    # df = new_df.loc[(new_df['Value'] > 0.10)]

    
    
    df = df.sort_values(['count','Value'], ascending=[True, False])

    print(df)
    sns.scatterplot(data=df, x="Value", y="count", hue='label', alpha = 0.5)
    # plt.yscale('log')
    plt.show()

def confusion_matrix(data, model):
    '''
    Make a confusion matrix from prediction data
    '''

    df = pd.read_csv(f'predition_{data}_Bert.csv', sep='\t')

    
    # df = df.replace(1,0)
    # df = df.replace(0,1)
    
    df = df.replace(1,'hate')
    df = df.replace(0,'noHate')
    
    contingency_matrix = pd.crosstab(df['predictions'], df['True'], normalize= 'columns')
    print(contingency_matrix)
    import matplotlib.pyplot as plt
    import seaborn as sn
    import matplotlib

    matplotlib.rcParams.update({'font.size': 16})
    plt.clf()



    res = sn.heatmap(contingency_matrix.T, annot=True, fmt='.2f', cmap="Blues", cbar=True)
    # plt.show()

    plt.savefig(f"error_matrices/error_matrx_{data}_{model}.pdf", bbox_inches='tight', dpi=100)




confusion_matrix('HateExplain', 'Bert')

# make_scatter_plot('HateBert', 'Gibert')

# test_train_split()
