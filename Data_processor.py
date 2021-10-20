from os import sep, truncate
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import sklearn
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib
from sklearn.metrics import classification_report

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


def make_scatter_plot(data, model):
    '''
    Process data and make a scatter plot
    '''

    df = pd.read_csv(f'Values/{data}/Values_{data}_{model}.csv')
    
    new_df = df.groupby('Word').mean()

    new_df['count'] = df.groupby('Word').count()['Value']

    df = new_df.loc[(new_df['Value'] > 0.10)]
    
    
    df = df.sort_values(['count','Value'], ascending=[True, False])
    sns.scatterplot(data=df, x="Value", y="count", hue='label', alpha = 0.5)

    plt.show()

def confusion_matrix(data, model, plot=False):
    '''
    Make a confusion matrix from prediction data
    '''

    df = pd.read_csv(f'Predictions/prediction_{data}_{model}.csv', sep='\t')
    
    df = df.replace(1,'hate')
    df = df.replace(0,'noHate')
    
    contingency_matrix = pd.crosstab(df['predictions'], df['True'], normalize= 'columns')
    print(contingency_matrix)

    matplotlib.rcParams.update({'font.size': 16})
    plt.clf()



    res = sns.heatmap(contingency_matrix.T, annot=True, fmt='.2f', cmap="Blues", cbar=True)
    plt.savefig(f"error_matrices/error_matrx_{data}_{model}.pdf", bbox_inches='tight', dpi=100)
    if plot == True:
        plt.show()

def get_metrics(data, model):
    df = pd.read_csv(f'Predictions/prediction_{data}_{model}.csv', sep='\t')
    print(classification_report(df['True'], df['predictions']))

def describe(data):
    test = pd.read_csv(f'{data}_data/testData.csv')
    train = pd.read_csv(f'{data}_data/trainData.csv')

    df = pd.concat([test,train])

    print(df.describe())
    print(df['Labels'].value_counts())


if __name__ == "__main__":

    model = 'HateBert'
    data = 'HateExplain'

    # test_train_split()

    confusion_matrix(data, model)

    get_metrics(data, model)

    # make_scatter_plot(data, model)


    