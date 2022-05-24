"""
Project: Covid Risk Analysis
Author: Ryder Davidson
Date: 11.01.21
"""

import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from sklearn.model_selection import train_test_split


class ANN(nn.Module):
    """
    A class used to create a simple feed-forward neural network
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=8, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def risk_level(case_metric):
    """
    Switch statement to grade covid risk-level given cases per 100,000

    Parameters
    ----------
    case_metric : float
        Number of covid cases per 100,000 people

    Return
    ------
    int
        Risk level (0-4 inclusive) of given input conditions
    """

    if case_metric < 1:
        return 0
    elif case_metric < 10:
        return 1
    elif case_metric < 25:
        return 2
    elif case_metric < 75:
        return 3
    else:
        return 4


def risk_analysis(query_state, query_day, training_epochs):
    pd.set_option('display.max_columns', None)

    # --- read in covid19 data and state population data --- #
    qstate = query_state
    qday = query_day
    epochs = training_epochs
    limit = '?$limit=50000'

    pd.options.mode.chained_assignment = None

    url_covid_stats = "https://data.cdc.gov/resource/9mfq-cb36.json" + limit
    url_pop_stats = "https://raw.githubusercontent.com/ryderdavidson/covid-risk-analysis/main/state_populations.json"

    df_covid = pd.read_json(url_covid_stats)
    df_pop = pd.read_json(url_pop_stats)

    # --- use only cols: date, state, new_case, new_death data
    # --- sort the resulting dataframe by date (since 01/22/20)

    df_prime = df_covid.filter(['submission_date', 'state', 'new_case', 'new_death'], axis=1)
    df_prime = df_prime.sort_values(by=['submission_date'])

    # --- create a day, month, year column for neural network

    date_format = "%Y-%m-%dT%H:%M:%S.%f"
    df_prime.insert(1, 'day', df_prime['submission_date'].apply(lambda x: datetime.strptime(x, date_format).day))
    df_prime.insert(2, 'month', df_prime['submission_date'].apply(lambda x: datetime.strptime(x, date_format).month))
    df_prime.insert(3, 'year', df_prime['submission_date'].apply(lambda x: datetime.strptime(x, date_format).year))

    # --- create population column by state/year

    index = 0
    for i in df_pop['state']:
        df_prime.loc[(df_prime['state'] == i) & (df_prime['year'] == 2020), 'population'] = df_pop.loc[
            index, 'pop_2020']
        df_prime.loc[(df_prime['state'] == i) & (df_prime['year'] == 2021), 'population'] = df_pop.loc[
            index, 'pop_2021']
        df_prime.loc[(df_prime['state'] == i) & (df_prime['year'] == 2022), 'population'] = df_pop.loc[
            index, 'pop_2022']
        index = index + 1

    df_prime['case_per_100K'] = df_prime['new_case'] / (df_prime['population'] / 100000)
    df_prime['risk_level'] = df_prime['case_per_100K'].apply(lambda x: risk_level(x))
    df_prime = df_prime.astype({'population': int})

    # --- query for state and date in future viz. [1, 7] days

    col_title = 'risk_in_' + str(qday) + '_day'

    qdf = df_prime.loc[df_prime['state'] == qstate]
    qdf.index = range(0, len(qdf))
    qdf[col_title] = qdf['risk_level'].shift(-qday)

    # qdf.plot(x='submission_date', y=['day', 'month', 'year', 'new_case', 'new_death', 'case_per_100K'], kind='line', figsize=(5, 5))
    #
    # mp.show()

    scaled_df = qdf.copy()

    scaled_df = scaled_df.drop(['submission_date', 'state'], axis=1)
    query_row = scaled_df.iloc[len(scaled_df) - 1, 0:].copy()
    query_row = query_row.fillna(0)
    scaled_df = scaled_df.dropna()
    scaled_df = scaled_df.append(query_row, ignore_index=True)

    print(scaled_df)

    for column in scaled_df.drop([col_title], axis=1).columns:
        scaled_df[column] = (
                (scaled_df[column] - scaled_df[column].min()) / (scaled_df[column].max() - scaled_df[column].min()))

    query_row = scaled_df.iloc[len(scaled_df) - 1, 0:(len(scaled_df.columns) - 1)].copy()
    scaled_df = scaled_df.iloc[0:(len(scaled_df) - 1), :].copy()

    X = scaled_df.drop([col_title], axis=1).values
    y = scaled_df[col_title].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    query_row = torch.FloatTensor(query_row)

    model = ANN()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    loss_arr = []
    percentage_complete = 10

    for i in range(epochs):
        y_hat = model.forward(X_train)
        loss = criterion(y_hat, y_train)
        loss_arr.append(loss)

        if i % (0.1 * epochs) == 0:
            print(f'{percentage_complete}% complete')
            percentage_complete = percentage_complete + 10

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # predictions = []
    #
    # with torch.no_grad():
    #     for i in X_test:
    #         y_hat = model.forward(i)
    #         predictions.append(y_hat.argmax().item())
    #
    # df_test = pd.DataFrame({'Y': y_test, 'Y_hat': predictions})
    # df_test['correct'] = [1 if corr == pred else 0 for corr, pred in zip(df_test['Y'], df_test['Y_hat'])]
    #
    # print(df_test['correct'].sum() / len(df_test))

    query_prediction = model.forward(query_row)
    return_arr = [query_prediction.argmax().item(), loss_arr[len(loss_arr) - 1].item()]

    return return_arr


x = risk_analysis('NYC', 2, 2000)
print(x)
