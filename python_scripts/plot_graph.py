import pandas as pd
import plotly.express as px

df = pd.read_csv('training_log_100000_0.001.csv')

print(df.head())

fig = px.line(df, x = 'Epochs', y = 'Error', title='Epoch vs Error')
fig.show()