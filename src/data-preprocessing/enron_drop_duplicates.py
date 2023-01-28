from distutils.command.config import dump_file
import pandas as pd

df = pd.read_pickle('/p/fm/MundraNikita/Project/nikita_master_thesis/masterthesis-nikita/Code/src/dataset/Enron/thesis_full_enron')

pd.set_option('display.max_columns', None)

print("Sender",type(df['Sender'][0]))
print("Recipient", type(df['Recipient'][0]))
print("CC", type(df['CC'][2]))
print("BCC", type(df['BCC'][0]))
print("date", type(df['date'][0]))
print("subject", type(df['subject'][0]))
print("Text", type(df['Text'][0]))
print("Mime", type(df['Mime'][0]))
print("OriginFolder", type(df['OriginFolder'][0]))
print("ContentType", type(df['ContentType'][0]))
print("ContentEncoding", type(df['ContentEncoding'][0]))
print(df.iloc[0])
'''df.drop_duplicates(subset = ['Sender', 'date', 'subject', 'Text'], inplace = True)
df1 = df.groupby(['OriginFolder'])['OriginFolder'].count().sort_values()
df1.to_csv('foldercount.csv')'''
