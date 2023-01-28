import pandas as pd
from glob import glob
import re
import os
pd.set_option('display.max_columns', None)
count = 0
list_dir = []
folder_dict = {}
file_dict = {}
dir = r'/p/fm/MundraNikita/Project/maildir'
for dir1 in os.listdir(dir):

    list_dir.append(dir1)
print(list_dir)

# code to count number of folders for each user
for i in list_dir:
    count = 0
    file_count = 0
    path =  os.path.join(dir, i)
    for sub_dir in os.listdir(path):
        count = count+1
        file_path = os.path.join(path, sub_dir)
        for files in glob(f'{file_path}/*',recursive=True):
            print(files)
            file_count = file_count+1
    folder_dict[i] = count 
    file_dict[i] = file_count
df_folder_count = pd.DataFrame(folder_dict.items(), columns = ['User', 'FolderCount'])
df_folder_count.sort_values(by = ['FolderCount'], inplace=True)
df_file_count = pd.DataFrame(file_dict.items(), columns = ['User', 'FileCount'])
df_file_count.sort_values(by = ['FileCount'], inplace=True)
df_folder_count.to_csv('folder_count.csv')
df_file_count.to_csv('file_count.csv')



'''for i in list_users:
    for dir1 in dir+'/'+i:
        print(dir)'''
    
    
'''string = "/p/fm/MundraNikita/Project/maildir/blair-l/personnel___promotions/1_.txt"
pattern = re.search('(?<=maildir\/)(\w+)', string)
print(pattern.group(0))
pattern1 = re.search('maildir(?<=\w+)(\w+)', string)
print(pattern1)'''