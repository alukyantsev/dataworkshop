import pandas as pd
import numpy as np

# сводная информация по пропускам
def gap_info(df):

    mis_val = df.isnull().sum()
    mis_val_percent = 100 * mis_val / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table. \
        rename( columns = {0 : 'Missing Values', 1 : '% of Total Values'} )
    mis_val_table_ren_columns = mis_val_table_ren_columns[ mis_val_table_ren_columns.iloc[:,1] != 0 ]. \
        sort_values('% of Total Values', ascending=False).round(1)
    print('Your selected dataframe has ' + str(df.shape[1]) + ' columns.\n'      
          'There are ' + str(mis_val_table_ren_columns.shape[0]) + ' columns that have missing values.')
    return mis_val_table_ren_columns

# возвращает перечень колонок со значением < 0
def gap_negative(df):

    list_negative = []
    for c in df.select_dtypes(include=['number']).columns:
        count = df[ df[c]<0 ][c].count()
        print('%d\t- %s' % (count, c))
        if count > 0:
            list_negative.append(c)
    return list_negative
