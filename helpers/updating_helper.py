import pandas as pd
import numpy as np


readHelper = pd.read_csv('helpers/output.csv')

dictOfHelper = readHelper.to_dict('list')


for key, value in dictOfHelper.items():
    print('key:', key, 'len of value:', len(value))


metadata = {
    'Column_Name': [],
    'Description': [],
    'Measurement': [],
    'customDescription': [],
    'Measurement_2': []
}


metadata['Column_Name'] = dictOfHelper['Column_Name']
metadata['Description'] = dictOfHelper['Description']
metadata['Measurement'] = dictOfHelper['Measurement']
metadata['customDescription'] = dictOfHelper['customDescription']
metadata['Measurement_2'] = dictOfHelper['Measurement_2']


'''
new_metadata = {
    'Column_Name': ['Population','GDP_Per_Capita'],
    'Description': ['Self descriptive','Self descriptive'],
    'Measurement': ['Numeric','Numeric'],
    'Coding (if given)': ['-','-'],
    'customDescription': ['Population from OWID','GDP per capita from OWID'],
    'Measurement_2': ['Numeric','Numeric']
}

for key in metadata:
    metadata[key].extend(new_metadata[key])

for key, value in metadata.items():
    print('key:', key, 'len of value:', len(value))

df = pd.DataFrame(metadata)
df.to_csv('helpers/output.csv', index=False)
'''
# TODO: Update data_description.csv in the end