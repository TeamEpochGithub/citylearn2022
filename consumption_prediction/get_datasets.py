import pandas as pd
import numpy as np

data = pd.read_csv('solar_data.csv')

building_1_solar = data[:8758]
building_2_solar = data[8578:17516]
building_3_solar = data[17516:26274]
building_4_solar = data[26274:35032]
building_5_solar = data[35032:]


print(building_1_solar.columns)
# building_1_solar = building_1_solar.drop(columns=data.columns['^Unnamed'])
#
building_1_solar.to_csv('./building_specific_models/building1_solar.csv')
building_2_solar.to_csv('./building_specific_models/building2_solar.csv')
building_3_solar.to_csv('./building_specific_models/building3_solar.csv')
building_4_solar.to_csv('./building_specific_models/building4_solar.csv')
building_5_solar.to_csv('./building_specific_models/building5_solar.csv')
# print(building_1_solar.to_string())
# print(data['carbon_intensity'][:8757])