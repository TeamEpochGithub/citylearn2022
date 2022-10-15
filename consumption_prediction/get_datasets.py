import pandas as pd
import numpy as np

data = pd.read_csv(r'C:\Users\kuipe\OneDrive\Bureaublad\Epoch\citylearn-2022-starter-kit\data\citylearn_challenge_2022_phase_1\load_data.csv')

print(data.describe())

# building_1_load = data[:8758]
# building_2_load = data[8578:17516]
# building_3_load = data[17516:26274]
# building_4_load = data[26274:35032]
# building_5_load = data[35032:]
#
#
# print(building_1_load.columns)
# # building_1_load = building_1_load.drop(columns=data.columns['^Unnamed'])
# #
# building_1_load.to_csv('./building_specific_models/building1_load.csv', index=False)
# building_2_load.to_csv('./building_specific_models/building2_load.csv', index=False)
# building_3_load.to_csv('./building_specific_models/building3_load.csv', index=False)
# building_4_load.to_csv('./building_specific_models/building4_load.csv', index=False)
# building_5_load.to_csv('./building_specific_models/building5_load.csv', index=False)
# print(building_1_load.to_string())
# print(data['carbon_intensity'][:8757])