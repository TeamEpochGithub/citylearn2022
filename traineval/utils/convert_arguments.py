import numpy as np


# TODO: finish this method, and methods that use it, to get scaled values
def environment_convert_scalars(argument_list):
    max_scalars = [12, 7, 24, 32.2, 32.2, 32.2, 32.2, 100, 100, 100, 100, 1017, 1017, 1017, 1017, 953, 953, 953, 953,
                   0.2818, 8, 4, 1, 12, 0.54, 0.54, 0.54, 0.54]
    min_scalars = [1, 1, 1, 5.6, 5.6, 5.6, 5.6, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0.0703, 0, 0, 0, -8.2, 0.27,
                   0.27, 0.27, 0.27]
    if len(argument_list) == 0:
        return []
    if isinstance(argument_list[0], str):
        argument_list = environment_convert_argument(argument_list)
    return np.array(max_scalars)[argument_list]


def get_max_scalars():
    return [12, 7, 24, 32.2, 32.2, 32.2, 32.2, 100, 100, 100, 100, 1017, 1017, 1017, 1017, 953, 953, 953, 953, 0.2818,
            8, 4, 1, 12, 0.54, 0.54, 0.54, 0.54]


def get_min_scalars():
    return [1, 1, 1, 5.6, 5.6, 5.6, 5.6, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0.0703, 0, 0, 0, -8.2, 0.27,
            0.27, 0.27, 0.27]


def environment_convert_argument(argument_list):
    argument_string_list = ["month",  # 0
                            "day_type",  # 1
                            "hour",  # 2
                            "outdoor_dry_bulb",  # 3
                            "outdoor_dry_bulb_temperature_predicted_6h",  # 4
                            "outdoor_dry_bulb_temperature_predicted_12h",  # 5
                            "outdoor_dry_bulb_temperature_predicted_24h",  # 6
                            "outdoor_relative_humidity",  # 7
                            "outdoor_relative_humidity_predicted_6h",  # 8
                            "outdoor_relative_humidity_predicted_12h",  # 9
                            "outdoor_relative_humidity_predicted_24h",  # 10
                            "diffuse_solar_irradiance",  # 11
                            "diffuse_solar_irradiance_predicted_6h",  # 12
                            "diffuse_solar_irradiance_predicted_12h",  # 13
                            "diffuse_solar_irradiance_predicted_24h",  # 14
                            "direct_solar_irradiance",  # 15
                            "direct_solar_irradiance_predicted_6h",  # 16
                            "direct_solar_irradiance_predicted_12h",  # 17
                            "direct_solar_irradiance_predicted_24h",  # 18
                            "carbon_intensity",  # 19
                            "non_shiftable_load",  # 20
                            "solar_generation",  # 21
                            "electrical_storage_soc",  # 22
                            "net_electricity_consumption",  # 23
                            "electricity_pricing",  # 24
                            "electricity_pricing_predicted_6h",  # 25
                            "electricity_pricing_predicted_12h",  # 26
                            "electricity_pricing_predicted_24h"]  # 27

    toggle_int_to_string = False
    if len(argument_list) == 0:
        return []
    if isinstance(argument_list[0], int):
        toggle_int_to_string = True

    if toggle_int_to_string:
        return np.array(argument_string_list)[argument_list]
    else:
        indices = []
        for x in argument_list:
            indices.append(argument_string_list.index(x))
        return indices


def get_environment_arguments(district_args, building_args):
    district_data_indices = environment_convert_argument(district_args)
    district_scalar_indices = environment_convert_scalars(district_args)

    building_data_indices = environment_convert_argument(building_args)
    building_scalar_indices = environment_convert_scalars(building_args)

    environment_arguments = {
        "district_indexes": district_data_indices,
        "district_scalars": district_scalar_indices,
        "building_indexes": building_data_indices,
        "building_scalars": building_scalar_indices}

    return environment_arguments
