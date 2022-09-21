import numpy as np

# TODO: finish this method, and methods that use it, to get scaled values
def environment_convert_scalars(argument_list):

    max_scalars = [12, 7, 24, 32.2, 32.2, 32.2, 32.2, 100, 100, 100, 100, 1017, 1017, 1017, 1017, 953, 953, 953, 953, 0.2818, 8, 4, 1, 12, 0.54, 0.54, 0.54, 0.54]
    min_scalars = [1, 1, 1, 5.6, 5.6, 5.6, 5.6, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0.0703, 0, 0, 0, -8.2, 0.27, 0.27, 0.27, 0.27]

    if isinstance(argument_list[0], str):
        argument_list = environment_convert_argument(argument_list)
    return np.array(max_scalars)[argument_list]


def environment_convert_argument(argument_list):

    argument_string_list = ["month",
                            "day_type",
                            "hour",
                            "outdoor_dry_bulb",
                            "outdoor_dry_bulb_temperature_predicted_6h",
                            "outdoor_dry_bulb_temperature_predicted_12h",
                            "outdoor_dry_bulb_temperature_predicted_24h",
                            "outdoor_relative_humidity",
                            "outdoor_relative_humidity_predicted_6h",
                            "outdoor_relative_humidity_predicted_12h",
                            "outdoor_relative_humidity_predicted_24h",
                            "diffuse_solar_irradiance",
                            "diffuse_solar_irradiance_predicted_6h",
                            "diffuse_solar_irradiance_predicted_12h",
                            "diffuse_solar_irradiance_predicted_24h",
                            "direct_solar_irradiance",
                            "direct_solar_irradiance_predicted_6h",
                            "direct_solar_irradiance_predicted_12h",
                            "direct_solar_irradiance_predicted_24h",
                            "carbon_intensity",
                            "non_shiftable_load",
                            "solar_generation",
                            "electrical_storage_soc",
                            "net_electricity_consumption",
                            "electricity_pricing",
                            "electricity_pricing_predicted_6h",
                            "electricity_pricing_predicted_12h",
                            "electricity_pricing_predicted_24h"]

    toggle_int_to_string = False
    if isinstance(argument_list[0], int):
        toggle_int_to_string = True

    if toggle_int_to_string:
        return np.array(argument_string_list)[argument_list]
    else:
        indices = []
        for x in argument_list:
            indices.append(argument_string_list.index(x))
        return indices