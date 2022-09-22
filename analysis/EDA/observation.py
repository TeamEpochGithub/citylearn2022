class Observation:
    def __init__(self, observation: list):
        """
        Class to get information from plain observation list with float values from the environment
        :param observation: Observation from env.step() function in CityLearn environment
        """
        self.month = observation[0]
        self.day_type = observation[1]
        self.hour = observation[2]
        self.outdoor_dry_bulb_temperature = observation[3]
        self.outdoor_dry_bulb_temperature_predicted_6h = observation[4]
        self.outdoor_dry_bulb_temperature_predicted_12h = observation[5]
        self.outdoor_dry_bulb_temperature_predicted_24h = observation[6]
        self.outdoor_relative_humidity = observation[7]
        self.outdoor_relative_humidity_predicted_6h = observation[8]
        self.outdoor_relative_humidity_predicted_12h = observation[9]
        self.outdoor_relative_humidity_predicted_24h = observation[10]
        self.diffuse_solar_irradiance = observation[11]
        self.diffuse_solar_irradiance_predicted_6h = observation[12]
        self.diffuse_solar_irradiance_predicted_12h = observation[13]
        self.diffuse_solar_irradiance_predicted_24h = observation[14]
        self.direct_solar_irradiance = observation[15]
        self.direct_solar_irradiance_predicted_6h = observation[16]
        self.direct_solar_irradiance_predicted_12h = observation[17]
        self.direct_solar_irradiance_predicted_24h = observation[18]
        self.carbon_intensity = observation[19]
        self.non_shiftable_load = observation[20]
        self.solar_generation = observation[21]
        self.electrical_storage_soc = observation[22]
        self.net_electricity_consumption = observation[23]
        self.electricity_pricing = observation[24]
        self.electricity_pricing_predicted_6h = observation[25]
        self.electricity_pricing_predicted_12h = observation[26]
        self.electricity_pricing_predicted_24h = observation[27]

    def get_info(self, variable_index_or_name):
        if isinstance(variable_index_or_name, str):
            variable_index_or_name = list(self.__dict__.keys()).index(variable_index_or_name)
        return [
            "Month of year ranging from 1 (January) through 12 (December).",
            "Day of week ranging from 1 (Monday) through 7 (Sunday).",
            "Hour of day ranging from 1 to 24.",
            "Outdoor dry bulb temperature.",
            "Outdoor dry bulb temperature predicted 6 hours ahead.",
            "Outdoor dry bulb temperature predicted 12 hours ahead.",
            "Outdoor dry bulb temperature predicted 24 hours ahead",
            "Outdoor relative humidity.",
            "Outdoor relative humidity predicted 6 hours ahead.",
            "Outdoor relative humidity predicted 12 hours ahead.",
            "Outdoor relative humidity predicted 24 hours ahead.",
            "Diffuse solar irradiance.",
            "Diffuse solar irradiance predicted 6 hours ahead.",
            "Diffuse solar irradiance predicted 12 hours ahead.",
            "Diffuse solar irradiance predicted 24 hours ahead.",
            "Direct solar irradiance.",
            "Direct solar irradiance predicted 6 hours ahead.",
            "Direct solar irradiance predicted 12 hours ahead.",
            "Direct solar irradiance predicted 24 hours ahead.",
            "Grid carbon emission rate.",
            "Total building non-shiftable plug and equipment loads.",
            "PV electricity generation.",
            "State of the charge (SOC) of the electrical_storage from 0 (no energy stored) to 1 (at full capacity).",
            "Total building electricity consumption.",
            "Electricity rate.",
            "Electricity rate predicted 6 hours ahead.",
            "Electricity rate predicted 12 hours ahead.",
            "Electricity rate predicted 24 hours ahead."
        ][variable_index_or_name]

    def __str__(self):
        variables = list(self.__dict__.keys())
        values = list(self.__dict__.values())
        return '\n'.join([f"Name: {variables[i]}    ---- Value: {values[i]}    ---- Desc: {self.get_info(variables[i])}"
                          for i in range(len(variables))])
