import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

if __name__ == "__main__":
    """wthr = pd.read_csv("weather.csv")[
        ["Outdoor Drybulb Temperature [C]", "Relative Humidity [%]", "Diffuse Solar Radiation [W/m2]",
         "Direct Solar Radiation [W/m2]"]]
    w_data = pd.concat([wthr, pd.read_csv("Building_1.csv")[["Month", "Hour"]]], axis=1)



    df = pd.read_csv("weather.csv")
    df2 = pd.read_csv("Building_1.csv")
    dfb2 = pd.read_csv("Building_2.csv")
    dfb3 = pd.read_csv("Building_3.csv")
    dfb4 = pd.read_csv("Building_4.csv")
    dfb5 = pd.read_csv("Building_5.csv")
    #df6h = df[["6h Prediction Outdoor Drybulb Temperature [C]", "6h Prediction Relative Humidity [%]", "6h Prediction Diffuse Solar Radiation [W/m2]", "6h Prediction Direct Solar Radiation [W/m2]"]]
    #df6h = df6h.shift(periods=6)
    df_now = df[["Outdoor Drybulb Temperature [C]","Relative Humidity [%]","Diffuse Solar Radiation [W/m2]","Direct Solar Radiation [W/m2]"]]

    df2.rename(columns={"Solar Generation [W/kW]": "1st gen"}, inplace=True)
    dfb2.rename(columns = {"Solar Generation [W/kW]": "2nd gen"}, inplace=True)
    dfb3.rename(columns={"Solar Generation [W/kW]": "3rd gen"}, inplace=True)
    dfb4.rename(columns={"Solar Generation [W/kW]": "4th gen"}, inplace=True)
    dfb5.rename(columns={"Solar Generation [W/kW]": "5th gen"}, inplace=True)
    #data = pd.concat([df_now,df6h])
    #data["Outdoor Drybulb Temperature [C]"].map({"NaN": 0})

    indexes = [*range(1, 8761)]

    data2 = pd.concat([df2["1st gen"], dfb2["2nd gen"], dfb3["3rd gen"], dfb4["4th gen"], dfb5["5th gen"]], axis=1)


    #data2=(data2-data2.mean())/data2.std() #normalization
    #data2 = data2.replace({0:np.NaN}) #removing 0

    b = pd.concat([df2["1st gen"], dfb2["2nd gen"], dfb3["3rd gen"], dfb4["4th gen"], dfb5["5th gen"]])
    data2["avg"] = b.groupby(b.index).mean() #adding mean

    data2["Index"] = indexes
    #data2 = data2.dropna() #removing 0
    data2 = data2.truncate(before=0, after=24*2)  #show x days
    #print(data2.head(10))



    w_data["1st gen"] = df2["1st gen"]
    corr_matrix = w_data.corr()
    #corr_matrix = data2.drop(["Index"], axis=1).corr()
    sns.heatmap(corr_matrix)



    #data2.plot(x="Index", y=["1st gen", "2nd gen", "3rd gen", "4th gen", "5th gen", "avg"], kind="line")
    #data2.plot(kind='line', y='Solar Generation [W/kW]', x='Index', color='r')

    plt.show()"""


    def simple_policy():
        # Simple policy
        carbon_data = pd.read_csv(
            r"C:\Users\bjorn\OneDrive\Documents\TU Delft\EPOCH\CityLearn\citylearn-2022-starter-kit\data\citylearn_challenge_2022_phase_1\carbon_intensity.csv")
        price_data = pd.read_csv(r"C:\Users\bjorn\OneDrive\Documents\TU Delft\EPOCH\CityLearn\citylearn-2022-starter-kit\data\citylearn_challenge_2022_phase_1\pricing.csv")[
            "Electricity Pricing [$]"]

        c_mean = carbon_data.mean()
        c_std = carbon_data.std()
        p_mean = price_data.mean()
        p_std = price_data.std()

        carbon_data = (carbon_data - c_mean) / (c_std)
        price_data = (price_data - p_mean) / (p_std)

        together = pd.concat([carbon_data, price_data], axis=1)
        together["sum"] = together["kg_CO2/kWh"] + together["Electricity Pricing [$]"]

        return carbon_data, price_data, together

    a,b,c = simple_policy()
    print(c)






