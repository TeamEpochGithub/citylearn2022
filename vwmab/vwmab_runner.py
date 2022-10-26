import pandas as pd
import numpy as np
import math
import vowpalwabbit
import json

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def get_cost(true_action, predicted_action):
    return abs(true_action - predicted_action)


def to_vw_example_format(context, cats_label=None):
    example_dict = {}
    if cats_label is not None:
        chosen_temp, cost, pdf_value = cats_label
        example_dict["_label_ca"] = {
            "action": chosen_temp,
            "cost": cost,
            "pdf_value": pdf_value,
        }
    example_dict["c"] = {
        "month={}".format(context["month"]): 1,
        "day_type={}".format(context["day_type"]): 1,
        "outdoor_dry_bulb={}".format(context["outdoor_dry_bulb"]): 1,
        "outdoor_dry_bulb_temperature_predicted_6h={}".format(context["outdoor_dry_bulb_temperature_predicted_6h"]): 1,
        "outdoor_dry_bulb_temperature_predicted_12h={}".format(
            context["outdoor_dry_bulb_temperature_predicted_12h"]): 1,
        "outdoor_dry_bulb_temperature_predicted_24h={}".format(
            context["outdoor_dry_bulb_temperature_predicted_24h"]): 1,
        "outdoor_relative_humidity={}".format(context["outdoor_relative_humidity"]): 1,
        "outdoor_relative_humidity_predicted_6h={}".format(context["outdoor_relative_humidity_predicted_6h"]): 1,
        "outdoor_relative_humidity_predicted_12h={}".format(context["outdoor_relative_humidity_predicted_12h"]): 1,
        "outdoor_relative_humidity_predicted_24h={}".format(context["outdoor_relative_humidity_predicted_24h"]): 1,
        "diffuse_solar_irradiance={}".format(context["diffuse_solar_irradiance"]): 1,
        "diffuse_solar_irradiance_predicted_6h={}".format(context["diffuse_solar_irradiance_predicted_6h"]): 1,
        "diffuse_solar_irradiance_predicted_12h={}".format(context["diffuse_solar_irradiance_predicted_12h"]): 1,
        "diffuse_solar_irradiance_predicted_24h={}".format(context["diffuse_solar_irradiance_predicted_24h"]): 1,
        "direct_solar_irradiance={}".format(context["direct_solar_irradiance"]): 1,
        "direct_solar_irradiance_predicted_6h={}".format(context["direct_solar_irradiance_predicted_6h"]): 1,
        "direct_solar_irradiance_predicted_12h={}".format(context["direct_solar_irradiance_predicted_12h"]): 1,
        "direct_solar_irradiance_predicted_24h={}".format(context["direct_solar_irradiance_predicted_24h"]): 1,
        "carbon_intensity={}".format(context["carbon_intensity"]): 1,
        "non_shiftable_load={}".format(context["non_shiftable_load"]): 1,
        "solar_generation={}".format(context["solar_generation"]): 1,
        "electrical_storage_soc={}".format(context["electrical_storage_soc"]): 1,
        "net_electricity_consumption={}".format(context["net_electricity_consumption"]): 1,
        "electricity_pricing={}".format(context["electricity_pricing"]): 1,
        "electricity_pricing_predicted_6h={}".format(context["electricity_pricing_predicted_6h"]): 1,
        "electricity_pricing_predicted_12h={}".format(context["electricity_pricing_predicted_12h"]): 1,
        "electricity_pricing_predicted_24h={}".format(context["electricity_pricing_predicted_24h"]): 1,
    }
    return json.dumps(example_dict)


def predict_action(vw, context):
    vw_text_example = to_vw_example_format(context)
    return vw.predict(vw_text_example)


def train():
    context = pd.read_csv(r"C:\Users\Lars\Documents\Epoch\CityLearn\citylearn-2022-starter-kit\vwmab\context.csv")

    context_target = context["action"]
    context_data = context.drop(["action"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(context_data, context_target, test_size=0.2, random_state=42)

    X_train = X_train.to_dict(orient="records")
    y_train = np.asarray(y_train)

    X_test = X_test.to_dict(orient="records")
    y_test = np.asarray(y_test)

    reward_rate = []
    hits = 0
    cost_sum = 0.0

    iterations = len(X_train)

    num_actions = 40
    bandwidth = 0.05

    # Instantiate VW learner
    vw = vowpalwabbit.Workspace(
        "--cats "
        + str(num_actions)
        + "  --bandwidth "
        + str(bandwidth)
        + " --min_value 0 --max_value 32 --json --chain_hash --coin --epsilon 0.2 -q :: --quiet"
    )

    for index, current_context in enumerate(X_train):

        if (index + 1) % 100 == 0:
            print(index + 1)

        current_action = y_train[index]

        predicted_action, pdf_value = predict_action(vw, current_context)

        cost = get_cost(current_action, predicted_action)

        if cost <= -0.05:
            hits += 1

        cost_sum += cost

        txt_ex = to_vw_example_format(context, cats_label=(predicted_action, cost, pdf_value))

        vw_format = vw.parse(txt_ex, vowpalwabbit.LabelType.CONTINUOUS)

        vw.learn(vw_format)

        vw.finish_example(vw_format)

        reward_rate.append(cost_sum / index + 1)

    vw.finish()


    plt.plot(range(1, iterations + 1), reward_rate)
    plt.xlabel("num_iterations", fontsize=14)
    plt.ylabel("reward rate", fontsize=14)
    plt.title("Plot")
    plt.ylim([0, 1])

    plt.show()

    print(cost_sum / iterations)


if __name__ == '__main__':
    train()
