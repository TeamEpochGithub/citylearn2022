import pandas as pd

pricing = pd.read_csv("../pricing.csv")["Electricity Pricing [$]"]
months = pd.read_csv("../Building_1.csv")[["Month", "Hour"]]


concat = pd.concat([pricing, months], axis=1, ignore_index=True)

for i in range(1, 13):
    month = concat[concat[1] == i]
    print(f"Month {i}\n{month[0].value_counts()} \n\n")
    previous_hour = 0

    for h in range(1, 25):
        hour = month[month[2]==h]
        values = dict(hour[0].value_counts())
        if values != previous_hour:
            print(f"Hour {h}: \n{hour[0].value_counts()} \n\n")
        previous_hour = values

        if len(values) > 1:
            print(f"Hour {h}: \n{hour.head(35)}")


#January (31 days): Price = 0.21 for hour 1-15, 0.5 for hour 16-20, 0.21 for hour 21-24
#February (28 days): Price = 0.21 for hour 1-15, 0.5 for hour 16-20, 0.21 for hour 21-24
#March (31 days): Price = 0.21 for hour 1-15, 0.5 for hour 16-20, 0.21 for hour 21-24
#April (30 days): Price = 0.21 for hour 1-15, 0.5 for hour 16-20, 0.21 for hour 21-24
#May (31 days): Price = 0.21 for hour 1-15, 0.5 for hour 16-20, 0.21 for hour 21-24 except last day: 0.22 for hour 24

#June (30 days): Price = 0.22 for hour 1-15, alternating 0.54 and 0.40 as (2, 2, 5, 2, 5, 2, 5, 2, 5)
# aka 0.54 on weekdays and 0.40 on weekends for hour 16-20, 0.22 for hour 21-24
#July (31 days): Price = 0.22 for hour 1-15, 0.54 if weekday or 0.40 if weekend for hour 16-20, 0.22 for hour 21-24
#August (31 days): Price = 0.22 for hour 1-15, 0.54 if weekday or 0.40 if weekend for hour 16-20, 0.22 for hour 21-24
#September (30 days): Price = 0.22 for hour 1-15, 0.54 if weekday or 0.40 if weekend for hour 16-20, 0.22 for hour 21-24
# except last day: 0.21 for hour 24

#October (31 days): Price = 0.21 for hour 1-15, 0.5 for hour 16-20, 0.21 for hour 21-24
#November (30 days): Price = 0.21 for hour 1-15, 0.5 for hour 16-20, 0.21 for hour 21-24
#December (31 days): Price = 0.21 for hour 1-15, 0.5 for hour 16-20, 0.21 for hour 21-24

def pricing(month, hour, day):
    if 1 <= month <= 5 or 10 <= month <= 12:
        if 1 <= hour <= 15 or 21 <= hour <= 24:
            price = 0.21 #Except for 31/05 at 24: 0.22
        else:
            price = 0.5
    else:
        if 1 <= hour <= 15 or 21 <= hour <= 24:
            price = 0.22 #Except for 30/09 at 24: 0.21
        else:
            if day >= 6:
                price = 0.40
            else:
                price = 0.54

    return price


