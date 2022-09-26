def hour_to_time(hour):
    times = [0, 0, 0, 0]  # morning, afternoon, evening, night
    if 7 <= hour < 11:
        times[0] = 1
    elif 11 <= hour < 17:
        times[1] = 1
    elif 17 <= hour < 23:
        times[2] = 1
    else:
        times[3] = 1
    return times
