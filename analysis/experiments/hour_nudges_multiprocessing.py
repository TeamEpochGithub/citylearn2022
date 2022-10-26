search_space = {
    "hour_0": range(1, 25),
    # "hour_1": range(1, 25),
    # "hour_2": range(1, 25),
    # "hour_3": range(1, 25),
    "action_0": [x / 50.0 for x in range(-20, 20, 1)[15:25]],
    # "action_1": [x / 50.0 for x in range(-20, 20, 1)[15:25]],
    # "action_2": [x / 50.0 for x in range(-20, 20, 1)[15:25]],
    # "action_3": [x / 50.0 for x in range(-20, 20, 1)[15:25]],
}
print(search_space)

from traineval.evaluation.tune_evaluation import evaluate

hours = range(1, 25)
actions = [x / 50.0 for x in range(-20, 20, 1)[15:25]]

from multiprocessing import Pool, cpu_count

if __name__ == '__main__':
    #
    # all_possibilities = []
    # for i in range(len(hours)):
    #     for j in range(len(actions)):
    #         params = {"hour_0": hours[i],
    #                   "action_0": actions[j]}
    #         all_possibilities.append(params)

    all_scores = {}

    with Pool(processes=(cpu_count() - 6)) as pool:
        for i in range(len(hours)):
            for j in range(len(actions)):
                res = pool.apply_async(evaluate, args=({"hour_0": hours[i], "action_0": actions[j]},))
                print(res.get())
                loss, params = res.get()
                all_scores[str(params)] = loss

    pool.close()
    pool.join()
    print(all_scores)
