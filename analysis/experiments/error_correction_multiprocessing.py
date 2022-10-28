from multiprocessing import Pool, cpu_count
from traineval.evaluation.tune_evaluation import evaluate

search_space = {
    "division_factor": range(1, 10),
    "threshold": [x / 10.0 for x in range(0, 20, 1)],
}
# print(search_space)

division_factors = search_space["division_factor"]
thresholds = search_space["threshold"]
error_hours = [[1], [24], [23, 24, 25], [1, 24], [1, 23, 24, 25], [1, 2, 23, 24, 25]]


def get_all_pairs():
    pairs = []
    for i in range(len(division_factors)):
        for j in range(len(thresholds)):
            pairs.append((i, j))
    return pairs


def get_params(all_pairs):
    all_params = []
    for p in all_pairs:
        all_params.append({"division_factor": division_factors[p[0]], "threshold": thresholds[p[1]]})
    return all_params


def get_hours():
    all_params = []
    for hours in error_hours:
        all_params.append({"error_hours": hours})
    return all_params


if __name__ == '__main__':
    all_scores = {}
    # all_pairs = get_all_pairs()
    # all_params = get_params(all_pairs)
    all_params = get_hours()
    with Pool(processes=(cpu_count() - 4)) as pool:
        res = pool.map_async(evaluate, all_params)
        print(res.get())
        loss, params = res.get()
        all_scores[str(params)] = loss

    pool.close()
    pool.join()
    print(all_scores)
    all_scores = {k: v for k, v in sorted(all_scores.items(), key=lambda item: item[1])}
    print(all_scores)
