from traineval.evaluation.tune_evaluation import evaluate
from multiprocessing import Pool, cpu_count


hours = range(1, 25)
actions = [x / 100.0 for x in range(-15, 15, 3)]

print(str(hours))
print(str(actions))

def get_all_pairs():
    pairs = []
    for i in range(len(hours)):
        for j in range(len(actions)):
            pairs.append((i, j))
    return pairs


def get_params(all_pairs):
    all_params = []
    for p in all_pairs:
        all_params.append({"hour": hours[p[0]], "action": actions[p[1]]})
    return all_params


if __name__ == '__main__':
    all_scores = {}
    all_pairs = get_all_pairs()
    all_params = get_params(all_pairs)
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
