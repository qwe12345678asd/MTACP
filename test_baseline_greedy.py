from utils.distance_tool import Euclidean_fun
import json, math, time, itertools, copy, random
from random import choice
from itertools import combinations, product
import numpy as np

def satisfy_fun(A, B):
    w_loc=A['Lw']
    t_loc=B['Lt']
    trave_dis = math.sqrt(sum([(a - b)**2 for (a, b) in zip(w_loc, t_loc)]))
    trave_time = math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(w_loc, t_loc)]))/1
    trave_cost = math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(w_loc, t_loc)]))* v
    if trave_dis>R or A['arrived_time']+trave_time>A['deadline'] or B['arrived_time']+trave_time>B['deadline'] \
            or B['budget']-trave_cost<=0:
        return False
    else:
        return True

def worker_score(worker_list):
    total = 0
    for w in worker_list:
        total += batch_worker_dict[w]['score']

    return total / len(worker_list)

def price_score(t, v, worker_list):
    """
    price_score = task_budget - all_task_worker_pair_distance * v
    """
    B=batch_task_dict[t]['Lt']
    if len(worker_list) == 0:
        return 0
    else:
        dis = 0
        for i in worker_list:
            A = batch_worker_dict[i]['Lw']
            dis += math.sqrt(sum([(a - b)**2 for (a, b) in zip(A, B)]))
            batch_worker_dict[i]['Lw']=task_dict[t]['Lt']
            batch_worker_dict[i]['arrived_time']=batch_worker_dict[i]['arrived_time']+math.sqrt(sum([(a - b)**2 for (a, b) in zip(A, B)]))/1
        return task_dict[t]['budget'] - dis * v


def check(task, worker_list):
    skill_workers = []
    for i in worker_list:
        skill_workers.append(worker_dict[i]['Kw'][0])

    if len(task_dict[task]['Kt']) == len(skill_workers):
        flag = True
    else:
        flag = False

    return flag

def conflict_check(best_assign):

    flag = True
    for i in best_assign.keys():
        for j in best_assign.keys():
            if i == j:
                continue
            else:
                if len(set(best_assign[i]).intersection(set(best_assign[j]))) != 0:
                    flag = False
                    break

    return flag

def baseline_greedy():
    assigned_workers = []
    greedy_best_assign = {}
    total_greedy_satisfaction = 0

    # get candidate
    for i in task_dict.keys():
        # check if dependency is satisfied
        satisfied=True
        for task_id in task_dict[i]['Dt']:
            if is_task_assigned[task_id] is False:
                satisfied=False
                break
        if satisfied is False :
            continue
        #check if conflict task is assigned
        has_conflict=False
        for task_id in task_dict[i]['Ct']:
            if is_task_assigned[task_id] is True:
                has_conflict=True
                break
        if has_conflict:
            continue
        skill_group[i] = {}
        candidate_list = []
        for j in worker_dict.keys():
            if satisfy_fun(worker_dict[j], task_dict[i])==True and len(
                    list(set(worker_dict[j]['Kw']).intersection(set(task_dict[i]['Kt'])))) != 0:
                candidate_list.append(j)
                candidate_worker.append(j)

        candidate[i] = candidate_list

        if len(candidate_list) > 0:

            greedy_best_assign[i] = []
            candidate_list = list(set(candidate_list).difference(set(assigned_workers)))
            # print(i, candidate_list)
            if len(candidate_list) == 0:
                continue
            # d 每个任务需要的技能array
            # skill_group has xx 能力的工人群
            d = [[] for i in range(len(task_dict[i]['Kt']))]
            for k in range(0, len(task_dict[i]['Kt'])):
                for j in candidate_list:
                    if worker_dict[j]['Kw'][0] == task_dict[i]['Kt'][k]:
                        d[k].append(j)
                skill_group[i][task_dict[i]['Kt'][k]] = d[k]

            worker_start = []
            record = []
            while len(worker_start) == 0:
                # random pick a candidate that available to build the worker_start
                x = choice(candidate_list)
                # record: x has been picked
                record.append(x)
                if x in assigned_workers:
                    continue
                else:
                    worker_start.append(x)
                # 全部尝试后 break
                if len(set(candidate_list).difference(set(record))) == 0:
                    break

            if len(worker_start) > 0:

                worker_start_index = 0
                for r in skill_group[i].keys():
                    if worker_start in skill_group[i][r]:
                        worker_start_index = r
                        break
                # print('start worker:', worker_start)

                best_s = 0
                best_list = [worker_start[0]]
                best_satisfaction = 0
                for r in skill_group[i].keys():
                    if r == worker_start_index:
                        continue
                    else:
                        best_step = []
                        for s in skill_group[i][r]:
                            if s in assigned_workers:
                                continue

                            best_list_copy = copy.deepcopy(best_list)
                            best_list_copy.append(s)
                            price_s = price_score(i, v, best_list_copy)
                            if price_s < 0:
                                continue
                            else:
                                cooperation_s = cooperation_score(best_list_copy)

                            if alpha * (cooperation_s / max_c) + (1 - alpha) * (
                                    price_s / max_p) > best_satisfaction:
                                best_satisfaction = alpha * (cooperation_s / max_c) + (1 - alpha) * (price_s / max_p)
                                best_step = best_list_copy

                        best_list = best_step
                # print(i, best_list, best_satisfaction)

                if len(best_list) != 0:
                    greedy_best_assign[i] = best_list
                    for k in best_list:
                        assigned_workers.append(k)
                    total_greedy_satisfaction += best_satisfaction
                    # update assigned task tag
                    if len(greedy_best_assign[i]) != 0:
                        count_partial += 1
                    if len(greedy_best_assign[i]) == len(task_dict[i]['Kt']):
                        count_full += 1
                        is_task_assigned[i]=True

        # print(i, greedy_best_assign[i])

    flag = conflict_check(greedy_best_assign)
    print('conflict greedy:', flag)
    print('basic greedy satisfaction:', total_greedy_satisfaction)
    print(greedy_best_assign)
    # for i in greedy_best_assign.keys():
    #     if len(greedy_best_assign[i]) != 0:
    #         count_partial += 1
    #     if len(greedy_best_assign[i]) >= len(task_dict[i]['Kt']):
    #         count_full += 1
    print('full:', count_full, 'partial:', count_partial)

if __name__ == '__main__':

    print('=============== Real data =====================')
    task_num=100
    worker_num=1000
    # 读取基本的数据
    with open('cooperation_group.json', 'r') as f_cooperation:
        cooperation_group_1 = json.load(f_cooperation)
    cooperation_group = {}
    for k in cooperation_group_1.keys():
        cooperation_group[int(k)] = cooperation_group_1[k]
    print('whole cooperation arr:', len(cooperation_group.keys()))

    count = 0
    with open('task_noreal.json', 'r') as f_task:
        task_dict_1 = json.load(f_task)
    task_dict = {}
    for k in task_dict_1.keys():
        if count <= task_num:
            count += 1
            task_dict[int(k)] = task_dict_1[k]
    print('task dict:', len(task_dict), task_dict[1])

    count = 0
    with open('worker_noreal.json', 'r') as f_worker:
        worker_dict_1 = json.load(f_worker)
    worker_dict = {}
    for k in worker_dict_1.keys():
        if count <= worker_num:
            count += 1
            worker_dict[int(k)] = worker_dict_1[k]
    print('worker dict:', len(worker_dict))

    with open('cooperation.json', 'r') as f_cooperation:
        cooperation_dict_1 = json.load(f_cooperation)
    cooperation_score_dict = {}
    for k in cooperation_dict_1.keys():
        cooperation_score_dict[int(k)] = cooperation_dict_1[k]
    print('cooperation dict:', len(cooperation_score_dict))


    print('=============== Synthetic data =====================')

    candidate = {}
    best_assign = {}
    R = 0.5
    v = 10
    alpha = 0.5
    beta = 0.05
    candidate_worker = []
    skill_group = {}

    max_c = 100
    print('max cooperation:', max_c)
    max_p = 100
    print('max price:', max_p)

    print('task dict:', task_dict[1])
    print('=========== baseline_greedy algorithm ==============')
    random_time = time.time()
    # 统计分配数
    count_full = 0
    #diy
    # 是否完成分配
    is_task_assigned=[False for i in range(0,len(task_dict)+1)]

    baseline_greedy()
    print('random time:', time.time() - random_time)