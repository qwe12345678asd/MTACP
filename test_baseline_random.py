import json, math, time, itertools, copy, random
from random import choice
from itertools import combinations, product
#from mip import *
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

def random_algorithm():
    assigned_workers = []
    random_best_assign = {}
    #end diy
    for i in batch_task_dict.keys():
        # check if dependency is satisfied
        if is_task_assigned[i] is False:
            satisfied=True
            for task_id in batch_task_dict[i]['Dt']:
                if is_task_assigned[task_id] is False:
                    satisfied=False
                    break
            if satisfied is False :
                continue
            #check if conflict task is assigned
            has_conflict=False
            for task_id in batch_task_dict[i]['Ct']:
                if is_task_assigned[task_id] is True:
                    has_conflict=True
                    break
            if has_conflict:
                continue

            skill_group[i] = {}
            candidate_list = []
            for j in batch_worker_dict.keys():
                if satisfy_fun(batch_worker_dict[j], batch_task_dict[i])==True and len(
                        list(set(batch_worker_dict[j]['Kw']).intersection(set(batch_task_dict[i]['Kt'])))) != 0:
                    candidate_list.append(j)
                    # candidate_worker.append(j)

            candidate[i] = candidate_list

            if len(candidate[i]) > 0:

                random_best_assign[i] = []

                d = [[] for i in range(len(batch_task_dict[i]['Kt']))]
                for k in range(0, len(batch_task_dict[i]['Kt'])):
                    for j in candidate[i]:
                        if batch_worker_dict[j]['Kw'][0] == batch_task_dict[i]['Kt'][k]:
                            d[k].append(j)
                    skill_group[i][batch_task_dict[i]['Kt'][k]] = d[k]

                worker_list = []
                for r in skill_group[i].keys():
                    skill_list = list(set(skill_group[i][r]).difference(set(assigned_workers)))
                    if len(skill_list) != 0:
                        worker_w = choice(skill_list)
                        worker_list.append(worker_w)
                    else:
                        worker_list=[]
                        break;
                if len(worker_list) != 0:
                    for w_id in worker_list:
                        assigned_workers.append(w_id)
                    profit_w = price_score(i, v, worker_list)
                    random_best_assign[i] = worker_list
                    score_w = worker_score(worker_list)
                    cur_satisfaction=alpha * (score_w / max_c) + (1 - alpha) * (profit_w / max_p)
                    print(i, worker_list, cur_satisfaction)
                    total_satisfaction.append(cur_satisfaction)
                    is_task_assigned[i]=True

        # print('candidate workers:', len(candidate[1]), 'total satisfaction:', total_satisfaction)

    return random_best_assign

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

    count = 1
    with open('task_noreal.json', 'r') as f_task:
        task_dict_1 = json.load(f_task)
    task_dict = {}
    for k in task_dict_1.keys():
        if count <= task_num:
            count += 1
            task_dict[int(k)] = task_dict_1[k]
    print('task dict:', len(task_dict), task_dict[1])

    count = 1
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
    R = 1
    v = 10
    alpha = 0.5
    beta = 0.05
    skill_group = {}

    max_c= 100
    max_p = 100


    print('task dict:', task_dict[1])

    print('=========== basic random algorithm ==============')

    w_max_deadline_key = max(worker_dict, key=lambda x: worker_dict[x]['deadline'])
    w_max_deadline=worker_dict[w_max_deadline_key]['deadline']
    t_max_deadline_key = max(task_dict, key=lambda y: task_dict[y]['deadline'])
    t_max_deadline=task_dict[t_max_deadline_key]['deadline']
    final_time = max(w_max_deadline,t_max_deadline)


    i=0
    arrived_task_num=0
    arrived_worker_num=0

    batch_task_dict= {}
    batch_worker_dict= {}
    random_time = time.time()

    # 是否完成分配
    is_task_assigned=[False for i in range(0,len(task_dict)+1)]
    total_satisfaction = []
    # 统计分配数
    count_full = []
    while i<final_time+1:
        arrived_task_count = 0
        for task_id in task_dict.keys():
            if task_id>=arrived_task_num and task_dict[task_id]['arrived_time']<=i:
                arrived_task_count+=1
                batch_task_dict[task_id]=task_dict[task_id]
        arrived_task_num+=arrived_task_count
        arrived_worker_count = 0
        for worker_id in worker_dict.keys():
            if worker_id>=arrived_worker_num and worker_dict[worker_id]['arrived_time'] <= i:
                arrived_worker_count += 1
                batch_worker_dict[worker_id] = worker_dict[worker_id]
        arrived_worker_num+=arrived_worker_count
        if len(batch_task_dict)!=0 and len(batch_worker_dict)!=0:
            random_algorithm()
        # total_satisfaction = sum(total_satisfaction)
        # print('total satisfaction:', total_satisfaction)

        # for task_id in batch_task_dict.keys():
        #     if batch_task_dict[task_id]['arrived_time']>i-10:
        #         batch_task_dict[task_id]=batch_task_dict[task_id]
        # for worker_id in worker_dict.keys():
        #     if worker_dict[worker_id]['arrived_time'] > i-10:
        #         batch_worker_dict[worker_id] = worker_dict[worker_id]
        i = i + 1
    pair_number=len(total_satisfaction)
    total_satisfaction = sum(total_satisfaction)
    print('pair number:', pair_number, 'total satisfaction:', total_satisfaction)
    print('random time:', time.time() - random_time)

    