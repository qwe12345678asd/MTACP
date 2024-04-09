import json, math, time, itertools, copy, random
from random import choice
from itertools import combinations, product
from utils.distance_tool import Euclidean_fun
from random_algorithm import random_algorithm
from greedy_algorithm import greedy_algorithm
from closest_algorithm import closest_algorithm
from sa_algorithm import baseline_sa
from gt_base_algorithm import gt_base_algorithm
from gt_base_its_algorithm import gt_base_its_algorithm
from gt_imp_algorithm import gt_imp_algorithm
from gt_imp_its_algorithm import gt_imp_its_algorithm
# from gt_new_algorithm import gt_new_algorithm
# from gt_new_its_algorithm import gt_new_its_algorithm
from group_greedy import group_greedy_algorithm
def dependency_check(task_dict,best_assign):
    dependency_check_flag=True
    for i in best_assign:
        # print(i)
        if best_assign[i]['assigned'] is True: 
            for d_id in task_dict[i]['Dt']:
                 if best_assign[d_id]['assigned'] is False:
                      dependency_check_flag=False
                      break
    return dependency_check_flag
def conflict_check(task_dict,best_assign):
    conflict_check_flag=True
    for i in best_assign:
        if best_assign[i]['assigned'] is True: 
            for d_id in task_dict[i]['Ct']:
                 if best_assign[d_id]['assigned'] is True:
                      conflict_check_flag=False
                      break
    return conflict_check_flag
if __name__ == '__main__':

    print('=============== Read data =====================')
    # task_num=2500
    # worker_num=500
    # 读取基本的数据
    with open('data/real/DIDI/task5000_3_6_0.5_3_0.5_U.json', 'r') as f_task:
        task_dict_1 = json.load(f_task)
    task_dict = {}
    for k in task_dict_1.keys():
        task_dict[int(k)] = task_dict_1[k]
    print('task dict:', len(task_dict), task_dict[1])

    with open('data/real/DIDI/worker3000_3_6_U.json', 'r') as f_worker:
        _worker_dict = json.load(f_worker)
    worker_dict = {}
    for k in _worker_dict.keys():
        worker_dict[int(k)] = _worker_dict[k]
    print('worker dict:', len(worker_dict), worker_dict[1])

    """parameters setting"""
    candidate = {}
    best_assign = {}
    reachable_dis = 0.6  #0.2 , 0.4 , 0.6 , 0.8 , 1
    move_cost = 30  #10,20,30,40,50
    alpha = 0.5
    beta = 0.05
    max_p = 100 #10,20,30,40,50
    max_s = 100
    worker_v=0.3 #0.1,0.2,0.3,0.4,0.5
    max_time=worker_dict[len(worker_dict)-1]['deadline']
    # for i in task_dict.keys():
    #     for j in worker_dict.keys():
    #         min_p = task_dict[i]['budget'] - Euclidean_fun(task_dict[i]['Lt'], worker_dict[j]['Lw']) * v
    #         if min_p > max_p:
    #             max_p = min_p
    print('max price is :', max_p)

    print('=========== basic random algorithm ==============')
    random_time = time.time()
    task_assign_condition,best_assign,full_assign,total_sat=random_algorithm(
        task_dict,worker_dict,max_time,move_cost,alpha,max_s,max_p,worker_v,reachable_dis)
    print('random time:', time.time() - random_time)
    #print("task_assign_condition: ",task_assign_condition)
    print("full_assign_cnt: ",full_assign)
    print("total_satisfaction: ",total_sat)
    flag=True
    task_set=set()
    for i in best_assign:
        for w in best_assign[i]['list']:
            if w in task_set:
                flag=False
                break
            task_set.add(w)
    print('duplicate check', "pass" if flag else "fail")
    print("denpedency check","pass" if dependency_check(task_dict,best_assign) else "fail")              
    print("conflict check","pass" if conflict_check(task_dict,best_assign) else "fail")              
    with open('./results/random_best_assign.json', 'w', encoding='utf-8') as fp_ba:
            json.dump(best_assign, fp_ba)


    print('=========== greedy algorithm ==============')
    # task_num=2000
    # worker_num=500
    # 读取基本的数据
    with open('data/real/DIDI/task5000_3_6_0.5_3_0.5_U.json', 'r') as f_task:
        task_dict_1 = json.load(f_task)
    task_dict = {}
    for k in task_dict_1.keys():
        task_dict[int(k)] = task_dict_1[k]
    print('task dict:', len(task_dict), task_dict[1])

    with open('data/real/DIDI/worker3000_3_6_U.json', 'r') as f_worker:
        _worker_dict = json.load(f_worker)
    worker_dict = {}
    for k in _worker_dict.keys():
        worker_dict[int(k)] = _worker_dict[k]
    print('worker dict:', len(worker_dict), worker_dict[1])

    random_time = time.time()
    task_assign_condition,best_assign,full_assign,total_sat=greedy_algorithm(
        task_dict,worker_dict,max_time,move_cost,alpha,max_s,max_p,worker_v,reachable_dis)
    print('greedy time:', time.time() - random_time)
    #print("task_assign_condition: ",task_assign_condition)
    print("full_assign_cnt: ",full_assign)
    print("total_satisfaction: ",total_sat)
    flag=True
    task_set=set()
    for i in best_assign:
        for w in best_assign[i]['list']:
            if w in task_set:
                flag=False
                break
            task_set.add(w)
    print('flag:', flag)
    print("denpedency check","pass" if dependency_check(task_dict,best_assign) else "fail")              
    print("conflict check","pass" if conflict_check(task_dict,best_assign) else "fail") 
    with open('./results/greedy_best_assign.json', 'w', encoding='utf-8') as fp_ba:
            json.dump(best_assign, fp_ba)

    print('=========== closest algorithm ==============')
    # task_num=1000
    # worker_num=500
    # 读取基本的数据
    with open('data/real/DIDI/task5000_3_6_0.5_3_0.5_U.json', 'r') as f_task:
        task_dict_1 = json.load(f_task)
    task_dict = {}
    for k in task_dict_1.keys():
        task_dict[int(k)] = task_dict_1[k]
    print('task dict:', len(task_dict), task_dict[1])

    with open('data/real/DIDI/worker3000_3_6_U.json', 'r') as f_worker:
        _worker_dict = json.load(f_worker)
    worker_dict = {}
    for k in _worker_dict.keys():
        worker_dict[int(k)] = _worker_dict[k]
    print('worker dict:', len(worker_dict), worker_dict[1])

    random_time = time.time()
    task_assign_condition,best_assign,full_assign,total_sat=closest_algorithm(
        task_dict,worker_dict,max_time,move_cost,alpha,max_s,max_p,worker_v,reachable_dis)
    print('closest time:', time.time() - random_time)
    #print("task_assign_condition: ",task_assign_condition)
    print("full_assign_cnt: ",full_assign)
    print("total_satisfaction: ",total_sat)
    flag=True
    task_set=set()
    for i in best_assign:
        for w in best_assign[i]['list']:
            if w in task_set:
                flag=False
                break
            task_set.add(w)
    print('flag:', flag)
    print("denpedency check","pass" if dependency_check(task_dict,best_assign) else "fail")
    print("conflict check","pass" if conflict_check(task_dict,best_assign) else "fail")
    with open('./results/closest_best_assign.json', 'w', encoding='utf-8') as fp_ba:
            json.dump(best_assign, fp_ba)


    print('=========== GT NEW improve algorithm ==============')
    # task_num=1000
    # worker_num=500
    # 读取基本的数据
    with open('data/real/DIDI/task5000_3_6_0.5_3_0.5_U.json', 'r') as f_task:
        task_dict_1 = json.load(f_task)
    task_dict = {}
    for k in task_dict_1.keys():
        task_dict[int(k)] = task_dict_1[k]
    print('task dict:', len(task_dict), task_dict[1])

    with open('data/real/DIDI/worker3000_3_6_U.json', 'r') as f_worker:
        _worker_dict = json.load(f_worker)
    worker_dict = {}
    for k in _worker_dict.keys():
        worker_dict[int(k)] = _worker_dict[k]
    print('worker dict:', len(worker_dict), worker_dict[1])

    # max_time=task_dict[len(task_dict)-1]['deadline']
    random_time = time.time()
    task_assign_condition,best_assign,full_assign,total_sat,worker_assign_time=gt_imp_algorithm(
        task_dict,worker_dict,max_time,move_cost,alpha,max_s,max_p,worker_v,reachable_dis)
    print('GT imp time:', time.time() - random_time)
    flag=True
    task_set=set()
    for i in best_assign:
        for w in best_assign[i]['list']:
            if w in task_set:
                flag=False
                # print(w)
            task_set.add(w)

    print("full_assign_cnt: ",full_assign)
    print("total_satisfaction: ",total_sat)
    print('duplicate check', "pass" if flag else "fail")
    print("denpedency check","pass" if dependency_check(task_dict,best_assign) else "fail")
    print("conflict check","pass" if conflict_check(task_dict,best_assign) else "fail")

    #print("worker_assign_time",worker_assign_time)
    # for tl in worker_assign_time.values():
    #     if len(set(tl))>1:
    #         print("find!!")
    with open('./results/GT_imp_best_assign.json', 'w', encoding='utf-8') as fp_ba:
            json.dump(best_assign, fp_ba)

    print('=========== GT New improve its algorithm ==============')
    # task_num=1000
    # worker_num=500
    # 读取基本的数据
    with open('data/real/DIDI/task5000_3_6_0.5_3_0.5_U.json', 'r') as f_task:
        task_dict_1 = json.load(f_task)
    task_dict = {}
    for k in task_dict_1.keys():
        task_dict[int(k)] = task_dict_1[k]
    print('task dict:', len(task_dict), task_dict[1])

    with open('data/real/DIDI/worker3000_3_6_U.json', 'r') as f_worker:
        _worker_dict = json.load(f_worker)
    worker_dict = {}
    for k in _worker_dict.keys():
        worker_dict[int(k)] = _worker_dict[k]
    print('worker dict:', len(worker_dict), worker_dict[1])

    beta=0.1
    random_time = time.time()
    task_assign_condition,best_assign,full_assign,total_sat,worker_assign_time=gt_imp_its_algorithm(
        task_dict,worker_dict,max_time,move_cost,alpha,max_s,max_p,worker_v,reachable_dis,beta)
    print('GT_imp_its time:', time.time() - random_time)
    flag=True
    task_set=set()
    for i in best_assign:
        for w in best_assign[i]['list']:
            if w in task_set:
                flag=False
                # print(w)
            task_set.add(w)

    print("full_assign_cnt: ",full_assign)
    print("total_satisfaction: ",total_sat)
    print('duplicate check', "pass" if flag else "fail")
    print("denpedency check","pass" if dependency_check(task_dict,best_assign) else "fail")
    print("conflict check","pass" if conflict_check(task_dict,best_assign) else "fail")

    #print("worker_assign_time",worker_assign_time)
    # for tl in worker_assign_time.values():
    #     if len(set(tl))>1:
    #         print("find!!")
    with open('./results/GT_imp_its_best_assign.json', 'w', encoding='utf-8') as fp_ba:
            json.dump(best_assign, fp_ba)

    print('=========== GT base algorithm ==============')
    # task_num=1000
    # worker_num=500
    # 读取基本的数据
    with open('data/real/DIDI/task5000_3_6_0.5_3_0.5_U.json', 'r') as f_task:
        task_dict_1 = json.load(f_task)
    task_dict = {}
    for k in task_dict_1.keys():
        task_dict[int(k)] = task_dict_1[k]
    print('task dict:', len(task_dict), task_dict[1])

    with open('data/real/DIDI/worker3000_3_6_U.json', 'r') as f_worker:
        _worker_dict = json.load(f_worker)
    worker_dict = {}
    for k in _worker_dict.keys():
        worker_dict[int(k)] = _worker_dict[k]
    print('worker dict:', len(worker_dict), worker_dict[1])

    random_time = time.time()
    task_assign_condition,best_assign,full_assign,total_sat,worker_assign_time=gt_base_algorithm(
        task_dict,worker_dict,max_time,move_cost,alpha,max_s,max_p,worker_v,reachable_dis)
    print('GT time:', time.time() - random_time)
    flag=True
    task_set=set()
    for i in best_assign:
        for w in best_assign[i]['list']:
            if w in task_set:
                flag=False
                # print(w)
            task_set.add(w)

    print("full_assign_cnt: ",full_assign)
    print("total_satisfaction: ",total_sat)
    print('duplicate check', "pass" if flag else "fail")
    print("denpedency check","pass" if dependency_check(task_dict,best_assign) else "fail")
    print("conflict check","pass" if conflict_check(task_dict,best_assign) else "fail")

    #print("worker_assign_time",worker_assign_time)  
    # for tl in worker_assign_time.values():
    #     if len(set(tl))>1:
    #         print("find!!")
    with open('./results/GT_best_assign.json', 'w', encoding='utf-8') as fp_ba:
            json.dump(best_assign, fp_ba)

    print('=========== GT base its algorithm ==============')
    # task_num=1000
    # worker_num=500
    # 读取基本的数据
    with open('data/real/DIDI/task5000_3_6_0.5_3_0.5_U.json', 'r') as f_task:
        task_dict_1 = json.load(f_task)
    task_dict = {}
    for k in task_dict_1.keys():
        task_dict[int(k)] = task_dict_1[k]
    print('task dict:', len(task_dict), task_dict[1])

    with open('data/real/DIDI/worker3000_3_6_U.json', 'r') as f_worker:
        _worker_dict = json.load(f_worker)
    worker_dict = {}
    for k in _worker_dict.keys():
        worker_dict[int(k)] = _worker_dict[k]
    print('worker dict:', len(worker_dict), worker_dict[1])

    beta=0.2
    random_time = time.time()
    task_assign_condition,best_assign,full_assign,total_sat,worker_assign_time=gt_base_its_algorithm(
        task_dict,worker_dict,max_time,move_cost,alpha,max_s,max_p,worker_v,reachable_dis,beta)
    print('GT_ITS time:', time.time() - random_time)
    flag=True
    task_set=set()
    for i in best_assign:
        for w in best_assign[i]['list']:
            if w in task_set:
                flag=False
                # print(w)
            task_set.add(w)

    print("full_assign_cnt: ",full_assign)
    print("total_satisfaction: ",total_sat)
    print('duplicate check', "pass" if flag else "fail")
    print("denpedency check","pass" if dependency_check(task_dict,best_assign) else "fail")
    print("conflict check","pass" if conflict_check(task_dict,best_assign) else "fail")

    #print("worker_assign_time",worker_assign_time)  
    # for tl in worker_assign_time.values():
    #     if len(set(tl))>1:
    #         print("find!!")
    with open('./results/GT_ITS_best_assign.json', 'w', encoding='utf-8') as fp_ba:
            json.dump(best_assign, fp_ba)


    print('=========== SA algorithm ==============')
    # task_num=1000
    # worker_num=500
    # 读取基本的数据
    with open('data/real/DIDI/task5000_3_6_0.5_3_0.5_U.json', 'r') as f_task:
        task_dict_1 = json.load(f_task)
    task_dict = {}
    for k in task_dict_1.keys():
        task_dict[int(k)] = task_dict_1[k]
    print('task dict:', len(task_dict), task_dict[1])

    with open('data/real/DIDI/worker3000_3_6_U.json', 'r') as f_worker:
        _worker_dict = json.load(f_worker)
    worker_dict = {}
    for k in _worker_dict.keys():
        worker_dict[int(k)] = _worker_dict[k]
    print('worker dict:', len(worker_dict), worker_dict[1])

    round_d = 2000
    p = 0.001
    random_time = time.time()
    # task_assign_condition,best_assign,full_assign,total_sat=baseline_sa(
    #     task_dict,worker_dict,max_time,move_cost,alpha,max_p,worker_v,reachable_dis,max_s,round_d,0.001)
    print('SA time:', time.time() - random_time)
    #print("task_assign_condition: ",task_assign_condition)
    print("full_assign_cnt: ",full_assign)
    print("total_satisfaction: ",total_sat)
    flag=True
    task_set=set()
    for i in best_assign:
        for w in best_assign[i]['list']:
            if w in task_set:
                flag=False
            task_set.add(w)
    print('duplicate check', "pass" if flag else "fail")
    print("denpedency check","pass" if dependency_check(task_dict,best_assign) else "fail")              
    print("conflict check","pass" if conflict_check(task_dict,best_assign) else "fail") 
   
    with open('./results/SA_best_assign.json', 'w', encoding='utf-8') as fp_ba:
            json.dump(best_assign, fp_ba)



    print('=========== group greedy algorithm ==============')
    # task_num=1000
    # worker_num=500
    # 读取基本的数据
    with open('data/real/DIDI/task5000_3_6_0.5_3_0.5_U.json', 'r') as f_task:
        task_dict_1 = json.load(f_task)
    task_dict = {}
    for k in task_dict_1.keys():
        task_dict[int(k)] = task_dict_1[k]
    print('task dict:', len(task_dict), task_dict[1])

    with open('data/real/DIDI/worker3000_3_6_U.json', 'r') as f_worker:
        _worker_dict = json.load(f_worker)
    worker_dict = {}
    for k in _worker_dict.keys():
        worker_dict[int(k)] = _worker_dict[k]
    print('worker dict:', len(worker_dict), worker_dict[1])

    random_time = time.time()
    task_assign_condition,best_assign,full_assign,total_sat=group_greedy_algorithm(
        task_dict,worker_dict,max_time,move_cost,alpha,max_p,worker_v,reachable_dis,max_s)
    print('greedy time:', time.time() - random_time)
    #print("task_assign_condition: ",task_assign_condition)
    print("full_assign_cnt: ",full_assign)
    print("total_satisfaction: ",total_sat)
    flag=True
    task_set=set()
    for i in best_assign:
        for w in best_assign[i]['list']:
            if w in task_set:
                flag=False
                break
            task_set.add(w)
    print('duplicate check', "pass" if flag else "fail")
    print("denpedency check","pass" if dependency_check(task_dict,best_assign) else "fail")              
    print("conflict check","pass" if conflict_check(task_dict,best_assign) else "fail") 
    with open('./results/group_greedy_best_assign.json', 'w', encoding='utf-8') as fp_ba:
            json.dump(best_assign, fp_ba)
