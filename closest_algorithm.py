import copy
import random
from random import choice
from itertools import combinations, product
from utils.score_tool import dependency_check,satisfy_check,distance_check,conflict_check,Euclidean_fun,satisfaction

def calculate_dis(task_dict, worker_dict, t, move_cost, cooperation_workers, alpha, worker_v, max_s, max_p):
    """satisfaction值=alpha * (合作总的工人声誉分 /合作工人数目) + (1 - alpha) * (profit_w / max_p)"""

    if len(cooperation_workers) == 0 or max_p == 0:
        return 0
    total_dis = 0
    for w in cooperation_workers:
        # total_score += worker_dict[w]['score']
        trave_dis = Euclidean_fun(task_dict[t]['Lt'], worker_dict[w]['Lw'])
        total_dis += trave_dis
    # profit_w = price_score(task_dict, worker_dict, t, v, cooperation_workers)
    return total_dis


def update_task_worker(task_dict,worker_dict,arrived_tasks,arrived_workers,current_time,sucess_to_assign_task,delete_to_assign_task):
    # 更新每轮的可用任务和工人
    for t_id in task_dict.keys():  # discard those meet their deadline set(task_dict[t_id]['Ct']).issubset(sucess_to_assign_task)
        if task_dict[t_id]['deadline'] < current_time or tuple(
            task_dict[t_id]['Dt']) in delete_to_assign_task:
            delete_to_assign_task.add(t_id)
    # arrived_tasks=after_discard_tasks
    # after_discard_workers=copy.deepcopy(arrived_workers)
    # for w_id in arrived_workers:  # discard those meet their deadline
    #     if worker_dict[w_id]['deadline'] >= current_time:
    #         after_discard_workers.discard(w_id)
    # arrived_workers=after_discard_workers
    for t_id in task_dict.keys():  # add new
        if task_dict[t_id]['arrived_time'] <= current_time and task_dict[t_id]['deadline'] >= current_time and \
                t_id not in sucess_to_assign_task and tuple(task_dict[t_id]['Ct']) not in sucess_to_assign_task \
                and tuple(
            task_dict[t_id]['Dt']) not in delete_to_assign_task and t_id not in delete_to_assign_task:
            arrived_tasks.add(t_id)
    for w_id in worker_dict.keys():  # add new
        if worker_dict[w_id]['arrived_time'] <= current_time and worker_dict[w_id]['deadline'] >= current_time:
            arrived_workers.add(w_id)
    return arrived_tasks,arrived_workers

def closest_algorithm(task_dict, worker_dict, max_time, move_cost, alpha, max_s,max_p,worker_v,reachable_dis):
    """
    v price_score 的超参数
    alpha,max_p satifaction计算的超参数
    """
    # performance
    full_assign = 0
    task_assign_condition = {}
    for i in task_dict.keys():
        task_assign_condition[i] = -1
    # final task_workers assignment
    best_assign = {}
    for i in task_dict.keys():
        best_assign[i] = {}
        best_assign[i]['list'] = []
        best_assign[i]['group'] = {}
        best_assign[i]['satisfaction'] = 0
        best_assign[i]['assigned']=False
        for k in task_dict[i]['Kt']:
            best_assign[i]['group'][k] = 0

    # 之前未分配任务-过期任务+本时间点新增的任务
    arrived_tasks = set()
    arrived_workers = set()
    sucess_to_assign_task = []
    delete_to_assign_task = set()
    for current_time in range(1, max_time+1):  # each time slice try to allocate
        arrived_tasks = set()
        arrived_workers = set()
        assigned_workers = set()
        sucess_to_assign_worker = set()
        arrived_tasks,arrived_workers=update_task_worker(task_dict,worker_dict,arrived_tasks,arrived_workers,current_time,sucess_to_assign_task,delete_to_assign_task)
        disrupted_tasks = list(arrived_tasks)
        random.shuffle(disrupted_tasks)
        skill_group = {}  # 每个任务需要的技能，技能对应的候选人
        for i in task_dict.keys():
            skill_group[i] = {}
            d = [[] for i in range(len(task_dict[i]['Kt']))]
            for k in range(0, len(task_dict[i]['Kt'])):
                skill_group[i][task_dict[i]['Kt'][k]] = d[k]

        for i in disrupted_tasks:  # 任务找工人
            task = task_dict[i]

            # denpendency check
            if dependency_check(i, task_assign_condition, task_dict) is False:
                continue
            # check if conflict task is assigned
            if conflict_check(i, task_assign_condition, task_dict):
                continue
            candidate_worker = []
            for w_id in arrived_workers:
                worker = worker_dict[w_id]
                # distance check
                if satisfy_check(task, worker,
                                  task['deadline'], worker['deadline'],
                                  current_time, worker_v,move_cost,reachable_dis):
                    candidate_worker.append(w_id)
            if len(candidate_worker) == 0:
                continue
            for k in range(0, len(task['Kt'])):
                for j in candidate_worker:
                    for s in range(0, len(worker_dict[j]['Kw'])):
                        if worker_dict[j]['Kw'][s] == task['Kt'][k]:
                            skill_group[i][task['Kt'][k]].append(j)

            # print(i, skill_group[i])
            worker_list = []
            success_assign_flag=False
            for r in skill_group[i].keys():
                skill_list = list(
                    set(skill_group[i][r]).difference(assigned_workers))
                if len(skill_list) != 0:
                    """greedy pick"""
                    greedy_best_pick=-1
                    greedy_best_dis=99
                    for current_worker in skill_list:
                        worker_list.append(current_worker)
                        current_dis=calculate_dis(task_dict, worker_dict, i, move_cost, worker_list, alpha,worker_v, max_s, max_p)
                        if current_dis < greedy_best_dis:
                            greedy_best_dis=current_dis
                            greedy_best_pick=current_worker
                        worker_list.pop()
                    worker_list.append(greedy_best_pick)
                    assigned_workers.add(greedy_best_pick)
            if len(worker_list) == len(task['Kt']):
                success_assign_flag=True
            else:
                for k in worker_list:
                    assigned_workers.discard(k)
            if success_assign_flag:
                task_assign_condition[i] = current_time
                full_assign += 1
                sucess_to_assign_task.append(i)
                best_assign[i]['list'] = worker_list
                # NEED UPDATE
                for w in worker_list:
                    best_assign[i]['group'][worker_dict[w]['Kw'][0]] = w
                    sucess_to_assign_worker.add(w)
                    assigned_workers.update(sucess_to_assign_worker)
                cur_task_satisfaction=satisfaction(task_dict, worker_dict, i, move_cost, worker_list, alpha,worker_v, max_s, max_p,current_time)
                best_assign[i]['satisfaction'] = cur_task_satisfaction
                    # 更新分配情况
                best_assign[i]['assigned']=True
                # print(i, worker_list)
                arrived_tasks = arrived_tasks.difference(set(sucess_to_assign_task))
                arrived_workers = arrived_workers.difference(
                    set(sucess_to_assign_worker))
        # print(worker_dict[1])
    # 输出当前分配和总的满意度
    total_sat=0
    for i in best_assign.keys():
        total_sat+=best_assign[i]['satisfaction']
    return task_assign_condition, best_assign, full_assign,total_sat

