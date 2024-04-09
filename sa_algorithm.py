import copy
import math
import random
from random import choice
from itertools import combinations, product
from utils.score_tool import dependency_check,satisfy_check, distance_check, conflict_check, satisfaction, Euclidean_fun, all_combin_worker
from utils.builder_tool import build_assignment, build_skill_group
# 模拟退火

def price_score(task_dict, worker_dict, t, v, worker_list):
    if len(worker_list) > 0:
        dis = 0
        for i in worker_list:
            dis += Euclidean_fun(task_dict[t]['Lt'], worker_dict[i]['Lw'])
        # aa=len(worker_list)/ len(task_dict[t]['Kt'])
        return task_dict[t]['budget'] - dis * v
    return 0

def satisfaction1(task_dict, worker_dict, ti,  v, w_list, alpha, worker_v, max_s,max_p):
    if len(w_list) == 0 or max_p == 0:
        return 0
    total_score = 0
    for w in w_list:
        total_score += worker_dict[w]['score']
    profit_w = price_score(task_dict, worker_dict, ti, v, w_list)

    return alpha*(total_score/len(w_list))/max_s+(1-alpha)*(profit_w/max_p)

def get_free_task(task_dict, arrived_tasks, best_assign):
    """
    计算当前可参与的任务
    """
    free_task = []
    for i in arrived_tasks:
        is_free = True
        for depend_id in task_dict[i]['Dt']:
            if best_assign[depend_id]['assigned'] is False:
                is_free = False
                break
        for conflict_id in task_dict[i]['Ct']:
            if best_assign[conflict_id]['assigned'] is True:
                is_free = False
                break
        if is_free:
            free_task.append(i)
    return free_task

def baseline_sa(task_dict, worker_dict, max_time, move_cost, alpha, max_p,worker_v,reachable_dis,max_s, round_d, p):
    """
    round_d 模拟退火轮次
    p 越大越难接受非更优解
    """
    # result
    best_assign = build_assignment(task_dict)
    # performance
    full_assign = 0
    task_assign_condition = {}
    for i in task_dict.keys():
        task_assign_condition[i] = -1
    # 之前未分配任务-过期任务+本时间点新增的任务
    arrived_tasks = set()
    arrived_workers = set()
    sucess_to_assign_task = set()
    delete_to_assign_task =set()
    for current_time in range(1, max_time+1):  # each time slice try to allocate
        #print(current_time, "...")
        sucess_to_assign_worker = set()
        # for t_id in task_dict.keys():  # discard those meet their deadline set(task_dict[t_id]['Ct']).issubset(sucess_to_assign_task)
        #     if task_dict[t_id]['deadline'] <= current_time or set(
        #             task_dict[t_id]['Ct']) & set(sucess_to_assign_task) or tuple(
        #             task_dict[t_id]['Dt']) in delete_to_assign_task:
        #         delete_to_assign_task.add(t_id)
        # arrived_tasks = after_discard_tasks
        # after_discard_workers = copy.deepcopy(arrived_workers)
        # for w_id in arrived_workers:  # discard those meet their deadline
        #     if worker_dict[w_id]['deadline'] >= current_time:
        #         after_discard_workers.discard(w_id)
        # arrived_workers = after_discard_workers
        for t_id in task_dict.keys():  # add new
            if task_dict[t_id]['arrived_time'] <= current_time and task_dict[t_id][
                'deadline'] >= current_time and t_id not in sucess_to_assign_task:
                arrived_tasks.add(t_id)
        for w_id in worker_dict.keys():  # add new
            if worker_dict[w_id]['arrived_time'] <= current_time and worker_dict[w_id]['deadline'] >= current_time:
                arrived_workers.add(w_id)

        available_tasks = get_free_task(task_dict, arrived_tasks, best_assign)
        skill_group = {}
        candidate_list = {}
        for i in available_tasks:
            task = task_dict[i]
            skill_group[i] = {}
            if i not in candidate_list:
                candidate_list[i] = []
            # denpendency check
            if dependency_check(i, task_assign_condition, task_dict) is False:
                continue
            # check if conflict task is assigned
            if conflict_check(i, task_assign_condition, task_dict):
                continue
            
            
            for w_id in arrived_workers:
                worker = worker_dict[w_id]
                # distance check
                if satisfy_check(task, worker,
                                  task['deadline'], worker['deadline'],
                                  current_time, worker_v,move_cost,reachable_dis) and worker_dict[w_id]['Kw'][0] in task_dict[i]['Kt']:
                    candidate_list[i].append(w_id)
                    if worker_dict[w_id]['Kw'][0] not in skill_group[i]:
                        skill_group[i][worker_dict[w_id]['Kw'][0]] = []
                    skill_group[i][worker_dict[w_id]['Kw'][0]].append(w_id)
            #if len(skill_group[i])<len(task_dict[i]['Kt']):
        # task 的所有 worker set组合
        worker_combine_dict = {}
        for t in available_tasks:
            skillset = []
            for k in skill_group[t].keys():
                if len(skill_group[t][k]) != 0:
                    skillset.append(skill_group[t][k])
            worker_combine_dict[t] = all_combin_worker(skillset)

        assigned_worker_sa = set()
        sa_assign = {}
        # 初始分配
        for t in available_tasks:
            sa_assign[t]=[]
            for wlist in worker_combine_dict[t]: # try all combination
                flag = True
                for s in wlist:
                    if s in assigned_worker_sa:
                        flag = False
                if flag is False:
                    continue
                t_set = wlist
                for k in t_set:
                    assigned_worker_sa.add(k)
                worker_combine_dict[t].remove(t_set)
                sa_assign[t] = t_set
                break
        initial_sa = 0
        new_sa = 0
        for i in sa_assign.keys():
            initial_sa += satisfaction1(
                task_dict, worker_dict, i, move_cost, sa_assign[i], alpha, worker_v, max_s, max_p)
        task_list = list(sa_assign.keys())
        t = round_d + 1
        # 模拟退火
        for i in range(round_d):
            if len(task_list)==0:
                break
            #print("SA-r:", i)
            # random choose task
            random_t = random.choice(task_list)
            temp_set = []
            if len(candidate_list[random_t]) == 0:
                t -= 1
                continue
            # find valid candidate combination
            for temp in worker_combine_dict[random_t]:
                if len(list(set(temp).intersection(set(assigned_worker_sa)))) >= 0:
                    temp_set.append(temp)
            if len(temp_set) == 0:
                continue
            old_set = sa_assign[random_t]
            random_set = choice(temp_set)

            new_s = satisfaction1(
                task_dict, worker_dict, random_t, move_cost, old_set, alpha, worker_v, max_s, max_p)
            # new_s1 = satisfaction1(
            #     task_dict, worker_dict, random_t, move_cost, old_set, alpha, worker_v, max_s, max_p)
            # new_s2 = satisfaction1(
            #     task_dict, worker_dict, random_t, move_cost, old_set, alpha, worker_v, max_s, max_p)
            old_s = satisfaction1(
                task_dict, worker_dict, random_t, move_cost, random_set, alpha, worker_v, max_s, max_p)
            if new_s > old_s or math.exp((new_s - old_s) / t * initial_sa) > p:
                for s in old_set:
                    if s in assigned_worker_sa:
                        assigned_worker_sa.remove(s)
                sa_assign[random_t] = random_set
                for s in random_set:
                    assigned_worker_sa.add(s)
                total_score = 0
                for i in sa_assign.keys():
                    total_score += satisfaction1(task_dict, worker_dict,
                                                i, move_cost, sa_assign[i], alpha, worker_v, max_s, max_p)
                new_sa = total_score
                if new_sa > initial_sa:
                    initial_sa = new_sa

            t -= 1
        # 统计成功完成的工作以及成功分配的工人
        # 将部分分配的工人撤回池中

        # for i in sa_assign:
        #     if len(task_dict[i]['Kt']) == len(sa_assign[i]):
        #         # 统计
        #         sucess_to_assign_task1.add(i)
        #         for w in sa_assign[i]:
        #             sucess_to_assign_worker1.add(w)
        #
        #         best_assign[i]['time'] = task_assign_condition[i] = current_time
        #         best_assign[i]['list'] = sa_assign[i]
        #         best_assign[i]['group'] = {}
        #         best_assign[i]['satisfaction'] = satisfaction1(
        #             task_dict, worker_dict, i, move_cost, sa_assign[i], alpha, worker_v, max_s, max_p)
        #         best_assign[i]['assigned'] = True
        #
        #         for k in task_dict[i]['Kt']:
        #             for w in sa_assign[i]:
        #                 if worker_dict[w]['Kw'][0]==k:
        #                     best_assign[i]['group'][k] = w
        #                     break
        #
        #     else:
        #         best_assign[i]['assigned'] = False
        #
        # confilcts=[]
        # skips=set()
        # for t in sucess_to_assign_task:
        #     task=task_dict[t]
        #     if t in skips:
        #             continue
        #     for c_id in task['Ct']:
        #         if c_id in skips:
        #             continue
        #         if c_id in sucess_to_assign_task:
        #             confilcts.append([t,c_id])
        #             skips.add(c_id)
        #             skips.add(t)
        # for con in confilcts:
        #     TC=[]
        #     for c_id in con:
        #         tc_list1 = []
        #         tc_list=[c_id]
        #         for d_id in tc_list:
        #             if d_id in sucess_to_assign_task:
        #                 tc_list1.append(d_id)
        #         TC.append(tc_list1)
        #     best_select=-1
        #     # cal sat and decide which will remain
        #     best_sat=-1
        #     for idx,tc in enumerate(TC):
        #         cur_sat=0
        #         for id in tc:
        #             cur_sat+=satisfaction1(task_dict, worker_dict,
        #                                             id, move_cost, sa_assign[id], alpha, worker_v, max_s, max_p)
        #         if cur_sat>best_sat:
        #             best_sat=cur_sat
        #             best_select=idx
        #     # discard others except best select
        #     for idx,tc in enumerate(TC):
        #         if idx==best_select:
        #             continue
        #         for i in tc:
        #             _sucess_to_assign_task=set(sucess_to_assign_task)
        #             _sucess_to_assign_task.discard(i)
        #             sucess_to_assign_task=set(_sucess_to_assign_task)
        #             for w in sa_assign[i]:
        #                 sucess_to_assign_worker.discard(w)
        #
        #             best_assign[i]['time'] = 0
        #             best_assign[i]['list'] = []
        #             best_assign[i]['group'] = {}
        #             best_assign[i]['satisfaction'] = 0
        #             best_assign[i]['assigned'] = False
        #             for k in task_dict[i]['Kt']:
        #                 best_assign[i]['group'][k] = 0

        each_sa={}
        for i in sa_assign:
            each_sa[i]=satisfaction1(task_dict, worker_dict,
                           i, move_cost, sa_assign[i], alpha, worker_v, max_s, max_p)
        for k in range(len(sa_assign)):
            id = max(each_sa, key=each_sa.get)
            w_list=sa_assign[id]
            if each_sa[id]>0:
                if conflict_check(id, task_assign_condition, task_dict) is False:
                    if len(task_dict[id]['Kt']) == len(sa_assign[id]) and id not in sucess_to_assign_task and len(list(set(w_list).intersection(set(sucess_to_assign_worker))))==0:
                        # 统计
                        best_assign[id]['time'] = task_assign_condition[id] = current_time
                        best_assign[id]['list'] = sa_assign[id]
                        best_assign[id]['group'] = {}
                        best_assign[id]['satisfaction'] = each_sa[id]
                        best_assign[id]['assigned'] = True
                        sucess_to_assign_task.add(id)
                        for w in sa_assign[id]:
                            trave_dis = Euclidean_fun(task_dict[id]['Lt'], worker_dict[w]['Lw'])
                            worker_dict[w]['arrived_time'] = max(worker_dict[w]['arrived_time'],
                                                                 task_dict[id]['arrived_time']) + trave_dis / worker_v
                            worker_dict[w]['Lw'] = task_dict[id]['Lt']
                            sucess_to_assign_worker.add(w)
                        print(id, sa_assign[id], best_assign[id]['satisfaction'])
            each_sa.pop(id, None)
        arrived_tasks = arrived_tasks.difference(set(sucess_to_assign_task))
        arrived_workers = arrived_workers.difference(
            set(sucess_to_assign_worker))
    # 输出当前分配和总的满意度
    total_sat = 0
    for i in best_assign.keys():
        total_sat += best_assign[i]['satisfaction']
        if best_assign[i]['assigned']== True:
            full_assign+=1
    return task_assign_condition, best_assign, full_assign, total_sat
