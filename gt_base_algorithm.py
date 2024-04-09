import json, math, time, itertools, copy, random
from utils.score_tool import dependency_check, satisfy_check, distance_check, conflict_check, satisfaction
from utils.distance_tool import Euclidean_fun
from closest_algorithm import closest_algorithm
from random import choice
import copy


def check_nash_equilibrium(before_strategy, after_strategy):

    return before_strategy == after_strategy


def check_nash_equilibrium_ITS(task_dict, worker_dict, tasks, before_strategy, after_strategy, beta,v, alpha, worker_v,max_s, max_p):
    if -1 in before_strategy:
        return True
    before_utility = utility(task_dict, worker_dict, tasks, before_strategy,v, alpha, worker_v,max_s,max_p)
    after_utility = utility(task_dict, worker_dict, tasks, after_strategy,v, alpha,worker_v,max_s, max_p)
    aa=abs(after_utility - before_utility)
    return aa>beta*before_utility

def price_score(task_dict, worker_dict, t, v, worker_list):
    if len(worker_list) > 0:
        dis = 0
        for i in worker_list:
            dis += Euclidean_fun(task_dict[t]['Lt'], worker_dict[i]['Lw'])
        aa=len(worker_list)/ len(task_dict[t]['Kt'])
        return task_dict[t]['budget']*aa - dis * v
    return 0

def satisfaction1(task_dict, worker_dict, ti,  v, w_list, alpha, worker_v, max_s,max_p):
    if len(w_list) == 0 or max_p == 0:
        return 0
    total_score = 0
    for w in w_list:
        total_score += worker_dict[w]['score']
    profit_w = price_score(task_dict, worker_dict, ti, v, w_list)

    return alpha*(total_score/len(w_list))/max_s+(1-alpha)*(profit_w/max_p)


def utility(task_dict, worker_dict, tasks, strategy, v, alpha,worker_v, max_s, max_p):
    u = 0
    task_pre_assign = {}
    for ti in tasks:
        task_pre_assign[ti]={}
        task_pre_assign[ti]['list'] = []
        task_pre_assign[ti]['group'] = {}
        for ki in task_dict[ti]['Kt']:
            task_pre_assign[ti]['group'][ki] = 0

    for i in strategy:
        if(strategy[i]==0):
            continue
        task_pre_assign[strategy[i]]['list'].append(i)
        task_pre_assign[strategy[i]]['group'][worker_dict[i]['Kw'][0]] = i
    for ai in task_pre_assign:
        if len(task_pre_assign[ai]['list']) != len(task_dict[ai]['Kt']):
            continue
        flag = True
        groups = task_pre_assign[ai]['group']
        for g in groups:
            if groups[g] == 0:
                flag = False
                break
        if flag:
            u += task_utility_imp_its(ai,task_dict, worker_dict, v,
                              task_pre_assign[ai]['list'], alpha, worker_v, max_s, max_p)
    return u


def price_score_imp_its(task_dict, worker_dict, t, v, worker_list,TC):
    related_tasks = get_related_tasks(TC, t)
    if len(worker_list) > 0:
        dis = 0
        for i in worker_list:
            dis += Euclidean_fun(task_dict[t]['Lt'], worker_dict[i]['Lw']) #0.42
        cur_task_cost=task_dict[t]['budget']/ len(task_dict[t]['Kt']) #- dis * v
        other_task_cost=0
        for ti in related_tasks:
            other_task_cost+=task_dict[ti]['budget']/ len(task_dict[ti]['Kt'])
        final_task_cost= (cur_task_cost+other_task_cost)/ (len(related_tasks)+1)
        if len(related_tasks)>0:
            return final_task_cost - dis * v * 10
        else:
            return cur_task_cost  - dis * v * 10
    return 0

def task_utility_imp_its(ti, task_dict, worker_dict, v, w_list, alpha, worker_v, max_s,max_p,TC):
    related_tasks = get_related_tasks(TC, ti)
    if len(w_list) == 0 or max_p == 0:
        return 0
    min_score = 100
    total_score=0
    for w in w_list:
        total_score+= worker_dict[w]['score']
        if min_score>total_score:
            min_score=total_score
    profit_w = price_score_imp_its(task_dict, worker_dict, ti, v, w_list,TC)
    return alpha * ( min_score/(len(related_tasks)+1)) / max_s + (1 - alpha) * (profit_w / max_p)

def price_score_imp1(task_dict, worker_dict, t, v, worker_list,TC):
    related_tasks = get_related_tasks(TC, t)
    count=0
    if len(worker_list) > 0:
        dis = 0
        for i in worker_list:
            dis += Euclidean_fun(task_dict[t]['Lt'], worker_dict[i]['Lw']) #0.42
        aa=len(task_dict[t]['Kt'])
        cur_task_cost=task_dict[t]['budget']/ (len(task_dict[t]['Kt'])) #- dis * v
        count+=len(task_dict[t]['Kt'])
        other_task_cost=0
        for ti in related_tasks:
            other_task_cost+=task_dict[ti]['budget']/ (len(task_dict[ti]['Kt']))
            count += len(task_dict[ti]['Kt'])
        final_task_cost= (cur_task_cost+other_task_cost)
        if len(related_tasks)>0:
            return final_task_cost*0.1 - dis * v * count*0.05
        else:
            return cur_task_cost*0.1- dis * v*0.05
    return 0


def price_score_imp(task_dict, worker_dict, t, v, cur_worker):

    dis = Euclidean_fun(task_dict[t]['Lt'], worker_dict[cur_worker]['Lw']) #0.42
    cur_task_cost=task_dict[t]['budget']/len(task_dict[t]['Kt'])#- dis * v

    return cur_task_cost- dis * v

def task_utility(ti, task_dict, worker_dict, v, cur_worker, alpha, worker_v, max_s,max_p):

    total_score = worker_dict[cur_worker]['score']/len(task_dict[ti]['Kt'])
    total_score=total_score

    profit_w = price_score_imp(task_dict, worker_dict, ti, v, cur_worker)

    return alpha*(total_score)/max_s+(1-alpha)*(profit_w/max_p)

def build_TC(task_dict,arrived_tasks,best_assign):
    tl=sorted(list(arrived_tasks))
    TC={}
    marked={}
    for item in tl:
        marked[item]=False
    for _, t in enumerate(tl[::-1]):
        if marked[t] is True:
            continue
        task=task_dict[t]
        tc=[]
        is_valid=True
        for depend_t in task['Dt']:
            if depend_t in arrived_tasks:
                tc.append(depend_t)
        if is_valid:
            tc.append(t)
            TC[t]=sorted(tc)
        marked[t]=True
    return TC


def get_related_tasks(dependency_graph, task):
    visited = set()
    dependent_tasks = set()

    def dfs(current_task):
        visited.add(current_task)
        for dependent_task in dependency_graph.get(current_task, []):
            if dependent_task not in visited:
                dfs(dependent_task)
                dependent_tasks.add(dependent_task)
    dfs(task)
    return dependent_tasks

def get_dependency_set(dependency_graph, task):
    dependencies = set()

    def dfs(current_task, visited):
        if current_task not in visited:
            visited.add(current_task)
            for dependent_task in dependency_graph.get(current_task, []):
                dependencies.add(dependent_task)
                dfs(dependent_task, visited)

    dfs(task, set())
    return dependencies

def random_strategy(task_dict, worker_dict, workers, tasks,
                    current_time, worker_v, v, alpha,max_s, max_p,move_cost,reachable_dis,
                    task_pre_assign,strategy):
    """获得一个符合要求的随机分配策略"""
    disrupted_tasks=list(copy.deepcopy(tasks))
    random.shuffle(disrupted_tasks)
    skill_group = {}  # 每个任务需要的技能，技能对应的候选人（过期工人？）
    for i in tasks:
        skill_group[i] = {}
        d = [[] for i in range(len(task_dict[i]['Kt']))]
        for k in range(0, len(task_dict[i]['Kt'])):
            skill_group[i][task_dict[i]['Kt'][k]] = d[k]

    assigned_workers=set()
    assigned_tasks = set()
    _assigned_workers=set()
    for i in task_pre_assign:
        # print(i)# 任务找工人
        # related_tasks = get_order_related_tasks(TC, i)
        # for re_task_id in related_tasks:
        success_assign_flag = False
        satisfied = True
        for task_id in task_dict[i]['Dt']:
            if task_id not in assigned_tasks:
                satisfied = False
                break
        if satisfied:
            task = task_dict[i]
            candidate_worker = []
            for w_id in workers:
                worker = worker_dict[w_id]
                # distance check
                if satisfy_check(task, worker,
                                    task['deadline'], worker['deadline'],
                                    current_time, worker_v,move_cost,reachable_dis):
                    candidate_worker.append(w_id)
            if len(candidate_worker) == 0:
                break
            for k in range(0, len(task['Kt'])):
                for j in candidate_worker:
                    for s in range(0, len(worker_dict[j]['Kw'])):
                        if worker_dict[j]['Kw'][s] == task['Kt'][k]:
                            skill_group[i][task['Kt'][k]].append(j)
            worker_list = []
            success_assign_flag = False
            for r in skill_group[i].keys():
                assigned_workers.update(_assigned_workers)
                skill_list = list(
                    set(skill_group[i][r]).difference(assigned_workers))
                if len(skill_list) != 0:
                    """random pick a worker"""
                    worker_w = choice(skill_list)
                    worker_list.append(worker_w)
                    _assigned_workers.add(worker_w)
                    if len(worker_list) == len(task['Kt']):
                        assigned_tasks.add(i)
                        success_assign_flag = True
                        break
            # update strategy
            if success_assign_flag:
                original_list = task_pre_assign[i]['list']
                task_pre_assign[i]['list']=original_list+worker_list
                for pw in worker_list:
                    assigned_workers.add(pw)
                    strategy[pw]=i
                # count=0
                # task_pre_assign[i]['group'][s_skill] = worker_list
                aa=0
                index = 0
                # for s_k in  task_pre_assign[i]['group']:
                filtered_keys = {key for key in task_pre_assign[i]['group'] if 10*index <key < 10*index+10}
                for s_skill in filtered_keys:
                    task_pre_assign[i]['group'][s_skill]=worker_list[aa]
                    aa+=1
            # else:
            #     break
    return task_pre_assign,strategy


def calculate_dis(task_dict, worker_dict, t, v, cooperation_workers, alpha, worker_v, max_s, max_p):
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


def choose_closest_worker(task_dict,ai,worker_dict,arrived_workers,current_time,worker_v,reachable_dis, v,alpha, max_s, max_p, sucess_to_assign_worker):
    skill_group = {}  # 每个任务需要的技能，技能对应的候选人
    d = [[] for i in range(len(task_dict[ai]['Kt']))]
    for k in range(0, len(task_dict[ai]['Kt'])):
        skill_group[task_dict[ai]['Kt'][k]] = d[k]

    task=task_dict[ai]
    candidate_worker = []
    worker_list = []
    success_assign_flag = False
    for w_id in arrived_workers:
        worker = worker_dict[w_id]
        # distance check
        if satisfy_check(task, worker,
                         task['deadline'], worker['deadline'],
                         current_time, worker_v,v,reachable_dis):
            candidate_worker.append(w_id)
    if len(candidate_worker) != 0:
        for k in range(0, len(task['Kt'])):
            for j in candidate_worker:
                if worker_dict[j]['Kw'][0] == task['Kt'][k]:
                    skill_group[task['Kt'][k]].append(j)
        # print(i, skill_group[i])

        for r in skill_group.keys():
            skill_list = list(
                set(skill_group[r]).difference(sucess_to_assign_worker))
            """greedy pick"""
            if len(skill_list) != 0:
                greedy_best_pick = -1
                greedy_best_dis = 99
                for current_worker in skill_list:
                    worker_list.append(current_worker)
                    current_dis = calculate_dis(task_dict, worker_dict, ai, v, worker_list, alpha, worker_v, max_s, max_p)
                    # current_cost = calculate_cost(task_dict, worker_dict, ai, v, worker_list, alpha, worker_v, max_s, max_p)
                    if current_dis < greedy_best_dis:
                        greedy_best_dis = current_dis
                        greedy_best_pick = current_worker
                    worker_list.pop()
                worker_list.append(greedy_best_pick)
                if len(worker_list) == len(task['Kt']):
                    worker_list = worker_list
                    success_assign_flag=True
                else:
                    success_assign_flag=False
    return success_assign_flag, worker_list


def ability_check(worker, task):
    return True if set(worker['Kw']).intersection(set(task['Kt'])) else False

def gt_base(task_dict, worker_dict, workers, tasks, current_time, worker_v,reachable_dis, move_cost, alpha,max_s, max_p):
    strategy = {}
    task_pre_assign = {}
    for ti in tasks:
        task_pre_assign[ti]={}
        task_pre_assign[ti]['list'] = []
        task_pre_assign[ti]['group'] = {}
        for ki in task_dict[ti]['Kt']:
            task_pre_assign[ti]['group'][ki] = 0
    for wi in workers:
        strategy[wi] = 0
    space = {}
    for wi in workers:
        space[wi] = []
        worker = worker_dict[wi]
        for ti in tasks:
            task = task_dict[ti]
            if ability_check(worker, task) and satisfy_check(task, worker,
                                                              task['deadline'], worker['deadline'],
                                                              current_time, worker_v,move_cost,reachable_dis):
                space[wi].append(ti)
    before_strategy = {-1: -1}
    count=0
    _assigned_task=set()

    task_pre_assign,strategy = random_strategy(task_dict, worker_dict, workers, tasks,
                                      current_time, worker_v, move_cost, alpha,max_s, max_p,move_cost,reachable_dis,
                                      task_pre_assign,strategy)

    while check_nash_equilibrium(before_strategy, strategy) is False:
        count+=1
        # print(count)
        before_strategy=copy.deepcopy(strategy)
        for wi in workers:
            # select best task for wi(in the perspective of wi)
            best_task = 0
            best_skill = 0
            best_task_conflict_w = 0
            best_list=[]
            utility_max = -1
            if strategy[wi]!=0:
                utility_max=task_utility(
                     strategy[wi], task_dict, worker_dict, move_cost, wi, alpha, worker_v, max_s, max_p)
            for ti in space[wi]:
                # if task_dict[ti]['budget'] < task_dict[task_dict[ti]['Ct']]['budget']:
                #     break
                # mod_strategy = copy.deepcopy(strategy)
                for w_skill in worker_dict[wi]['Kw']:
                    conflict_w = 0
                    cur_list = []
                    cur_skill = []
                    if any(wi == int(value) for value in task_pre_assign[ti]['group'].values()) is False:
                        if w_skill in task_dict[ti]['Kt']:
                            if len(task_pre_assign[ti]['list']) < len(task_dict[ti]['Kt']):
                                # ad
                                if task_pre_assign[ti]['group'][w_skill] == 0:
                                    cur_list.append(wi)
                                    cur_skill=w_skill
                                    cur_worker = wi
                                    for w_pre in task_pre_assign[ti]['list']:
                                        cur_list.append(w_pre)
                                        desired_value = w_pre
                                        matching_keys = [key for key, value in task_pre_assign[ti]['group'].items() if
                                                         value == desired_value]
                                        # cur_skill.append(matching_keys[0])
                                # replace 考虑有没有必要replace
                                else:
                                    origin_wlist = task_pre_assign[ti]['list']
                                    conflict_w = task_pre_assign[ti]['group'][w_skill]
                                    origin_u = task_utility(
                                         ti, task_dict, worker_dict, move_cost, conflict_w, alpha,worker_v, max_s, max_p)
                                    mod_wlist = []
                                    # conflict_w = task_pre_assign[ti]['group'][w_skill]
                                    for w in origin_wlist:
                                        if w != conflict_w:
                                            mod_wlist.append(w)
                                    mod_wlist.append(wi)
                                    mod_u = task_utility(
                                         ti, task_dict, worker_dict, move_cost, wi, alpha,worker_v, max_s, max_p)

                                    conflict_u = 0
                                    if set(task_dict[ti]['Ct']) in _assigned_task:
                                        conflict_wlist = task_pre_assign[task_dict[ti]['Ct']]['list']
                                        conflict_u = task_utility(
                                            ti, task_dict, worker_dict, move_cost, conflict_wlist, alpha, worker_v,
                                            max_s, max_p)
                                    if mod_u <= conflict_u:
                                        break

                                    if mod_u > origin_u:
                                        cur_skill=w_skill
                                        cur_worker=wi
                                        cur_list=mod_wlist
                                    else:
                                        continue
                            else:
                                # replace
                                if task_pre_assign[ti]['group'][w_skill] != 0:
                                    origin_wlist = task_pre_assign[ti]['list']
                                    conflict_w = task_pre_assign[ti]['group'][w_skill]
                                    origin_u = task_utility(
                                         ti, task_dict, worker_dict, move_cost, conflict_w, alpha,worker_v, max_s, max_p)
                                    mod_wlist = []
                                    for w in origin_wlist:
                                        if w != conflict_w:
                                            mod_wlist.append(w)
                                    mod_wlist.append(wi)
                                    mod_u = task_utility(
                                         ti, task_dict, worker_dict, move_cost, wi, alpha,worker_v, max_s, max_p)
                                    all_skills = list(task_pre_assign[ti]['group'].keys())

                                    conflict_u = 0
                                    if set(task_dict[ti]['Ct']) in _assigned_task:
                                        conflict_wlist = task_pre_assign[task_dict[ti]['Ct']]['list']
                                        conflict_u = task_utility(
                                            ti, task_dict, worker_dict, move_cost, conflict_wlist, alpha, worker_v,
                                            max_s, max_p)
                                    if mod_u <= conflict_u:
                                        break

                                    if mod_u > origin_u:
                                        cur_skill=w_skill
                                        cur_worker=wi
                                        cur_list=mod_wlist
                                    else:
                                        continue
                            cur_utility = task_utility(
                                 ti, task_dict, worker_dict, move_cost, cur_worker, alpha, worker_v, max_s, max_p)

                            conflict_u = 0
                            if set(task_dict[ti]['Ct']) in _assigned_task:
                                conflict_wlist = task_pre_assign[task_dict[ti]['Ct']]['list']
                                conflict_u = task_utility(
                                 ti, task_dict, worker_dict, move_cost, conflict_wlist, alpha, worker_v, max_s, max_p)

                            if cur_utility <= conflict_u:
                                cur_utility = 0

                            if cur_utility > utility_max:
                                utility_max = cur_utility
                                best_task = ti
                                best_worker = cur_worker
                                best_skill=cur_skill
                                best_task_conflict_w = conflict_w
                                conflict_skill=cur_skill
                                # if best_task_conflict_w!=0 and best_task!=strategy[best_task_conflict_w]:
                                #     print("at",wi,"find best_task!=strategy[best_task_conflict_w]",(best_task,strategy[best_task_conflict_w]))
                                best_list=cur_list

            # change assign
            # print(best_task, best_list)
            if best_task != 0:
                if set(task_dict[best_task]['Ct']) in _assigned_task:
                    # 替换冲突任务
                    if task_dict[best_task]['budget'] > task_dict[task_dict[best_task]['Ct']]['budget']:
                        task_pre_assign[task_dict[best_task]['Ct']]['list'] = []
                        # print(best_task, task_pre_assign[best_task]['list'])
                        for k in task_pre_assign[task_dict[best_task]['Ct']]['group']['Kt']:
                            task_pre_assign[task_dict[best_task]['Ct']]['group'][k] = []
                        _assigned_task.remove(task_dict[best_task]['Ct'])
                        before_t=strategy[wi]
                        if before_t!=0:
                            before_d_wi_list=set(task_pre_assign[before_t]['list'])
                            before_d_wi_list.discard(wi)
                            task_pre_assign[before_t]['list']=list(before_d_wi_list)
                            desired_value = wi
                            c_skill = [key for key, value in task_pre_assign[before_t]['group'].items() if
                                             value == desired_value]
                            task_pre_assign[before_t]['group'][c_skill[0]]=0

                        strategy[wi] = best_task
                        if(best_task_conflict_w in best_list):
                            count+=1
                            # print("at",wi,"find best_task_conflict_w in best_list!!")
                        task_pre_assign[best_task]['list'] = best_list
                        task_pre_assign[best_task]['group'][conflict_skill]=best_worker
                        _assigned_task.add(best_task)
                        if best_task_conflict_w != 0:
                            if strategy[best_task_conflict_w]!=0:

                                before_d_wi_list=set(task_pre_assign[strategy[best_task_conflict_w]]['list'])
                                before_d_wi_list.discard(best_task_conflict_w)
                                task_pre_assign[strategy[best_task_conflict_w]]['list']=list(before_d_wi_list)

                                if task_pre_assign[strategy[best_task_conflict_w]]['group'][conflict_skill]==best_task_conflict_w:
                                    task_pre_assign[strategy[best_task_conflict_w]]['group'][conflict_skill]=0
                            strategy[best_task_conflict_w] = 0
                else:
                    before_t = strategy[wi]
                    if before_t != 0:
                        before_d_wi_list = set(task_pre_assign[before_t]['list'])
                        before_d_wi_list.discard(wi)
                        task_pre_assign[before_t]['list'] = list(before_d_wi_list)
                        desired_value = wi
                        c_skill = [key for key, value in task_pre_assign[before_t]['group'].items() if
                                   value == desired_value]
                        task_pre_assign[before_t]['group'][c_skill[0]] = 0

                    strategy[wi] = best_task
                    if (best_task_conflict_w in best_list):
                        count += 1
                        # print("at",wi,"find best_task_conflict_w in best_list!!")
                    _assigned_task.add(best_task)
                    task_pre_assign[best_task]['list'] = best_list
                    task_pre_assign[best_task]['group'][conflict_skill] = best_worker

                    if best_task_conflict_w != 0:
                        if strategy[best_task_conflict_w] != 0:

                            before_d_wi_list = set(task_pre_assign[strategy[best_task_conflict_w]]['list'])
                            before_d_wi_list.discard(best_task_conflict_w)
                            task_pre_assign[strategy[best_task_conflict_w]]['list'] = list(before_d_wi_list)

                            if task_pre_assign[strategy[best_task_conflict_w]]['group'][
                                conflict_skill] == best_task_conflict_w:
                                task_pre_assign[strategy[best_task_conflict_w]]['group'][conflict_skill] = 0
                        strategy[best_task_conflict_w] = 0

        if count > 50:
            break

    return strategy, task_pre_assign

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

def get_available_task(task_dict, arrived_tasks, best_assign,current_time, TC):
    """
    计算当前可参与的任务
    """
    free_task = []
    for i in arrived_tasks:
        related_tasks = get_related_tasks(TC, i)
        is_free = True
        for conflict_id in task_dict[i]['Ct']:
            if best_assign[conflict_id]['assigned'] is True:
                is_free = False
                break
        for related_tasks_conflict in related_tasks:
            for id in task_dict[related_tasks_conflict]['Ct']:
                if best_assign[id]['assigned'] is True:
                    is_free = False
                    break
        if len(related_tasks)>2:
            is_free = False
        if is_free:
            free_task.append(i)
    return free_task

def gt_base_algorithm(task_dict, worker_dict, max_time, move_cost, alpha, max_s, max_p, worker_v,reachable_dis):
    # performance
    full_assign = 0
    worker_assign_time={}
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
        best_assign[i]['assigned'] = False
        for k in task_dict[i]['Kt']:
            best_assign[i]['group'][k] = 0

    arrived_tasks = set()
    arrived_workers = set()
    sucess_to_assign_task = set()
    delete_to_assign_task = set()
    for current_time in range(1, max_time+1):  # each time slice try to allocate
        arrived_tasks = set()
        arrived_workers = set()
        sucess_to_assign_worker = set()
        for t_id in task_dict.keys():  # discard those meet their deadline set(task_dict[t_id]['Ct']).issubset(sucess_to_assign_task)
            if task_dict[t_id]['deadline'] <= current_time or set(
                    task_dict[t_id]['Ct']) & sucess_to_assign_task or tuple(
                    task_dict[t_id]['Dt']) in delete_to_assign_task:
                delete_to_assign_task.add(t_id)
                # arrived_tasks = arrived_tasks.remove(t_id)
                # arrived_workers = arrived_workers.remove(t_id)
        for t_id in task_dict.keys():  # add new
            if task_dict[t_id]['arrived_time'] <= current_time and task_dict[t_id]['deadline'] >= current_time and \
                    t_id not in sucess_to_assign_task and tuple(task_dict[t_id]['Ct']) not in sucess_to_assign_task \
                    and tuple(task_dict[t_id]['Dt']) not in delete_to_assign_task and t_id not in delete_to_assign_task:
                arrived_tasks.add(t_id)
        for w_id in worker_dict.keys():  # add new
            if worker_dict[w_id]['arrived_time'] <= current_time and worker_dict[w_id]['deadline'] >= current_time:
                arrived_workers.add(w_id)
        flag = 1
        while flag!=0:
            workers = list(arrived_workers)
            tasks =  get_free_task(task_dict, arrived_tasks, best_assign)
            # 通过博弈获得本轮工人的策略，基于此策略，更新任务完成情况
            equ_strategy, task_pre_assign = gt_base(
                task_dict, worker_dict, workers, tasks, current_time, worker_v,reachable_dis,move_cost,alpha,max_s,max_p)

            flag = 0
            # 统计'list'不为[]的个数
            for ai in task_pre_assign:
                if len(task_pre_assign[ai]['list']) == len(task_dict[ai]['Kt']) and set(
                        task_pre_assign[ai]['list']).issubset(sucess_to_assign_worker) is False:
                # print(ai, task_pre_assign[ai]['list'])
                    if conflict_check(ai, task_assign_condition, task_dict) is False:
                        satisfied = True
                        task_pre_assign[ai]
                        for task_id in task_dict[ai]['Dt']:
                            if task_assign_condition[task_id] == -1:
                                satisfied = False
                        cur_task_pay=satisfaction1(
                                task_dict, worker_dict, ai, move_cost, task_pre_assign[ai]['list'], alpha,worker_v, max_s, max_p)
                        for task_id in task_dict[ai]['Ct']:
                            if task_id in task_pre_assign:
                                conflict_task_pay = satisfaction1(
                                task_dict, worker_dict, task_id, move_cost, task_pre_assign[task_id]['list'], alpha,worker_v, max_s, max_p)
                                if conflict_task_pay>cur_task_pay:
                                    satisfied = False
                        if satisfied!=False:
                            flag += 1
                            best_assign[ai]['list'] = task_pre_assign[ai]['list']
                            best_assign[ai]['group'] = task_pre_assign[ai]['group']
                            best_assign[ai]['satisfaction'] = satisfaction(
                                task_dict, worker_dict, ai, move_cost, best_assign[ai]['list'], alpha,worker_v, max_s, max_p,current_time)
                            best_assign[ai]['assigned'] = True
                            for w_assigned in best_assign[ai]['list']:
                                sucess_to_assign_worker.add(w_assigned)
                                if w_assigned not in worker_assign_time:
                                    worker_assign_time[w_assigned]=[]
                                worker_assign_time[w_assigned].append({current_time:ai})
                                # if len(best_assign[ai]['list'])!=len(set(best_assign[ai]['list'])):
                                #     worker_assign_time[w_assigned].append('Err')
                            task_assign_condition[ai] = current_time
                            sucess_to_assign_task.add(ai)
                            print(ai,best_assign[ai]['list'],best_assign[ai]['satisfaction'])

            print(worker_dict[3])
            non_empty_list_count = sum(1 for value in task_pre_assign.values() if value['list'])
            print(non_empty_list_count)
            print('预分配:', flag)
            total_sat = 0
            for i in best_assign.keys():
                total_sat += best_assign[i]['satisfaction']
            print(total_sat)
            print(len(sucess_to_assign_task))
            print(current_time)
            arrived_tasks = arrived_tasks.difference(sucess_to_assign_task)
            arrived_workers = arrived_workers.difference(sucess_to_assign_worker)

    # 输出当前分配和总的满意度
    total_sat = 0
    for i in best_assign.keys():
        total_sat += best_assign[i]['satisfaction']
        if best_assign[i]['assigned'] == True:
            full_assign += 1
    return task_assign_condition, best_assign, full_assign, total_sat,worker_assign_time
