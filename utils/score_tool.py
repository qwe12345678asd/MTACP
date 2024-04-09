import json, math, time, itertools, copy, random
from utils.distance_tool import Euclidean_fun
from itertools import combinations, product




def dependency_check(i, task_assign_condition, task_dict):
    """检测前置依赖任务是否完成"""
    satisfied = True
    for task_id in task_dict[i]['Dt']:
        if task_assign_condition[task_id] == -1:
            satisfied = False
            break
    return satisfied

def conflict_check(i, task_assign_condition, task_dict):
    has_conflict = False
    for task_id in task_dict[i]['Ct']:
        if task_assign_condition[task_id] > 0:
            has_conflict = True
            break
    return has_conflict
def distance_check(task_location, worker_location, task_deadline, worker_deadline, current_time, worker_v):
    """检测任务和工人的距离是否符合分配条件"""
    distance = Euclidean_fun(task_location, worker_location)
    return distance < (min(worker_deadline, task_deadline)-current_time+1)*worker_v

def satisfy_check(task, worker, task_deadline, worker_deadline, current_time, worker_v,move_cost,reachable_dis):
    worker_location=worker['Lw']
    task_location=task['Lt']
    trave_dis = Euclidean_fun(task_location, worker_location)
    trave_time = trave_dis/worker_v
    trave_cost = trave_dis * move_cost
    if trave_dis > reachable_dis or worker['arrived_time']+trave_time > worker['deadline'] or worker['arrived_time']+trave_time > task['deadline']\
            or task['arrived_time']+trave_time > task['deadline'] or task['budget']-trave_cost<=0:
        return False
    else:
        return True

def satisfy_check_game(related_tasks,task_dict, worker,current_time, worker_v,move_cost,reachable_dis,TC):
    worker_location = worker['Lw']
    flag=1
    for i in related_tasks:
        task_location=task_dict[i]['Lt']
        trave_dis = Euclidean_fun(task_location, worker_location)
        trave_time = trave_dis/worker_v
        trave_cost = trave_dis * move_cost
        if trave_dis > reachable_dis or worker['arrived_time']+trave_time > worker['deadline'] or worker['arrived_time']+trave_time > task_dict[i]['deadline']\
                or task_dict[i]['arrived_time']+trave_time > task_dict[i]['deadline'] or task_dict[i]['budget']-trave_cost<=0:
            flag=0
    if flag==1:
        return True
    else:
        return False


def satisfaction(task_dict, worker_dict, t, move_cost, cooperation_workers, alpha, worker_v, max_s, max_p,current_time):
    """satisfaction值=alpha * (合作总的工人声誉分 /合作工人数目) + (1 - alpha) * (profit_w / max_p)"""

    if len(cooperation_workers) == 0 or max_p == 0:
        return 0
    total_score = 0
    profit_w = price_score(task_dict, worker_dict, t, move_cost, cooperation_workers)
    for w in cooperation_workers:
        total_score += worker_dict[w]['score']
        trave_dis = Euclidean_fun(task_dict[t]['Lt'], worker_dict[w]['Lw'])
        worker_dict[w]['arrived_time'] = max(worker_dict[w]['arrived_time'], task_dict[t]['arrived_time'],current_time)+ trave_dis / worker_v
        worker_dict[w]['Lw'] = task_dict[t]['Lt']

    return alpha*(total_score/len(cooperation_workers))/max_s+(1-alpha)*(profit_w/max_p)

def price_score(task_dict, worker_dict, t, move_cost, worker_list):
    if len(worker_list) > 0:
        dis = 0
        for i in worker_list:
            dis += Euclidean_fun(task_dict[t]['Lt'], worker_dict[i]['Lw'])
        return task_dict[t]['budget'] - dis * move_cost
    return 0

def skill_check(task_dict, worker_dict,task_id,worker_id):
    """检查工人是否具有任务需要的技能"""
    return len(list(set(worker_dict[worker_id]['Kw']).intersection(set(task_dict[task_id]['Kt'])))) != 0

def all_combin_worker(workerset):
    """返回多个工人组的分组组合，每组最多取出一个，可以不取"""
    combin_list = []
    for k in range(len(workerset), 0, -1):
        for linelist in list(combinations(workerset, k)):
            linelist = list(linelist)
            for i in product(*linelist):
                i = list(i)
                combin_list.append(i)
    return combin_list