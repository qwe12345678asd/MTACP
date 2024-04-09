import os
import json
import math
import copy
import random
from random import choice
from itertools import combinations, product
import numpy as np


def Euclidean_fun(A, B):
    return math.sqrt(sum([(a - b)**2 for (a, b) in zip(A, B)]))


def gen_task_distribution(average_tasks_per_time_unit, max_time_span):
    average_tasks_per_time_unit = average_tasks_per_time_unit # 每分钟平均任务数量
    max_time_span = max_time_span  # 生成时间跨度
    # 生成符合泊松分布的随机数
    return np.random.poisson(lam=average_tasks_per_time_unit, size=max_time_span)


def gen_worker_distribution(average_workers_per_time_unit, max_time_span):
    average_workers_per_time_unit = average_workers_per_time_unit  # 每分钟平均任务数量
    max_time_span = max_time_span  # 生成时间跨度
    # 生成符合泊松分布的随机数
    return np.random.poisson(lam=average_workers_per_time_unit, size=max_time_span)


class data_process(object):

    def task_data_fun(self, T, min_skill, max_skill, budget_range, min_pre_task_num, max_pre_task_num,min_task_deadline_span, max_task_deadline_span, DP,CP):
        """构建task_dict
        随机生成任务的对应技能要求Kt、location、budget、
        T 生成任务个数
        min_pre_task_num 最小的依赖任务数量
        max_pre_task_num 最大的依赖任务数量
        max_conflict_num 最大的冲突任务数量
        """
        # task的技能要求数随机生成
        K = []
        while T != len(K):
            x = random.randint(min_skill, max_skill)  # task所要求的技能数
            K.append(x)
        # print(K, len(K))   # 每个task的技能
        Lt = []
        task_dict = {}
        i = 0  # 当前task序号
        cnt = 0
        max_time_span = 6 # 生成时间跨度
        average_tasks_per_time_unit =int(T/max_time_span)+10   # 每分钟平均请求数量
        # 任务到达时间的分布
        tasks = gen_task_distribution(
            average_tasks_per_time_unit, max_time_span)
        while cnt != T:
            # 输出每个时间点的请求数量
            break_sign = False
            if break_sign:
                break
            for tick, task_count in enumerate(tasks):
                print(f"Tick {tick + 1}: {task_count} tasks")
                for _ in range(task_count):
                    if cnt == T:
                        break_sign = True
                        break
                    l1 = random.uniform(0, 1)
                    l2 = random.uniform(0, 1)
                    LL = [l1, l2]
                    # 成本
                    Cb = random.randint(budget_range[0], budget_range[1])
                    # 技能
                    while len(Lt) != K[i]:
                        x = random.randint(1, 3)
                        if x in Lt:
                            continue
                        else:
                            Lt.append(x)
                            Lt.sort()
                    # Dependency 存放前置任务的编号
                    dependency = []
                    den_pro = random.randint(1, DP * 10) / 10
                    if random.random()<den_pro:
                        # den_pro=random.randint(1, DP*10)/10
                        Cb = random.randint(50, 100)
                        dependency_count = min(random.randint(
                            min_pre_task_num, max_pre_task_num), i)
                        while dependency_count >0:
                            dependency_count=dependency_count-1
                            x = random.randint(1+i//2, i)
                            if x in dependency:
                                continue
                            else:
                                dependency.append(x)
                        dependency.sort()
                    # conflict
                    conflict = []
                    # arrived time
                    arrived_time = tick+1
                    # set deadline
                    deadline = arrived_time + \
                        random.randint(min_task_deadline_span,
                                       max_task_deadline_span)
                    # add item to task_dict
                    task_dict[i + 1] = {"Lt": LL, "Kt": Lt,
                                        "budget": Cb,
                                        "Dt": dependency,
                                        "Ct": conflict,
                                        "arrived_time": arrived_time,
                                        "deadline": deadline
                                        }
                    Lt = []
                    i += 1
                    cnt += 1
                if break_sign:
                    break
        # add conflict(两两冲突的生成)
        conflict_dict = {}
        for t_id in task_dict:
            con_pro = random.randint(1, CP * 10) / 10
            if random.random() < con_pro:
                candidate_id = random.randint(1, T)
                if candidate_id == t_id:
                    continue
                if candidate_id in conflict_dict:
                    continue
                if candidate_id in task_dict[t_id]["Dt"]:
                    continue
                if t_id in task_dict[candidate_id]["Dt"]:
                    continue
                task_dict[t_id]["Ct"].append(candidate_id)
                task_dict[candidate_id]["Ct"].append(t_id)
                conflict_dict[t_id] = candidate_id
                conflict_dict[candidate_id] = t_id
        period=min_task_deadline_span

        save_path = 'data/syn/worker'  # 替换为你想要保存的路径
        # os.makedirs(save_path, exist_ok=True)
        filename = f'task{T}_{max_skill}_{period}_{DP}_{max_pre_task_num}_{CP}_U.json'
        # 拼接完整文件路径
        file_path = os.path.join(save_path, filename)

        with open(file_path, 'w', encoding='utf-8') as fp_task:
            json.dump(task_dict, fp_task)
        print('generated task dict:', len(task_dict.keys()))

        return task_dict

    def worker_data_fun(self, W, skill_quantity, skill_range, min_worker_deadline_span, max_worker_deadline_span):
        # set default skill_quantity
        K = []
        while W != len(K):
            x = random.randint(1, skill_quantity)  # task所要求的技能数
            K.append(x)

        worker_dict = {}
        i = 0  # 当前worker序号
        cnt = 0
        max_time_span = 6  # 生成时间跨度
        average_workers_per_time_unit = int(W/max_time_span)+5 # 每分钟平均请求数量
        # 工人到达时间的分布
        workers = gen_worker_distribution(
            average_workers_per_time_unit, max_time_span)
        while cnt != W:
            # 输出每个时间点的请求数量


            break_sign = False
            if break_sign:
                break
            for tick, worker_count in enumerate(workers):
                print(f"Tick {tick + 1}: {worker_count} workers")
                for _ in range(worker_count):
                    if cnt == W:
                        break_sign = True
                        break
                    worker_dict[i+1] = {}
                    # location
                    worker_dict[i+1]['Lw'] = [
                        random.uniform(0, 1), random.uniform(0, 1)]
                    # build worker skill set
                    Kt = []
                    while len(Kt) != K[i]:
                        x = random.randint(1, 3)
                        if x in Kt:
                            continue
                        else:
                            Kt.append(x)
                            Kt.sort()
                    worker_dict[i+1]['Kw'] = Kt
                    worker_dict[i+1]['arrived_time'] = tick+1
                    worker_dict[i+1]['deadline'] = worker_dict[i+1]['arrived_time'] + \
                        random.randint(min_worker_deadline_span,
                                       max_worker_deadline_span)
                    worker_dict[i+1]['score'] = random.uniform(1, 100)
                    i += 1
                    cnt += 1
        period=min_worker_deadline_span

        save_path = 'data/syn/worker'  # 替换为你想要保存的路径
        # os.makedirs(save_path, exist_ok=True)
        filename = f'worker{W}_{skill_quantity}_{period}_U.json'
        file_path = os.path.join(save_path, filename)
        with open(file_path, 'w', encoding='utf-8') as fp_worker:
            json.dump(worker_dict, fp_worker)
        print('generated  worker dict', len(worker_dict.keys()))
        return worker_dict


# test code
dp = data_process()
aa=dp.task_data_fun(T=2500, min_skill=1, max_skill=3, budget_range=[
                 1, 100], min_pre_task_num=1, max_pre_task_num=3,
                 min_task_deadline_span=6, max_task_deadline_span=6, DP=0.5,CP=0.5)
bb=dp.worker_data_fun(500, 1, 3, 6, 6)