import copy
from random import choice
from itertools import combinations, product
from utils.score_tool import dependency_check, satisfy_check, distance_check, conflict_check, Euclidean_fun, \
	satisfaction

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


def max_weighted_matching(graph):
	# 初始化结果列表
	result = []

	while True:
		# 遍历图中所有节点
		for node in graph.keys():
			if not isinstance(node, str) and len(result) < len(graph[node]):
				continue

			# 获取当前节点可能的匹配对象
			matches = [edge[1] for edge in graph[node]]

			# 如果没有可选的匹配对象，则跳过该节点
			if not matches:
				continue

			# 根据权重进行排序
			sorted_matches = sorted(matches, key=lambda x: graph[(x, node)]['weight'], reverse=True)

			# 从第一个匹配对象开始尝试寻找路径
			path = [(node, sorted_matches[0])]
			current_node = sorted_matches[0]

			# 在路径上查找其他匹配对象并添加到路径中
			while (current_node, node) in graph:
				next_nodes = [edge[1] for edge in graph[(current_node, node)]]
				if not next_nodes:
					break
				else:
					path.append((next_nodes[0], node))
					current_node = next_nodes[0]

			# 将路径添加到结果列表中
			result.extend(path)

			# 更新已经匹配的边
			for i in range(len(path) - 2, -1, -1):
				u, v = path[i][0], path[i + 1][1]
				del graph[(u, v)]

			# 清空已经匹配的边后判断是否还存在未匹配的边
			if any([isinstance(key, tuple) for key in graph.keys()]):
				return result

			# 若不再存在未匹配的边，返回结果列表
			return result


def hungarian_algorithm(graph):
	matching = nx.bipartite.maximum_matching(graph)
	return matching


def get_order_in_related_tasks(dependency_graph, task):
	visited = set()
	dependent_tasks = []

	def dfs(current_task):
		visited.add(current_task)
		dependent_tasks.append(current_task)  # 将当前任务添加到结果列表
		for dependent_task in dependency_graph.get(current_task, []):
			if dependent_task not in visited:
				dfs(dependent_task)

	dfs(task)
	dependent_tasks.reverse()  # 反转结果列表
	return dependent_tasks


def get_order_related_tasks(dependency_graph, task):
	visited = set()
	dependent_tasks = []

	def dfs(current_task):
		visited.add(current_task)
		for dependent_task in dependency_graph.get(current_task, []):
			if dependent_task not in visited:
				dfs(dependent_task)
				dependent_tasks.append(dependent_task)

	dfs(task)
	dependent_tasks.append(task)
	return dependent_tasks

def get_order_in_related_tasks(dependency_graph, task):
    visited = set()
    dependent_tasks = []

    def dfs(current_task):
        visited.add(current_task)
        dependent_tasks.append(current_task)  # 将当前任务添加到结果列表
        for dependent_task in dependency_graph.get(current_task, []):
            if dependent_task not in visited:
                dfs(dependent_task)

    dfs(task)
    dependent_tasks.reverse()  # 反转结果列表
    return dependent_tasks

def init_bipartite(tasks, workers):
	G = nx.Graph()

	# 添加任务节点
	G.add_nodes_from(tasks, bipartite=0)

	# 添加工人节点
	G.add_nodes_from(workers, bipartite=1)

	return G


def price_score(task_dict, worker_dict, t, v, worker_list):
	if len(worker_list) > 0:
		dis = 0
		for i in worker_list:
			dis += Euclidean_fun(task_dict[t]['Lt'], worker_dict[i]['Lw'])
		return task_dict[t]['budget'] - dis * v
	return 0


def satisfaction1(task_dict, worker_dict, ti, v, w_list, alpha, worker_v, max_s, max_p):
	if len(w_list) == 0 or max_p == 0:
		return 0
	total_score = 0
	for w in w_list:
		total_score += worker_dict[w]['score']
	profit_w = price_score(task_dict, worker_dict, ti, v, w_list)

	return alpha * (total_score / len(w_list)) / max_s + (1 - alpha) * (profit_w / max_p)


def price_score_be(task_dict, worker_dict, t, v, w):
	dis = 0
	dis = Euclidean_fun(task_dict[t]['Lt'], worker_dict[w]['Lw'])
	return task_dict[t]['budget'] - dis * v


def satisfaction_be(task_dict, worker_dict, ti, v, w, alpha, worker_v, max_s, max_p):
	total_score = worker_dict[w]['score']
	profit_w = price_score_be(task_dict, worker_dict, ti, v, w)

	return alpha * (total_score) / max_s + (1 - alpha) * (profit_w / max_p)


def init_best_assign(task_dict):
	best_assign = {}
	for i in task_dict.keys():
		best_assign[i] = {}
		best_assign[i]['list'] = []
		best_assign[i]['group'] = {}
		best_assign[i]['satisfaction'] = 0
		best_assign[i]['assigned'] = False
		for k in task_dict[i]['Kt']:
			best_assign[i]['group'][k] = 0
	return best_assign


def update_task_worker(task_dict, worker_dict, arrived_tasks, arrived_workers, current_time, sucess_to_assign_task,
					   delete_to_assign_task):
	# 更新每轮的可用任务和工人
	# after_discard_tasks = copy.deepcopy(arrived_tasks)
	for t_id in task_dict.keys():  # discard those meet their deadline set(task_dict[t_id]['Ct']).issubset(sucess_to_assign_task)
		if task_dict[t_id]['deadline'] <= current_time or set(task_dict[t_id]['Ct']) & sucess_to_assign_task or tuple(
				task_dict[t_id]['Dt']) in delete_to_assign_task:
			delete_to_assign_task.add(t_id)
		# arrived_tasks = after_discard_tasks
		# after_discard_workers = copy.deepcopy(arrived_workers)
	# for t_id in delete_to_assign_task:  # discard those meet their deadline
	# 	arrived_tasks.discard(t_id)
		# arrived_workers = after_discard_workers
	for t_id in task_dict.keys():  # add new
		if task_dict[t_id]['arrived_time'] <= current_time and task_dict[t_id]['deadline'] >= current_time and \
				t_id not in sucess_to_assign_task and tuple(task_dict[t_id]['Ct']) not in sucess_to_assign_task \
				and tuple(task_dict[t_id]['Dt']) not in delete_to_assign_task:
			arrived_tasks.add(t_id)
	for w_id in worker_dict.keys():  # add new
		if worker_dict[w_id]['arrived_time'] <= current_time and worker_dict[w_id]['deadline'] >= current_time:
			arrived_workers.add(w_id)
	return arrived_tasks, arrived_workers


def build_TC(task_dict, arrived_tasks):
	tl = sorted(list(arrived_tasks))
	TC = {}
	marked = {}
	for item in tl:
		marked[item] = False
	for _, t in enumerate(tl[::-1]):
		if marked[t] is True:
			continue
		task = task_dict[t]
		tc = []
		is_valid = True
		for depend_t in task['Dt']:
			if depend_t in arrived_tasks:
				tc.append(depend_t)
		if is_valid:
			tc.append(t)
			TC[t] = sorted(tc)
		marked[t] = True
	return TC

def get_order_in_related_tasks(dependency_graph, task):
    visited = set()
    dependent_tasks = []

    def dfs(current_task):
        visited.add(current_task)
        dependent_tasks.append(current_task)  # 将当前任务添加到结果列表
        for dependent_task in dependency_graph.get(current_task, []):
            if dependent_task not in visited:
                dfs(dependent_task)

    dfs(task)
    dependent_tasks.reverse()  # 反转结果列表
    return dependent_tasks

def get_available_task(task_dict, arrived_tasks, best_assign,current_time, TC, delete_to_assign_task):
    """
    计算当前可参与的任务
    """
    free_task = []
    for i in arrived_tasks:
        related_tasks = get_order_in_related_tasks(TC, i)
        is_free = True
        # for depend_id in task_dict[i]['Dt']:
        #     if best_assign[depend_id]['assigned'] is False:
        #         is_free = False
        #         break
        for related_tasks_conflict in related_tasks:
            for id in task_dict[related_tasks_conflict]['Ct']:
                if best_assign[id]['assigned'] is True:
                    is_free = False
                    break
        # for related_tasks_id in related_tasks:
        #     for id in task_dict[related_tasks_conflict]['Ct']:
        # if related_tasks.intersection(delete_to_assign_task) is True:
        #     is_free = False
        # if len(related_tasks)>10:
        #     is_free = False
        if is_free:
            free_task.append(i)
    return free_task

def count(C):
	"""
    统计完成个数
    """
	return len(C)


def group_greedy_algorithm(task_dict, worker_dict, max_time, move_cost, alpha, max_p, worker_v, reachable_dis, max_s):
	# performance
	full_assign = 0
	task_assign_condition = {}
	# final task_workers assignment
	best_assign = init_best_assign(task_dict)
	# 每个任务需要的技能，技能对应的候选人
	# skill_group = {}
	# for i in task_dict.keys():
	# 	skill_group[i] = {}
	# 	d = [[] for i in range(len(task_dict[i]['Kt']))]
	# 	for k in range(0, len(task_dict[i]['Kt'])):
	# 		skill_group[i][task_dict[i]['Kt'][k]] = d[k]
	task_assign_condition = {}
	for i in task_dict.keys():
		task_assign_condition[i] = -1
	# 之前未分配任务-过期任务+本时间点新增的任务
	# assigned_workers = set()
	arrived_tasks = set()
	arrived_workers = set()
	sucess_to_assign_task = set()
	delete_to_assign_task = set()
	for current_time in range(1, max_time+1):  # each time slice try to allocate
		arrived_tasks = set()
		arrived_workers = set()
		sucess_to_assign_worker = set()
		# delete_to_assign_task = set()
		arrived_tasks, arrived_workers = update_task_worker(task_dict, worker_dict, arrived_tasks, arrived_workers,
															current_time, sucess_to_assign_task, delete_to_assign_task)


		TC = build_TC(task_dict, arrived_tasks)
		TC = {key: TC[key] for key in sorted(TC.keys())}

		tasks = get_available_task(task_dict, arrived_tasks, best_assign, current_time, TC, delete_to_assign_task)

		TC_Group= {}
		for i in tasks:
			TC_Group[i] = sorted(get_order_in_related_tasks(TC, i))

		# 候选分配
		for _, tc in TC_Group.items():
			count_each_group_sa = {}
			best_satisfaction = 0
			success_assign_flag = False
			for _, tc in TC_Group.items():
				# print(tc)
				edges = []
				cur_worker_list = []
				for cur_task_id in tc:
					if cur_task_id in sucess_to_assign_task:
						count_each_group_sa[tc[-1]] = 0
						continue
					if conflict_check(cur_task_id, task_assign_condition, task_dict):
						delete_to_assign_task.update(tc)
						TC_Group = {key: value for key, value in TC_Group.items() if
							  not any(num in value for num in tc)}
						break
					task = task_dict[cur_task_id]
					for k in range(0, len(task['Kt'])):
						for w_id in arrived_workers:
							if w_id not in sucess_to_assign_worker:
								worker = worker_dict[w_id]
								# distance check
								if satisfy_check(task, worker,
												 task['deadline'], worker['deadline'],
												 current_time, worker_v, move_cost, reachable_dis):
									for s in range(0, len(worker_dict[w_id]['Kw'])):
										if worker_dict[w_id]['Kw'][s] == task['Kt'][k]:
											# skill_group[cur_task_id][task['Kt'][k]].append(w_id)
											cur_satisfaction = satisfaction_be(task_dict, worker_dict, cur_task_id,
																			   move_cost, w_id,
																			   alpha,
																			   worker_v, max_s, max_p)
											if cur_satisfaction < 0:  # 保证每条边权重大于0
												cur_satisfaction = 0
											edges.append(
												(k * 10000 + cur_task_id, 100000 + w_id, -cur_satisfaction))

				# 提取节点
				nodes = sorted(list(set(i for i, _, _ in edges) | set(j for _, j, _ in edges)))

				# 使用字典记录节点索引的映射关系
				node_index = {node: index for index, node in enumerate(nodes)}

				# 构建一个节点数 * 节点数的矩阵，初始化为0
				num_nodes = len(nodes)
				cost_matrix = np.zeros((num_nodes, num_nodes))

				# 将权重填充到矩阵对应的位置
				for left, right, weight in edges:
					cost_matrix[node_index[left], node_index[right]] = weight

				# 使用 linear_sum_assignment 求解最大权重匹配
				row_ind, col_ind = linear_sum_assignment(cost_matrix)

				# 构建匹配结果字典
				matching_result = {nodes[row]: nodes[col] for row, col in zip(row_ind, col_ind)}
				filtered_result = {key: value for key, value in matching_result.items() if key < 100000 and value > 100000}
				# filtered_result = {key: value for key, value in filtered_result.items() if value > 100000}
				matching_edges = [(node, filtered_result[node]) for node in filtered_result if node in nodes]

				# 构建一个字典，以任务和工人的元组为键，权重为值
				edges_dict = {(t, w): value for t, w, value in edges}

				# 构建 target_value 字典
				target_value = {task: -edges_dict.get((task, worker), 0) for task, worker in matching_edges}


				# target_value = {}
				# for task, worker in matching_edges:
				# 	aa = next((value for t, w, value in edges if t == task and w == worker), None)
				# 	target_value[task] = -aa

				task_id = [key for key in matching_result.keys() if key < 10000]

				cur_satisfaction = 0
				flag = True
				for id in sorted(task_id):
					count = 0
					sum_sa = 0
					for key in target_value.keys():
						if key % 10000 == id:
							count += 1
							sum_sa += target_value[key]
							cur_worker_list.append(matching_result[key] % 100000)
					if count != len(task_dict[id]['Kt']):
						flag = False
						break
					else:
						cur_satisfaction += sum_sa / count

				if flag:
					cur_satisfaction = cur_satisfaction
					cur_worker_list = cur_worker_list
				else:
					cur_satisfaction = 0

				if cur_satisfaction > best_satisfaction:
					best_satisfaction = cur_satisfaction
					best_task_group = cur_task_id
					best_task_worker = cur_worker_list
					success_assign_flag = True

			if success_assign_flag:
				related_tasks = get_order_in_related_tasks(TC, best_task_group)
				print(related_tasks, best_task_worker, best_satisfaction)
				flag=True
				# NEED UPDATE
				for t in related_tasks:
					if conflict_check(t, task_assign_condition, task_dict):
						# delete_to_assign_task.update(t)
						delete_to_assign_task.discard(t)
						# TC_Group.pop(t, None)
						# TC_Group = {key: value for key, value in TC_Group.items() if
						# 	  not any(num in value for num in related_tasks)}
						print("冲突")
						flag=False
						break
					if flag:
						for t in related_tasks:
							sucess_to_assign_task.add(t)
							task_assign_condition[t] = current_time
							len_skill = len(task_dict[t]['Kt'])
							w_list = best_task_worker[:len_skill]
							del best_task_worker[:len_skill]
							for w in w_list:
								sucess_to_assign_worker.add(w)
								trave_dis = Euclidean_fun(task_dict[t]['Lt'], worker_dict[w]['Lw'])
								worker_dict[w]['arrived_time'] = max(worker_dict[w]['arrived_time'],
																	 task_dict[t]['arrived_time']) + trave_dis / worker_v
								worker_dict[w]['Lw'] = task_dict[t]['Lt']
							best_assign[best_task_group]['satisfaction'] = best_satisfaction
							# 更新分配情况
							best_assign[best_task_group]['assigned'] = True
							# print(related_tasks, best_task_worker, best_assign[best_task_group]['satisfaction'] )
							TC_Group = {key: value for key, value in TC_Group.items() if
								  not any(num in value for num in related_tasks)}
							arrived_tasks = arrived_tasks.difference(set(sucess_to_assign_task))
							arrived_workers = arrived_workers.difference(
								set(sucess_to_assign_worker))
			else:
				break
		total_sat = 0
		for i in best_assign.keys():
			total_sat += best_assign[i]['satisfaction']
		print('----------------------', current_time, total_sat)
		# # NEED UPDATE
		# arrived_tasks = arrived_tasks.difference(set(sucess_to_assign_task))
		# arrived_workers = arrived_workers.difference(
		# 	set(sucess_to_assign_worker))
	total_sat = 0
	for i in best_assign.keys():
		total_sat += best_assign[i]['satisfaction']
	return task_assign_condition, best_assign, full_assign, total_sat



