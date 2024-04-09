# import numpy as np

# average_requests_per_minute = 10  # 每分钟平均请求数量

# # 模拟一小时的请求情况
# time_span = 60  # 时间跨度为60分钟

# # 生成符合泊松分布的随机数
# requests = np.random.poisson(lam=average_requests_per_minute, size=time_span)

# # 输出每个时间点的请求数量
# for minute, request_count in enumerate(requests):
#     print(f"Minute {minute + 1}: {request_count} requests")

# # 计算总请求数量
# total_requests = np.sum(requests)
# print(f"Total requests: {total_requests}")
from itertools import combinations, product
def all_combin_worker(workerset):
    """分组组合，每组最多取出一个，可以不取"""
    combin_list = []
    for k in range(len(workerset), 0, -1):
        for linelist in list(combinations(workerset, k)):
            linelist = list(linelist)
            for i in product(*linelist):
                i = list(i)
                combin_list.append(i)
    return combin_list


print(all_combin_worker([[1,2,3],[4,5,6]]))

li=[1]
for i in li:
    if i+1<10:
        li.append(i+1)
    
print(li)