def can_form_line(edges):
    """
    判断给定的边列表是否能构成一条直线
    edges: List[List[int]], 每个元素形如[a,b]表示a指向b
    return: bool, 是否能构成一条直线
    """
    if not edges:
        return True
    if len(edges) == 1:
        return True
        
    # 构建图的邻接表
    graph = {}
    in_degree = {}
    for a, b in edges:
        if a not in graph:
            graph[a] = []
        if b not in in_degree:
            in_degree[b] = 0
        graph[a].append(b)
        in_degree[b] += 1
    
    # 找到起点(入度为0的点)
    start = None
    for a, _ in edges:
        if a not in in_degree:
            start = a
            break
    if start is None:
        return False # 存在环
        
    # 从起点开始遍历
    curr = start
    path = [curr]
    while curr in graph:
        if len(graph[curr]) > 1:
            return False # 一个点有多个后继
        curr = graph[curr][0]
        if curr in path:
            return False # 存在环
        path.append(curr)
        
    # 检查是否访问了所有点
    return len(path) == len(set([x for edge in edges for x in edge]))

# 测试用例
def test_can_form_line():
    # 测试用例1: 可以构成直线
    assert can_form_line([[1,2], [2,3], [3,4]]) == True
    
    # 测试用例2: 空列表
    assert can_form_line([]) == True
    
    # 测试用例3: 单个边
    assert can_form_line([[1,2]]) == True
    
    # 测试用例4: 存在环
    assert can_form_line([[1,2], [2,3], [3,1]]) == False
    
    # 测试用例5: 一个点有多个后继
    assert can_form_line([[1,2], [1,3], [2,4]]) == False
    
    # 测试用例6: 不连通
    assert can_form_line([[1,2], [3,4]]) == False
    
    print("All test cases passed!")

if __name__ == "__main__":
    test_can_form_line()
