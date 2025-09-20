def can_form_line(edges):
    """
    Determine if the given list of edges can form a line
    edges: List[List[int]], each element is [a,b] representing a directed edge from a to b
    return: bool, whether the edges can form a line
    """
    if not edges:
        return True
    if len(edges) == 1:
        return True
        
    # Build adjacency list representation of the graph
    graph = {}
    in_degree = {}
    for a, b in edges:
        if a not in graph:
            graph[a] = []
        if b not in in_degree:
            in_degree[b] = 0
        graph[a].append(b)
        in_degree[b] += 1
    
    # Find start node (node with in-degree 0)
    start = None
    for a, _ in edges:
        if a not in in_degree:
            start = a
            break
    if start is None:
        return False # Contains cycle
        
    # Traverse from start node
    curr = start
    path = [curr]
    while curr in graph:
        if len(graph[curr]) > 1:
            return False # Node has multiple successors
        curr = graph[curr][0]
        if curr in path:
            return False # Contains cycle
        path.append(curr)
        
    # Check if all nodes are visited
    return len(path) == len(set([x for edge in edges for x in edge]))

# Test cases
def test_can_form_line():
    # Test case 1: Can form a line
    assert can_form_line([[1,2], [2,3], [3,4]]) == True
    
    # Test case 2: Empty list
    assert can_form_line([]) == True
    
    # Test case 3: Single edge
    assert can_form_line([[1,2]]) == True
    
    # Test case 4: Contains cycle
    assert can_form_line([[1,2], [2,3], [3,1]]) == False
    
    # Test case 5: Node has multiple successors
    assert can_form_line([[1,2], [1,3], [2,4]]) == False
    
    # Test case 6: Not connected
    assert can_form_line([[1,2], [3,4]]) == False

    # Test case 7: 
    assert can_form_line([[1,2], [2,3], [3,4], [4,1]]) == False
    
    print("All test cases passed!")

if __name__ == "__main__":
    test_can_form_line()
