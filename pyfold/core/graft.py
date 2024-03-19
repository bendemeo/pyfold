from collections import deque
import numpy as np
import cvxpy as cp
from .geom import absolute_angle, find_incident_lines

# build adjacency graph 
def build_adjacency_dict(vertices, lines):
    adj_dict = {vertex: [] for vertex in vertices}
    
    for line in lines:
        # Assuming each line has 'start' and 'end' attributes that are vertex instances
        if line.end != line.start:  # Avoid adding a vertex as its own neighbor
            adj_dict[line.start].append(line.end)
            adj_dict[line.end].append(line.start)
    
    return adj_dict

def is_graph_connected(vertices, lines):
    adjacency_dict = build_adjacency_dict(vertices, lines)

    # If the graph has no vertices, consider it as connected
    if not adjacency_dict:
        return True
    
    # Initialize all vertices as not visited
    visited = {vertex: False for vertex in adjacency_dict}
    
    # Start DFS from the first vertex in the dictionary
    start_vertex = next(iter(adjacency_dict))
    dfs_visit(start_vertex, visited, adjacency_dict)
    
    # Check if all vertices were visited
    return all(visited.values())

def dfs_visit(vertex, visited, adjacency_dict):
    # Mark the current vertex as visited
    visited[vertex] = True
    
    # Recur for all adjacent vertices that have not been visited
    for adjacent in adjacency_dict[vertex]:
        if not visited[adjacent]:
            dfs_visit(adjacent, visited, adjacency_dict)


# def find_incident_lines(vertex, lines):
#     """
#     Finds all lines that are incident to a given vertex.

#     Parameters:
#     - vertex: The Vertex instance to check.
#     - lines: A list of Line instances.

#     Returns:
#     - A list of Line instances that are incident to the given vertex.
#     """
#     incident_lines = [line for line in lines if line.start == vertex or line.end == vertex]
#     return incident_lines


# def solve_graft(vertices, lines, start_vertex):
#     # Initialize the BFS queue and visited set
#     queue = deque([start_vertex])
#     visited = {start_vertex}
    
#     # Calculate the degree of each vertex
#     degree = {vertex: 0 for vertex in vertices}
#     for line in lines:
#         degree[line.start] += 1
#         degree[line.end] += 1
    
#     while queue:
#         current_vertex = queue.popleft()  # Dequeue a vertex
#         incident_lines = find_incident_lines(current_vertex, lines)  # Find incident lines
        
#         # Apply solve_vertex only if the vertex's degree is not 1 (not a leaf)
#         if degree[current_vertex] != 1:
#             solve_vertex(current_vertex, incident_lines)
        
#         # Enqueue adjacent vertices and mark them as visited
#         for line in incident_lines:
#             adjacent_vertex = line.end if line.start == current_vertex else line.start
#             if adjacent_vertex not in visited:
#                 visited.add(adjacent_vertex)  # Mark as visited
#                 queue.append(adjacent_vertex)  # Enqueue for BFS


def solve_graft(pattern, min_distance=1.):

    constraint_matrices = []

    for i,v in enumerate(pattern.vertices):
        if v.boundary: # don't try to solve boundary vertices
            continue

        #find incident lines
        idxs, lines = list(zip(*find_incident_lines(v, pattern.lines, return_index=True)))
        # print(idxs)
        # print(lines)

        # find neighbors of the vertex
        neighbors = []
        for line in lines:
            if line.start == v:
                nbr = line.end
            else:
                nbr = line.start
            neighbors.append(nbr)

        angles = np.array([absolute_angle(v, n) for n in neighbors])
        angle_matrix = np.vstack([np.cos(angles + np.pi/2.), np.sin(angles + np.pi/2.)])
        
        new_constraint = np.zeros( (2, len(pattern.lines)))
        new_constraint[:,idxs] = angle_matrix
        constraint_matrices.append(new_constraint)
    
    constraint_matrix = np.vstack(constraint_matrices)
    print(constraint_matrix.shape)
    print(constraint_matrix)
    
    #now solve it subject to nonzero distances, trying to keep it uniform.
    print('solving...')
    x = cp.Variable(len(pattern.lines))
    objective = cp.Minimize(sum([cp.abs(x[i]-min_distance)**2 for i in range(len(pattern.lines))]))
    constraints = [constraint_matrix @ x == np.zeros(constraint_matrix.shape[0]), x >= min_distance]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if x.value is not None:
        return(x.value)
    else:
        raise ValueError('No solution found! Check that graft path is a spiderweb. If not, no graft can be made')


def solve_vertex(vertex, lines):
    """
    Solves for graft widths at a given vertex for the incident lines,
    ensuring rectangles can arrange to meet at an n-gon while preserving angles.
    """
    # Pre-calculate sine and cosine adjustments
    angle_adjustment = np.pi / 2.
    
    neighbors = []
    for line in lines:
        if line.start == vertex:
            neighbors.append(line.end)
        else:
            assert line.end == vertex, 'lines must be incident to vertex'
            neighbors.append(line.start)

    angles = np.array([absolute_angle(vertex, n) for n in neighbors])

    angle_matrix = np.vstack([np.cos(angles + angle_adjustment), np.sin(angles + angle_adjustment)])
    
    # Initialize constraints vector
    constraints = np.zeros(angle_matrix.shape[0])
    
    # Update angle matrix and constraints for lines with predefined graft_width
    for i, line in enumerate(lines):
        graft_width = getattr(line, 'graft_width', None)
        if graft_width is not None:
            constraint_row = np.zeros(len(lines))
            constraint_row[i] = 1  # Impose constraint on the i-th line
            
            angle_matrix = np.concatenate((angle_matrix, constraint_row.reshape(1, -1)), axis=0)
            constraints = np.append(constraints, graft_width)

    
    # Solve for graft widths that are uniform. TODO maybe implement different strategies.
    try:
        widths_solution = find_uniform_solution(angle_matrix, constraints)
    except ValueError as e:
        print('vertex {} with coordinates ({},{}) could not be solved!'.format(vertex.name,vertex.x, vertex.y))
        raise e

    
    # Assign graft widths to lines without predefined widths
    for width, line in zip(widths_solution, lines):
        if not hasattr(line, 'graft_width'):
            setattr(line, 'graft_width', width)

def find_uniform_solution(X, b, min_width=0.):
    n = X.shape[1]
    x = cp.Variable(n)
    
    # Objective: Minimize sum of absolute differences between elements of x
    # This is a proxy for uniformity
    #objective = cp.Minimize(sum([cp.abs(x[i] - x[j]) for i in range(n) for j in range(i + 1, n)]))

    # Experimental: try to keep it close to 10. 
    objective = cp.Minimize(sum([cp.abs(x[i]-10)**2 for i in range(n)]))
        
    # Experimental: screw it, maximize it
    #objective = cp.Minimize(-sum([cp.abs(x[i])**2 for i in range(n)]))

    # Constraints include Ax = b and x >= 0
    constraints = [X @ x == b, x >= min_width]
    
    # Define and solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if x.value is not None:
        return(x.value)
    else:
        raise ValueError('No solution found! Check that graft path is a spiderweb. If not, no graft can be made')