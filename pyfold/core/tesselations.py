from .geom import Vertex, Line, rotate, dilate, find_circumcenter, find_next_edge_clockwise, enumerate_faces, absolute_angle
from .graft import solve_graft
import numpy as np
from scipy.spatial import Voronoi, Delaunay
from .pattern import Pattern, merge_patterns, TwistPattern
import copy
from collections import deque


def delaunay_triangulation(vertices, **kwargs):

    # Extract coordinates from vertices
    points = np.array([[v.x, v.y] for v in vertices])

    # Construct the Voronoi diagram
    vor = Voronoi(points)

    # Initialize an empty list to hold the edges
    edges = []

    # Iterate over Voronoi ridge points to find adjacent vertices
    for point_pair in vor.ridge_points:
        start_vertex = vertices[point_pair[0]]
        end_vertex = vertices[point_pair[1]]

        # Create an edge for each pair of adjacent vertices and add to the list
        edges.append(Line(start_vertex, end_vertex, **kwargs))

    return edges

def voronoi_diagram(vertices, **line_kwargs):
    # create Voronoi diagram from vertices
    result = Pattern()

    # add Voronoi vertices
    vor = Voronoi([[v.x, v.y] for v in vertices])
    vor_verts = [Vertex(*coords) for coords in vor.vertices.tolist()]
    result.vertices = vor_verts

    # add Voronoi edges
    for v1, v2 in vor.ridge_vertices:
        if np.min([v1, v2]) > 0: #-1 denotes point at infinity
            result.add_line(vor_verts[v1],vor_verts[v2], **line_kwargs)
    
    return(result)



def delaunay_to_voronoi_mapping(vertices):
    points = np.array([[vertex.x, vertex.y] for vertex in vertices])
    delaunay = Delaunay(points)

    # Mapping Delaunay triangles (faces) to their circumcenters (Voronoi vertices)
    face_to_circumcenter = {}
    for i, simplex in enumerate(delaunay.simplices):
        # simplex is a triangle in the Delaunay triangulation
        # i is the index to find its circumcenter

        delaunay_face = [vertices[index] for index in simplex]
        circumcenter = find_circumcenter(*delaunay_face)

        face_to_circumcenter[tuple(delaunay_face)] = circumcenter
    
    return face_to_circumcenter


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


def twist_and_scale(face, center, rotation_angle=0, scale_factor=0):

    transformed_face = copy.deepcopy(face)

    for v in transformed_face.vertices:
        v.dilate(factor=scale_factor,center=center)
        v.rotate(angle=rotation_angle, center=center)
    
    # lines come along for the ride
        
    return(transformed_face)


def twist_from_vertices(vertices, rotation_angle, scaling_factor):
    # Initialize the output Pattern with all vertices and no edges
    output_pattern = Pattern()
    #output_pattern.vertices = copy.deepcopy(vertices)
    
    # Step 2: Construct the Delaunay triangulation and add corresponding edges
    delaunay = Delaunay(np.array([[v.x, v.y] for v in vertices]))
    # for simplex in delaunay.simplices:
    #     for i in range(3):
    #         start_vertex = vertices[simplex[i]]
    #         for j in range(i+1, 3):
    #             end_vertex = vertices[simplex[j]]
    #             output_pattern.lines.append(Line(start_vertex, end_vertex))
    
    # Map for Delaunay to Voronoi
    del_to_vor_center = delaunay_to_voronoi_mapping(vertices)

    # map from vertex to transformed images
    vertex_to_transformed = {vertex: [] for vertex in vertices}

    face_to_transformed_face = {}
    # Step 3: Transform each face
    for simplex, voronoi_vertex in del_to_vor_center.items():
        face_vertices = simplex
        center = voronoi_vertex

        face_pattern = Pattern()
        face_pattern.vertices = copy.deepcopy(face_vertices)
        face_pattern.lines = [Line(face_pattern.vertices[0], face_pattern.vertices[1]),
                               Line(face_pattern.vertices[1], face_pattern.vertices[2]), 
                               Line(face_pattern.vertices[2], face_pattern.vertices[0])]
        
        transformed_face = dilate(face_pattern, center, scaling_factor)
        transformed_face = rotate(transformed_face, center, rotation_angle)
        face_to_transformed_face[tuple(face_pattern.vertices)]  = transformed_face.vertices

        output_pattern = merge_patterns(output_pattern, transformed_face)

        # Update mapping
        for original_vertex, transformed_vertex in zip(face_vertices, transformed_face.vertices):
            vertex_to_transformed[original_vertex].append(transformed_vertex)
        
        # # Connect original to transformed vertices
        # for original, transformed in zip(face_vertices, transformed_face.vertices):
        #     output_pattern.lines.append(Line(original, transformed))
    

    for simplex, neighbors in zip(delaunay.simplices, delaunay.neighbors):
        simplex_face = tuple(vertices[idx] for idx in simplex)

        for i, neighbor_simplex in enumerate(neighbors):
            if neighbor_simplex != -1:  # -1 indicates no neighbor
                neighbor_face = tuple(vertices[idx] for idx in delaunay.simplices[neighbor_simplex])
                shared_vertices = set(simplex_face) & set(neighbor_face)
                if len(shared_vertices) == 2:
                    transformed_simplex = face_to_transformed_face[simplex_face]
                    transformed_neighbor = face_to_transformed_face[neighbor_face]
                    for v in shared_vertices:
                        pair = list(set(vertex_to_transformed[v]) & (set(transformed_simplex) | set(transformed_neighbor)))
                        assert len(pair) < 3, "bug in simplex pairing"
                        if len(pair) > 1:
                            newline = Line(*pair)
                            if newline not in output_pattern.lines:
                                output_pattern.lines.append(newline)

    return output_pattern


def twist_tesselation(pattern, reconstruct_reciprocal=False, angle=0., factor=1.):
    if reconstruct_reciprocal or not hasattr(pattern, 'reciprocal_diagram'):
        pattern.construct_reciprocal_diagram()
    

    transformed_faces = []

    for face in pattern.faces:
        center = pattern.face_to_dual_vertex[face]

        transformed_face = twist_and_scale(face, center, angle, factor)
        transformed_faces.append(transformed_face)
    
    vert_to_dual_lines = {v:[] for v in pattern.vertices}

    vert_to_dual_face = {}
    line_to_linking_face = {v:[] for v in pattern.lines}

    # mappings linking twist tesselation to primal and dual elements
    dual_face_to_primal_vert = {}
    primal_face_to_dual_vert = {}
    linking_face_to_primal_line = {}
    linking_face_to_dual_line = {}

    res = Pattern()
    for i,face in enumerate(transformed_faces):
        for vertex in face.vertices:
            res.vertices.append(vertex)
        for line in face.lines:
            res.lines.append(line)
        face.type = 'primal face'

        res.faces.append(face)
        
        # find adjacent faces
        orig_face = pattern.faces[i]
        adjacent_faces = [pattern.faces[j] for j in pattern.face_adjacency_graph[i]]
        for adj_face in adjacent_faces:
            shared_verts = orig_face.intersection(adj_face)
            
            # face.vertices[shared_verts[0]], face.vertices[shared_verts[1]]

            shared_vert = shared_verts[0] # draw line for first clockwise vertex
            shared_vert_orig_idxs = [orig_face.vertices.index(v) for v in shared_verts]
            shared_vert_adj_idxs =[adj_face.vertices.index(v) for v in shared_verts]


            primal_line = Line(*shared_verts)
            transformed_line = Line(*[face.vertices[idx] for idx in shared_vert_orig_idxs])

            line_to_linking_face[primal_line].append(transformed_line)

            idx = pattern.faces.index(adj_face)

            transformed_adj = transformed_faces[idx]
            newline = Line(face.vertices[shared_vert_orig_idxs[0]], transformed_adj.vertices[shared_vert_adj_idxs[0]])
            newline.type='crease'
            res.lines.append(newline)

            if shared_vert.boundary:
                newline.boundary = True
                newline.type='boundary'
            
            line_to_linking_face[primal_line].append(newline)

            vert_to_dual_lines[shared_vert].append(newline)
            # for orig_idx, adj_idx in zip(shared_vert_orig_idx, shared_vert_adj_idx):
            #     res.add_line(face.vertices[orig_idx], transformed_adj.vertices[adj_idx])
        

    # mark dual faces and linking faces
    for v in pattern.vertices:
        # identify the dual face
        if v.boundary:
            continue
        dual_lines = vert_to_dual_lines[v]
        #walk around the lines, enumerating the vertex - id face
        face_pattern=Pattern(); face_pattern.lines=dual_lines
        _, dual_face = face_pattern.construct_faces(inplace=False, return_boundary=True)

        dual_face.type = 'dual face'

        res.faces.append(dual_face)
        vert_to_dual_face[v] = dual_face

    for l in pattern.lines:
        if l.boundary:
            continue
        linking_lines = line_to_linking_face[l]
        face_pattern=Pattern(); face_pattern.lines=linking_lines
        _, linking_face = face_pattern.construct_faces(inplace=False, return_boundary=True)

        linking_face.type = 'linking face'
        line_to_linking_face[l] = linking_face
        linking_face_to_primal_line[linking_face] = l
        linking_face_to_dual_line[linking_face] = pattern.line_to_dual_line[l]

        res.faces.append(linking_face)

    dual_face_to_primal_vert = {y:x for x,y in vert_to_dual_face.items()}
    primal_face_to_dual_vert = {face_t:pattern.face_to_dual_vertex[pattern.faces[i]] for i,face_t in enumerate(transformed_faces)}


    output = TwistPattern(twist_pattern=res, primal_pattern=pattern, dual_pattern=pattern.reciprocal_diagram,
                          dual_face_to_primal_vert = dual_face_to_primal_vert,
                          primal_face_to_dual_vert = primal_face_to_dual_vert,
                          linking_face_to_primal_line = linking_face_to_primal_line,
                          linking_face_to_dual_line = linking_face_to_dual_line)
    return(output)

# def construct_reciprocal_dual(pattern, **kwargs):

#     ds = solve_graft(pattern, **kwargs) #distances of dual edges
#     faces = enumerate_faces(pattern.lines, return_boundary=False)

#     adj = generate_face_adjacency_graph(faces)
#     start = 0

#     visited = set()
#     queue = deque([start])
#     start_v = [0,0]
#     cur_v = [0,0]

#     while queue:
#         current_face = queue.popleft()
#         if current_face not in visited:
#             visited.add(current_face)
#             print(f"Visiting face {current_face}")  # Example operation



#             # Enqueue all adjacent, unvisited faces
#             for neighbor in adj[current_face]:
#                 if neighbor not in visited:
#                     queue.append(neighbor)


# def construct_reciprocal_diagram(pattern, **kwargs):

#     faces = enumerate_faces(pattern.lines, return_boundary=False)

#     face_adjacency_graph = generate_face_adjacency_graph(faces)

#     ds = solve_graft(pattern, **kwargs) #distances of dual edges

#     edge_lengths = {l:d for l,d in zip(pattern.lines, ds) if not l.boundary}

#     result = Pattern()

#     # Initialize the position of the first face's vertex (in the dual graph) at the origin
#     face_positions = {0: Vertex(0,0)}  # Assuming face 0 as the root
#     visited = set([0])
#     queue = deque([0])
#     result.vertices.append(face_positions[0]) # add current face

#     while queue:
#         current_face = queue.popleft()

#         for adjacent_face in face_adjacency_graph[current_face]:
#             if adjacent_face not in visited:
#                 # Find the shared edge to determine the direction and distance

#                 shared_vertices = faces[current_face].intersection(faces[adjacent_face]) # shared edge in counter-clockwise order

#                 if len(shared_vertices) != 2:
#                     print(faces[current_face])
#                     print(faces[adjacent_face])
#                     print(shared_vertices)
#                 assert len(shared_vertices)==2, "faces share more than 2 vertices; this case is not handled"

#                 edge_length = edge_lengths[Line(shared_vertices[0], shared_vertices[1])]

#                 edge_angle = absolute_angle(*shared_vertices)

#                 # Calculate the direction vector for the reciprocal edge
#                 # Orthogonal direction, hence subtracting pi/2 to rotate clockwise by pi/2.
#                 # this is correct given counter-clockwise orderin of shared_vertices.
#                 direction = np.array([np.cos(edge_angle - np.pi/2), np.sin(edge_angle - np.pi/2)])

#                 # Calculate the new position by stepping in the direction by the edge length
#                 new_position = np.array([face_positions[current_face].x, face_positions[current_face].y]) + direction * edge_length
#                 face_positions[adjacent_face] = Vertex(x=new_position[0], y=new_position[1])

#                 result.vertices.append(face_positions[adjacent_face])
                

#                 visited.add(adjacent_face)
#                 queue.append(adjacent_face)

#             # add the needed line between adjacent faces (regardless if visited)    
#             result.add_line(face_positions[current_face], face_positions[adjacent_face])
    
#     return result


# def twist_from_vertices(vertices, rotation_angle, scaling_factor):
#     output_pattern = Pattern()
#     output_pattern.vertices = vertices[:]  # Copy original vertices

#     # Construct Delaunay triangulation
#     delaunay = Delaunay(np.array([[vertex.x, vertex.y] for vertex in vertices]))
    
#     # Vertex to transformed vertices mapping
#     vertex_to_transformed = {vertex: [] for vertex in vertices}

#     # Process each Delaunay triangle
#     for simplex in delaunay.simplices:
#         original_vertices = [vertices[i] for i in simplex]
#         circumcenter = find_circumcenter(*original_vertices)

#         transformed_vertices = [dilate_pattern([circumcenter], circumcenter, scaling_factor)[0],
#                                 rotate_pattern([circumcenter], circumcenter, rotation_angle)[0]
#                                 for _ in original_vertices]
        
#         # Update mapping
#         for original_vertex, transformed_vertex in zip(original_vertices, transformed_vertices):
#             vertex_to_transformed[original_vertex].append(transformed_vertex)
    
#     # Draw lines between transformed vertices from adjacent Delaunay triangles
#     for simplex, neighbors in zip(delaunay.simplices, delaunay.neighbors):
#         for i, neighbor_simplex in enumerate(neighbors):
#             if neighbor_simplex != -1:  # -1 indicates no neighbor
#                 shared_vertices = set(simplex) & set(delaunay.simplices[neighbor_simplex])
#                 for v_idx in shared_vertices:
#                     original_vertex = vertices[v_idx]
#                     for transformed_vertex in vertex_to_transformed[original_vertex]:
#                         for neighbor_transformed_vertex in vertex_to_transformed[vertices[v_idx]]:
#                             if transformed_vertex != neighbor_transformed_vertex:
#                                 output_pattern.lines.append(Line(transformed_vertex, neighbor_transformed_vertex))
    
#     return output_pattern

