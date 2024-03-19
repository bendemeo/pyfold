from .geom import Vertex, Line, rotate_pattern, dilate_pattern, find_circumcenter
import numpy as np
from scipy.spatial import Voronoi, Delaunay
from .pattern import Pattern, merge_patterns
import copy


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
        
        transformed_face = dilate_pattern(face_pattern, center, scaling_factor)
        transformed_face = rotate_pattern(transformed_face, center, rotation_angle)
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

                        # for transformed_vertex in vertex_to_transformed[original_vertex]:
                        #     for neighbor_transformed_vertex in vertex_to_transformed[vertices[v_idx]]:
                        #         if transformed_vertex != neighbor_transformed_vertex:
                        #             newline = Line(transformed_vertex, neighbor_transformed_vertex)
                        #             if newline not in output_pattern.lines:
                        #                 output_pattern.lines.append(Line(transformed_vertex, neighbor_transformed_vertex))

    # Draw lines between transformed vertices from adjacent Delaunay triangles
    # for simplex, neighbors in zip(delaunay.simplices, delaunay.neighbors):
    #     for i, neighbor_simplex in enumerate(neighbors):
    #         if neighbor_simplex != -1:  # -1 indicates no neighbor
    #             shared_vertices = set(simplex) & set(delaunay.simplices[neighbor_simplex])
    #             if len(shared_vertices) == 2:
    #                 for v_idx in shared_vertices:
    #                     original_vertex = vertices[v_idx]
    #                     for transformed_vertex in vertex_to_transformed[original_vertex]:
    #                         for neighbor_transformed_vertex in vertex_to_transformed[vertices[v_idx]]:
    #                             if transformed_vertex != neighbor_transformed_vertex:
    #                                 newline = Line(transformed_vertex, neighbor_transformed_vertex)
    #                                 if newline not in output_pattern.lines:
    #                                     output_pattern.lines.append(Line(transformed_vertex, neighbor_transformed_vertex))
    
    # for i, simplex in enumerate(delaunay.simplices):
    #         for j, neighbor_index in enumerate(delaunay.neighbors[i]):
    #             if neighbor_index != -1:
    #                 neighbor_simplex = delaunay.simplices[neighbor_index]
    #                 # Check if simplex and neighbor_simplex share an entire edge
    #                 shared_vertices = set(simplex) & set(neighbor_simplex)
    #                 if len(shared_vertices) == 2:  # They share an entire edge
    #                     # Find the transformed vertices corresponding to the shared edge
    #                     trans_vertices1 = vertex_to_transformed[vertices[list(shared_vertices)[0]]]
    #                     trans_vertices2 = vertex_to_transformed[vertices[list(shared_vertices)[1]]]
    #                     for v1 in trans_vertices1:
    #                         for v2 in trans_vertices2:
    #                             newline = Line(v1, v2)
    #                             if newline not in output_pattern.lines:
    #                                 output_pattern.lines.append(Line(v1, v2))
    return output_pattern

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