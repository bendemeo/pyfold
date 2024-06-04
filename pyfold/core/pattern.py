import numpy as np
# from .tesselations import construct_reciprocal_diagram
from .geom import generate_face_adjacency_graph, absolute_angle, find_incident_lines, find_angle, dilate, rotate, euclidean_dist
from .graft import solve_graft
from .geom import Vertex, Line, Face, find_next_edge_clockwise, find_boundary_face, find_centroid
from collections import deque
import plotly.graph_objects as go
from plotly.io import write_image
import numpy as np
from numbers import Number
import logging
from copy import deepcopy
import matplotlib.pyplot as plt
import warnings

line_cmap = {
    'mountain':'red',
    'valley':'blue',
    'reference':'lightgrey',
    'crease':'green',
    'boundary':'black'
}

class Pattern:
    def __init__(self):
        self.vertices = []
        self.lines = []
        self.command_history = []  # For undo functionality
        self.faces = []
        self.name_to_vertices = {} # maps vertex names to list(s) of vertices
        self.line_ids = set()
        self.vert_ids = set()

    def add_vertex(self, x, y, name=None, boundary=False):
        vertex = Vertex(x, y, name=name, boundary=boundary)
        
        return self.append_vertex(vertex)




    def append_vertex(self, vertex):

        if id(vertex) in self.vert_ids:
            #warnings.warn('vertex {} already in pattern; not adding'.format(vertex))
            return
        
        self.vert_ids.add(id(vertex))

        self.vertices.append(vertex)
        

        if vertex.name in self.name_to_vertices:
            self.name_to_vertices[vertex.name].append(vertex)
        else:
            self.name_to_vertices[vertex.name] = [vertex]

        return vertex

    def get_vertices_by_coords(self, query_vert):
        return [v for v in self.vertices if v == query_vert]
            
    def add_line(self, start, end, line_type='crease', adjust_faces=False):
        new_line = Line(start, end, line_type=line_type)
        return self.append_line(new_line, adjust_faces=adjust_faces)

    def append_line(self, new_line, adjust_faces=True):
        # add a given pre-constructed line to the pattern

        if id(new_line) in self.line_ids:
            #warnings.warn('line {} already in pattern; not adding'.format(new_line))
            return
        
        self.line_ids.add(id(new_line))

        if not self.has_vertex(new_line.start):
            self.append_vertex(new_line.start)
        
        if not self.has_vertex(new_line.end):
            self.append_vertex(new_line.end)
         
        # # # remove duplicate line, if exists
        self.lines.append(new_line)

        new_line.start.lines.append(new_line)
        new_line.end.lines.append(new_line)

        if adjust_faces:
            # Check for closed loops in both directions
            loop1 = self.walk_around_face(new_line, reverse=False)
            loop2 = self.walk_around_face(new_line, reverse=True)

            new_faces = []
            if loop1:
                new_faces.append(loop1)
            if loop2:
                new_faces.append(loop2)
            
            # print(loop1)
            # print(loop2)
            # edge case where this is the first face created
            if loop1 and loop2 and (loop1.canonical_representation() == loop2.canonical_representation()):
                new_faces = [loop1] # same face
                
            #new_faces = set(new_faces) # edge case where this is the first face created
            # print(new_faces)

            if len(new_faces) == 2:
                # The new line has divided an existing face into two
                self.divide_existing_face(new_faces, new_line)
            elif new_faces:
                # One new face created, add it directly to the pattern
                for face in new_faces:
                    self.append_face(face)
                    face.update_neighbors()
        return(new_line)

    def append_verts_lines(self, verts, lines):
        for v in verts:
            self.append_vertex(v)
        for l in lines:
            self.append_line(l)

    
    def remove_line(self, line):
        remove_idxs = []

        for i,l in enumerate(self.lines):
            if l == line:
                remove_idxs.append(i)
        
        for idx in reversed(remove_idxs):
            del self.lines[idx]
        
        line.remove()
        self.line_ids.discard(id(line))
    
    def add_face(self, vertices):
        new_face = Face(vertices=vertices)
        return self.append_face(new_face)
        
        # should I add line.add_face() and vert.add_face() for lines and vertices in face?

    def append_face(self, face):
        self.faces.append(face)
        for line in face.lines:
            line.add_face(face)
        for vert in face.vertices:
            vert.add_face(face)
        return face
    
    def remove_face(self, face):
        #ATTN: this removes the face and anything with the same footprint. Not ideal. 
        remove_idxs = []

        for i,f in enumerate(self.faces):
            if id(face)== id(f):
                remove_idxs.append(i)

        for idx in reversed(remove_idxs): # go backwards so indices don't change
            del self.faces[idx]

        face.remove() # clear pointers


    def divide_existing_face(self, new_faces, new_line):

        # # Collect lines from the two new faces, excluding the new_line
        # lines_from_new_faces = set()
        # for face in new_faces:
        #     lines_from_new_faces.update(set(face.lines))
        # lines_from_new_faces.discard(new_line)  # Exclude the newly added line

        lines_from_new_faces = set(list(new_faces)[0].lines).union(set(list(new_faces)[1].lines))

        # Identify the old face that contains all these lines
        old_face = None
        for face in self.faces:
            # Create a set of the face's lines for easier comparison
            face_lines_set = set(face.lines)
            #print(len(face_lines_set))
            #print(face_lines_set & lines_from_new_faces)

            # Check if this face's lines contain all lines from the new faces
            if face_lines_set.issubset(lines_from_new_faces):
                
                old_face = face
                old_face.remove() 
                self.remove_face(old_face)
                break
            
        # Add the new faces to the pattern
        for new_face in new_faces:
            self.append_face(new_face)

        for face in new_faces: # make sure adjacency graph is up to date
            face.update_neighbors()
        





    def find_line_by_id(self, line_id):
        for line in self.lines:
            if line.id == line_id:
                return line
        return None  # Or raise an exception/error

    def remove_isolated_vertices(self):
        # Identify all vertices that are part of at least one line
        connected_vertices = []
        for line in self.lines:
            connected_vertices.append(line.start)
            connected_vertices.append(line.end)

        # Filter the vertices list to only include those that are connected
        self.vertices = [vertex for vertex in self.vertices if vertex in connected_vertices]

    def walk_around_face(self, start_line, reverse=False):
        #TODO: this does not work if there is an interior edge invading the face...
        if not reverse:
            forward_pair = (start_line.start, start_line.end)
        else:
            forward_pair = (start_line.end, start_line.start)
        start_vertex = forward_pair[0]
        current_vertex = forward_pair[0]
        next_vertex = forward_pair[1]
        face = [start_vertex]
        visited_verts = [start_vertex]
        face_lines = [start_line]
        #next_edge = start_line 
        while True:
            face.append(next_vertex)

            visited_verts.append(next_vertex) 
            next_edge = find_next_edge_clockwise(current_vertex, next_vertex)

            if not next_edge:
                # terminated at leaf node; no face
                return(None)
            
            if next_edge.start == next_vertex:
                current_vertex, next_vertex = next_vertex, next_edge.end
            elif next_edge.end == next_vertex:
                current_vertex, next_vertex = next_vertex, next_edge.start
            else:
                raise ValueError('issue with find_next_edge_clockwise')

            face_lines.append(next_edge)
            
            if (next_vertex == start_vertex): # back around
                return(Face(vertices=face, lines=face_lines))
            elif(next_vertex in visited_verts):
                # print(next_vertex)
                # print(current_vertex)
                break
        
        return(None)

    def perform_command(self, command):
        command.execute()
        self.command_history.append(command)

    def undo_last_command(self):
        if self.command_history:
            command = self.command_history.pop()
            command.undo()
    
    def mark_boundary(self):
        boundary_face = find_boundary_face(self.faces)
        boundary_face.boundary = True

        for line in boundary_face.lines:
            line.boundary=True
            line.type = 'boundary'

        for v in boundary_face.vertices:
            v.boundary = True
    
    def solve_graft(self,  target_distances=None, min_distance=0.01):
        import cvxpy as cp

        if target_distances is None:
            dists = {}
            for line in self.lines:
                if line.boundary:
                    dists[line] = 0 # doesn't matter
                else:
                    assert len(line.faces) == 2, 'Line borders {} faces; should be 2'.format(len(line.faces))
                    centroids = [face.centroid() for face in line.faces]
                    centroid_dist = euclidean_dist(*centroids)
                    dists[line] = centroid_dist

            target_distances = dists

        constraint_matrices = []
        lines_list = list(self.lines)

        for v in self.vertices:
            if v.boundary: # don't try to solve boundary vertices
                continue

            #find incident lines

            lines = v.lines
            idxs = [lines_list.index(line) for line in lines] # index of constraint matrix

            # idxs, lines = list(zip(*find_incident_lines(v, self.lines, return_index=True)))
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
            
            new_constraint = np.zeros( (2, len(self.lines)))
            new_constraint[:,idxs] = angle_matrix
            constraint_matrices.append(new_constraint)
        
        constraint_matrix = np.vstack(constraint_matrices)
        print('constraint matrix nullspace has rank {}'.format(constraint_matrix.shape[1]-np.linalg.matrix_rank(constraint_matrix)))
        print('there are {} boundary lines'.format(len([l for l in self.lines if l.boundary])))

        
        #now solve it subject to nonzero distances, trying to keep it uniform.
        x = cp.Variable(len(self.lines))
        if target_distances is not None:
            print('solving to target...')
            objective = cp.Minimize(sum([cp.abs(x[i]-target_distances[lines_list[i]])**2 for i in range(len(self.lines))]))
            min_distance = max(np.min(list(target_distances.values()))/10., min_distance) # don't go below 1/10 the lowest dist
            constraints = [constraint_matrix @ x == np.zeros(constraint_matrix.shape[0]), x >= min_distance]
        else:
            print("solving constant...")
            objective = cp.Minimize(sum([cp.abs(x[i]-min_distance)**2 for i in range(len(self.lines))]))
            constraints = [constraint_matrix @ x == np.zeros(constraint_matrix.shape[0]), x >= min_distance]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        
        if x.value is not None:
            return(x.value)
        else:
            raise ValueError('No solution found! Check that graft path is a spiderweb. If not, no graft can be made')

    def construct_reciprocal_diagram(self, reconstruct_faces=False, **kwargs):
        if reconstruct_faces:
            self.reconstruct_faces()

        ds = self.solve_graft(**kwargs) #distances of dual edges

        edge_lengths = {l:d for l,d in zip(self.lines, ds)}

        recip = Pattern()
        faces_list = list(self.faces)

        # Initialize the position of the first face's vertex (in the dual graph) at the origin
        # Assuming face 0 as the root

        visited = set([faces_list[0]])
        queue = deque([faces_list[0]])

        # initialize mapping
        origin = recip.add_vertex(0,0)
        faces_list[0].dual_vertex = origin
        origin.dual_face = faces_list[0]
        #face_positions = {faces_list[0]:recip.add_vertex(0,0)}
        #self.face_to_dual_vertex = {}
        #self.vertex_to_dual_face = {}

        while queue:
            current_face = queue.popleft()
            if current_face.boundary:
                # stop here
                continue

            for adjacent_face in current_face.neighbors:
                if adjacent_face.boundary: # don't try to embed the boundary
                    continue
                shared_vertices = current_face.intersection(adjacent_face) # shared edge in counter-clockwise order
                assert len(shared_vertices)==2, "faces share more than 2 vertices; this case is not handled"
                
                # add the needed line between adjacent faces (regardless if visited)
                shared_lines = list(set(set(current_face.lines) & set(adjacent_face.lines)))
                assert len(shared_lines) == 1, "Two adjacent faces share {} lines!".format(len(shared_lines))

                shared_line = shared_lines[0]
                if adjacent_face not in visited:
                    # Find the shared edge to determine the direction and distance

                    edge_length = edge_lengths[shared_line]
                    edge_angle = absolute_angle(*shared_vertices)

                    # Calculate the direction vector for the reciprocal edge
                    # Orthogonal direction, hence subtracting pi/2 to rotate clockwise by pi/2.
                    # this is correct given counter-clockwise orderin of shared_vertices.
                    direction = np.array([np.cos(edge_angle - np.pi/2), np.sin(edge_angle - np.pi/2)])

                    # Calculate the new position by stepping in the direction by the edge length
                    new_position = np.array([current_face.dual_vertex.x, current_face.dual_vertex.y]) + direction * edge_length

                    dual_vert = recip.add_vertex(x=new_position[0], y=new_position[1])
                    adjacent_face.dual_vertex = dual_vert
                    dual_vert.dual_face = adjacent_face

                    #face_positions[adjacent_face] = recip.add_vertex(x=new_position[0], y=new_position[1])


                    visited.add(adjacent_face)
                    queue.append(adjacent_face)


                #primal_line = self.lines[self.lines.index(Line(shared_vertices[0], shared_vertices[1]))]
                if id(current_face)<id(adjacent_face): #ensures edge added only once
                    #print('recip has {} faces and {} lines'.format(len(recip.faces), len(recip.lines)))
                    if shared_line.start == shared_vertices[0]:
                        new_line = recip.add_line(current_face.dual_vertex, adjacent_face.dual_vertex)
                        #new_line = Line(face_positions[current_face], face_positions[adjacent_face])
                    elif shared_line.start == shared_vertices[1]: # be consistent about orientation.
                        new_line = recip.add_line(adjacent_face.dual_vertex,current_face.dual_vertex)
                    shared_line.dual_line = new_line
                    new_line.dual_line = shared_line
                    #print('recip has {} faces and {} lines'.format(len(recip.faces), len(recip.lines)))

                # if shared_line.boundary:
                #     new_line.boundary = True
                #     new_line.type = 'boundary'
                #     new_line.start.boundary = True
                #     new_line.end.boundary =True
                
                
        recip.reconstruct_faces()
        print('recip has {} faces and {} lines'.format(len(recip.faces), len(recip.lines)))

        # construct vertex to dual face mapping
        for v in self.vertices:
            if v.boundary:
                continue
            incident_faces = [f for f in v.faces if not f.boundary]
            incident_lines = [l for l in v.lines if not l.boundary]
            dual_face_verts = [f.dual_vertex for f in incident_faces]
            #print(dual_face_verts)
            dual_face_lines = [l.dual_line for l in incident_lines]
            # print(len(dual_face_lines))
            # print(len(dual_face_verts))
            # print(np.max([len(x.lines) for x in recip.faces]))
            #print(dual_face_lines)
            # dual_face = set.intersection(*[set(v.faces) for v in dual_face_verts])
            # dual_face = [face for face in dual_face if len(set(face.lines)&set(dual_face_lines))==len(set(face.lines))]

            # print([f for f in recip.faces if len(f.vertices)>10])
            dual_face = recip.faces[recip.faces.index(Face(vertices=dual_face_verts, lines=dual_face_lines))]
            v.dual_face = dual_face
            dual_face.dual_vertex = v

            # print(dual_face in recip.faces)

            #dual_face = [f for f in recip.faces if set(f.vertices).issubset(set(dual_face_verts)) & set(f.lines).issubset(set(dual_face_lines))]
            #assert len(dual_face) == 1, 'unhandled around primal vertex: {}'.format(v.name)
            #self.vertex_to_dual_face[v] = list(dual_face)[0]

        # mutually-reciprocal dual
        self.reciprocal_diagram = recip
        recip.reciprocal_diagram = self

        # # TODO make sure this changes when you shift the dual arround

        # boundary
        for v in recip.vertices:
            primal_face = v.dual_face
            if np.any([vert.boundary for vert in primal_face.vertices]):
                v.boundary = True
        for l in recip.lines:
            primal_line = l.dual_line
            if primal_line.touches_boundary():
                l.boundary = True
                l.type = 'boundary'
        for f in recip.faces:
            if np.all([l.boundary for l in f.lines]):
                f.boundary = True

    def merge(self, other, copy=True):
        vertex_map = {}
        result = Pattern()
        for v in self.vertices + other.vertices:
            if copy:
                new_vertex = result.add_vertex(v.x, v.y, name=v.name)
                vertex_map[v] = new_vertex
            else:
                result.append_vertex(v)
        
        for l in self.lines + other.lines:
            if copy:
                start = vertex_map[l.start]
                end = vertex_map[l.end]
                result.add_line(start=start, end=end, line_type=l.type, adjust_faces=False)
            else:
                result.append_line(l, adjust_faces=False)
        
        #result.reconstruct_faces()
            
        return(result)
    
        
    
    def has_vertex(self, vert):
        return id(vert) in self.vert_ids


    def dilate_towards_dual(self, factor=1.):

        if not hasattr(self, 'reciprocal_diagram'):
            logging.info('Constructing reciprocal diagram...')
            self.construct_reciprocal_diagram()
            logging.info('Reciprocal diagram constructed successfully')
        
        transformed_faces = []
        transformed_verts = []
        transformed_lines = []
        for primal_face in self.faces:
            if not hasattr(primal_face, 'dual_vertex'):
                continue
            dual_vert = primal_face.dual_vertex

            # dilate towards reciprocal dual vert
            transformed_face = deepcopy(primal_face)
            transformed_face.dilate(factor, center=dual_vert)

            transformed_face.origin_face = primal_face
            transformed_faces.append(transformed_face)

            # useful for finding neighbor pairs
            for i, l in enumerate(transformed_face.lines):
                l.origin_line = primal_face.lines[i]
            
            for i, v in enumerate(transformed_face.vertices):
                v.origin_vert = primal_face.vertices[i]

            transformed_verts += list(transformed_face.vertices)
            transformed_lines += list(transformed_face.lines)
        
        result = Pattern()
        result.faces = transformed_faces
        result.lines = transformed_lines
        result.vertices = transformed_verts

        return(result)

    def twist_tesselation(self, angle=0., factor=1.):

        primal_faces = self.dilate_towards_dual(factor=factor)

        dual_faces = self.reciprocal_diagram.dilate_towards_dual(factor=1.-factor)

        twist = merge_patterns(primal_faces, dual_faces)

        


        return dual_faces
    

    def reconstruct_faces(self, build_neighbor_graph=True, verbose=False):
        self.faces = []

        # clear face pointers
        for line in self.lines:
            line.faces = []
        for vert in self.vertices:
            vert.faces = []

        faces = set()

        for i,line in enumerate(self.lines):
            if verbose:
                print('{}/{}'.format(i, len(self.lines)), end='\r')

            face1 = self.walk_around_face(line, reverse=False)
            face2 = self.walk_around_face(line, reverse=True)

            if face1:
                faces.add(face1)
            if face2:
                faces.add(face2)


        for face in list(faces):
            self.append_face(face)
        
        if build_neighbor_graph:
            for face in self.faces:
                face.update_neighbors()

        
        #self.faces = list(faces)


    def contract_line(self, line):
        # squeeze a line to its midpoint
        midpoint = find_centroid([line.start, line.end])

        verts_to_connect = set()

        for l in line.start.lines:
            if l == line:
                continue
            if l.end == line.start:
                verts_to_connect.add(l.start)
            elif l.start == line.start:
                verts_to_connect.add(l.end)
            self.remove_line(l)

        for l in line.end.lines:
            if l == line:
                continue
            if l.end == line.end:
                verts_to_connect.add(l.start)
            elif l.start == line.end:
                verts_to_connect.add(l.end)
            self.remove_line(l)
        
        self.remove_line(line)
        self.remove_vertex(line.start)
        self.remove_vertex(line.end)
        self.append_vertex(midpoint)

        # print(len(verts_to_connect))
        # print(len(set(verts_to_connect)))
        for v in verts_to_connect:
            self.add_line(v, midpoint, adjust_faces=False)
        # print(len(midpoint.lines))
        # print(len(set(midpoint.lines)))
        

    def remove_vertex(self, vertex):
        self.vertices = [v for v in self.vertices if not (v is vertex)]
        self.vert_ids.discard(id(vertex))


    def construct_faces(self, inplace=True, return_boundary=False):
        visited_edges = [] # with orientation
        faces = []



        for start_edge in self.lines:
            #print(start_edge)
            forward_pair = (start_edge.start, start_edge.end)
            reverse_pair = (start_edge.end, start_edge.start)

            if forward_pair not in visited_edges:
                start_vertex = forward_pair[0]
                current_vertex = forward_pair[0]
                next_vertex = forward_pair[1]
                face = [start_vertex]
                face_lines = [self.lines[self.lines.index(Line(start_vertex, next_vertex))]]
                while True:
                    visited_edges.append(tuple([current_vertex, next_vertex]))

                    face.append(next_vertex)

                    next_edge = find_next_edge_clockwise(self.lines, Line(current_vertex, next_vertex),current_vertex=next_vertex)
                    face_lines.append(next_edge)

                    if next_edge.start == next_vertex:
                        current_vertex, next_vertex = next_vertex, next_edge.end
                    elif next_edge.end == next_vertex:
                        current_vertex, next_vertex = next_vertex, next_edge.start
                    else:
                        raise ValueError('issue with find_next_edge_clockwise')
                    
                    if next_vertex == start_vertex: # back around
                        break
                    
                faces.append(Face(vertices=face, lines=face_lines))

            if reverse_pair not in visited_edges:
                start_vertex = reverse_pair[0]
                current_vertex = reverse_pair[0]
                next_vertex = reverse_pair[1]
                face = [start_vertex]
                face_lines = [self.lines[self.lines.index(Line(start_vertex, next_vertex))]]
                while True:
                    visited_edges.append(tuple([current_vertex, next_vertex]))

                    face.append(next_vertex)

                    next_edge = find_next_edge_clockwise(self.lines, Line(current_vertex, next_vertex),current_vertex=next_vertex)
                    face_lines.append(next_edge)

                    if next_edge.start == next_vertex:
                        current_vertex, next_vertex = next_vertex, next_edge.end
                    elif next_edge.end == next_vertex:
                        current_vertex, next_vertex = next_vertex, next_edge.start
                    else:
                        raise ValueError('issue with find_next_edge_clockwise')
                    if next_vertex == start_vertex: # back around
                        break
                faces.append(Face(vertices=face, lines=face_lines))

        
        # find boundary face
        boundary = find_boundary_face(faces)
        interior_faces = list(set([x for x in faces if x != boundary]))

        # edge case of pattern with a hole in it (maybe to be filled later)
        interior_faces = [f for f in interior_faces if not np.all([l.boundary for l in f.lines])]

 
        
        if not inplace:
            if return_boundary:
                return interior_faces, boundary
            else:
                return interior_faces
        else:
            self.faces = interior_faces
            self.boundary = boundary

            for i,v in enumerate(boundary.vertices):
                v.boundary = True

                # make the lines boundary too
                next_v = boundary.vertices[(i+1) % len(boundary.vertices)]
                boundary_line = self.lines[self.lines.index(Line(v, next_v))]
                boundary_line.type='boundary'
                boundary_line.boundary=True
    
        
    def compute_angle_deficit(self):
    # compute angle deficits for interior vertices

        for vertex in self.vertices:
            if vertex.boundary:
                vertex.angle_deficit = np.nan

            incident_edges = find_incident_lines(vertex, self.lines)
            incident_edges.sort(key=lambda edge: absolute_angle(vertex, edge.end if edge.start == vertex else edge.start))

            angles = []
            for i in range(len(incident_edges)):
                # Get the next edge in the sorted list, wrapping around
                next_edge = incident_edges[(i + 1) % len(incident_edges)]
                
                # Calculate angle between current edge and next edge
                if incident_edges[i].end == vertex or incident_edges[i].start == vertex:
                    v1 = incident_edges[i].start if incident_edges[i].end == vertex else incident_edges[i].end
                    v2 = vertex
                    v3 = next_edge.end if next_edge.start == vertex else next_edge.start
                    
                    angle = find_angle(v1, v2, v3)
                    angles.append(angle)

            # Sum of odd and even angles
            odd_angles_sum = sum(angles[1::2])
            even_angles_sum = sum(angles[0::2])
            
            # Compute and assign angle deficit
            angle_deficit = abs(odd_angles_sum - even_angles_sum)
            vertex.angle_deficit = np.degrees(angle_deficit)

    


    def plot_interactive(self, width=800, height=800, path=None, linewidth=1., line_cmap=line_cmap, vsize=2, vertex_hover=True, line_hover=True):
    # Initialize an empty figure
        fig = go.Figure()

        # Create a trace for the middle points of the lines with invisible markers
        if line_hover:
            middle_node_trace = go.Scatter(
                x=[],  # Initialize as an empty list
                y=[],  # Initialize as an empty list
                text=[],  # Initialize as an empty list for hover text
                mode='markers',
                hoverinfo='text',
                showlegend=False,
                marker=dict(
                    opacity=0  # Make markers invisible
                )
            )

        # Loop through each line in your pattern
        for line in self.lines:
            x_values = [line.start.x, line.end.x]
            y_values = [line.start.y, line.end.y]
            line_name = line.name  # Assuming each line has a name attribute

            if isinstance(linewidth, Number):
                line_width = linewidth
            else:
                if hasattr(line, linewidth):
                    line_width=getattr(line, linewidth)
                else:
                    line_width=0.   

            # Append line trace to figure
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines',
                line=dict(color=line_cmap[line.type], width=line_width),
                showlegend=False,
                hoverinfo='none'  # No hover info for lines themselves
            ))

            # Calculate and append the midpoint for hover text
            if line_hover:
                midpoint_x = (line.start.x + line.end.x) / 2
                midpoint_y = (line.start.y + line.end.y) / 2
                middle_node_trace['x'] += (midpoint_x,)  # Add midpoint x as a tuple element
                middle_node_trace['y'] += (midpoint_y,)  # Add midpoint y as a tuple element
                middle_node_trace['text'] += (line_name,)  # Add line name as hover text

        # Add the middle node trace for line hover information
        if line_hover:
            fig.add_trace(middle_node_trace)

        for vertex in self.vertices:
            # Using vertex.name as hover text, assuming each vertex has a name attribute
            if vertex_hover:
                hoverinfo='text'
            else:
                hoverinfo='none'
            fig.add_trace(go.Scatter(x=[vertex.x], y=[vertex.y],
                                    mode='markers', hoverinfo=hoverinfo, 
                                    text=[vertex.name],
                                    marker=dict(color='black', size=vsize),  # Set all vertices to one color, e.g., blue
                                    showlegend=False))  # No legend for vertices


        # Update the layout to adjust the figure size and keep the axes to scale
        fig.update_layout(
            clickmode='event+select',
            width=width,
            height=height,
            autosize=False,
            paper_bgcolor='white',  # Sets the background color of the figure
            plot_bgcolor='white',   # Sets the background color of the plotting area
            xaxis=dict(
                scaleanchor='y',  # This constrains the aspect ratio to match the y-axis
                scaleratio=1,  # This ensures that one unit in x is equal to one unit in y
            ),
            yaxis=dict(
                constrain='domain'  # This option, combined with scaleanchor, helps keep the aspect ratio
            )
        )

        # Hide axis lines
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

        # Save the figure to an SVG file
        if path is not None:
            write_image(fig, path, format='svg')


        return fig
    
    def plot(self, width=8, height=8, linewidth=1., line_cmap=None, vsize=1, path=None, dpi=100,
             annot_verts=False, annot_lines=False, fontdict={}, ax=None, show=True):
        if line_cmap is None:
            line_cmap = {'mountain':'red',
                         'valley':'blue',
                         'reference':'gray',
                         'crease':'green',
                         'boundary':'black',
                         'default':'lightgray'}
            #line_cmap = {'default': 'black'}  # Default color map

        if ax is None:
            fig, ax = plt.subplots(figsize=(width, height))
            fig.set_dpi(dpi)

        # Plot each line with properties
        for line in self.lines:
            x_values = [line.start.x, line.end.x]
            y_values = [line.start.y, line.end.y]
            line_color = line_cmap.get(line.type, line_cmap['default'])
            ax.plot(x_values, y_values, color=line_color, linewidth=linewidth)

        # Plot each vertex
        for vertex in self.vertices:
            ax.plot(vertex.x, vertex.y, 'o', color='black', markersize=vsize)
            if annot_verts:
                ax.text(vertex.x, vertex.y, s=vertex.name, fontdict=fontdict)

        # Set plot properties to ensure the plot looks clean and properly proportioned
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')  # Hide axes

        
        # Save the figure if a path is provided
        if path is not None:
            plt.savefig(path, format='svg')
        
        if show:
            plt.show()


class TwistPattern(Pattern):
    """
    Class for twist tesselations; implements smart plotting and adjustment
    """

    def __init__(self, base_pattern, dilation_factor, rotation_angle, recompute_reciprocal=False, close_edges=False, **kwargs):
        self.base_pattern = base_pattern


        if recompute_reciprocal:
            base_pattern.compute_reciprocal_diagram(**kwargs)

        if not hasattr(base_pattern, 'reciprocal_diagram'):
            raise ValueError('base pattern has no reciprocal diagram. Compute it or set recompute_reciprocal to True to compute.')
        
        self.primal_faces = base_pattern.dilate_towards_dual(factor=dilation_factor)
        self.dual_faces = base_pattern.reciprocal_diagram.dilate_towards_dual(factor=1.-dilation_factor)



        # print(len(self.primal_faces.vertices))
        # print(len(self.dual_faces.vertices))
        #print(len(self.primal_faces.vertices.union(self.dual_faces.vertices)))

        twist = self.primal_faces.merge(self.dual_faces, copy=False)
        twist.reconstruct_faces()

        # clear mountain/valley assignments
        for line in twist.lines: 
            line.type = 'crease'

        origin_to_dilated_verts = {}
        for vert in self.primal_faces.vertices:
            if vert.boundary:
                if vert.origin_vert in origin_to_dilated_verts:
                    origin_to_dilated_verts[vert.origin_vert].append(vert)
                else:
                    origin_to_dilated_verts[vert.origin_vert] = [vert]
        

        if close_edges:
            for v1, v2 in origin_to_dilated_verts.values():
                twist.add_line(v1, v2, line_type='boundary', adjust_faces=False)



        #print(set(twist.faces) & set(self.primal_faces.faces))
        for face in twist.faces:
            if face in self.primal_faces.faces:
                face.type='primal face'
            elif face in self.dual_faces.faces:
                face.type='dual face'
            else:
                face.type = 'linking face'

        self.twist_pattern=twist

    def plot_interactive(self, show_primal=True, show_dual=True, shade_faces=True, width=600, height=600, **kwargs):
        fig = self.twist_pattern.plot_interactive(**kwargs)
        #fig = go.Figure()

        if shade_faces:
            new_traces = []
            legend_added = set()

            for face in self.twist_pattern.faces:

                # Determine if this type has been added to the legend
                if face.type in legend_added:
                    show_legend = False
                else:
                    show_legend = True
                    legend_added.add(face.type)
                
                x_coords = [vertex.x for vertex in face.vertices]
                y_coords = [vertex.y for vertex in face.vertices]

                new_trace = go.Scatter(x=x_coords, y=y_coords, fill='toself',
                                        mode='none',  # Change to 'lines+markers' if you want to show vertices
                                        line=dict(width=2),
                                        # Customize color based on face type
                                        fillcolor='lightblue' if face.type == 'primal face' else 'lightpink' if face.type == 'dual face' else 'whitesmoke' if face.type =='linking face' else 'white',
                                        name=face.type if show_legend else '',  # Only provide name for the first occurrence
                                        showlegend=show_legend) # Only show legend for the first occurrence)
                
                new_traces.append(new_trace)

            #print(len([f for f in self.twist_pattern.faces if f.type =='dual face']))
            # Extract the current traces from fig
            current_traces = list(fig.data)

            # Create a new list of traces with new_trace added at the beginning
            all_traces = new_traces + current_traces

            # Create a new figure with the reordered traces
            fig = go.Figure(data=all_traces, layout=fig.layout)

        return(fig)
    
    
    
    def assign_creases(self):
        # assign crease parity from an orientation of the primal lines
        # Will orient always from line.start to line.end 

        # primal_vert_to_dual_face = {y:x for x,y in self.dual_face_to_primal_vert.items()}
        # dual_vert_to_primal_face = {y:x for x,y in self.primal_face_to_dual_vert.items()}



        for face in self.linking_face_to_dual_line.keys():
            primal_line = self.linking_face_to_primal_line[face]
            dual_line = self.linking_face_to_dual_line[face]


            if primal_line.start in primal_vert_to_dual_face:
                start_face = primal_vert_to_dual_face[primal_line.start]
                start_line = list(set(start_face.lines) & set(face.lines))[0]
                start_line.type='mountain'

            if dual_line.start in dual_vert_to_primal_face:
                dual_start_face = dual_vert_to_primal_face[dual_line.start]
                dual_start_line = list(set(dual_start_face.lines) & set(face.lines))[0]
                dual_start_line = self.twist_pattern.lines[self.twist_pattern.lines.index(dual_start_line)]
                dual_start_line.type = 'valley'

            if primal_line.end in primal_vert_to_dual_face:
                end_face = primal_vert_to_dual_face[primal_line.end]
                end_line = list(set(end_face.lines) & set(face.lines))[0]
                end_line.type='valley'

            if dual_line.end in dual_vert_to_primal_face:
                dual_end_face = dual_vert_to_primal_face[dual_line.end]
                dual_end_line = list(set(dual_end_face.lines) & set(face.lines))[0]
                dual_end_line = self.twist_pattern.lines[self.twist_pattern.lines.index(dual_end_line)]
                dual_end_line.type = 'mountain'


        # for line in self.primal_pattern.lines:
        #     linking_face = self.line_to_linking_face[line]

        #     if not line.start.boundary:
        #         start_face = self.vert_to_dual_face[line.start]
        #         start_line = list(set(start_face.lines) & set(linking_face.lines))[0]
        #         start_line.type='mountain'

                
        #         start_face_dual = self.dual_pattern.vert_to_dual_face
        #         start_line_cross = list(set(start_face.lines) & set(link))
        #     if not line.end.boundary:

        #         end_face = self.vert_to_dual_face[line.end]
        #         end_line = list(set(end_face.lines) & set(linking_face.lines))[0]
        #         end_line.type='valley'

            



def merge_patterns(pattern1, pattern2):
    # Initialize the composite pattern
    composite_pattern = Pattern()

    # Helper function to find if a vertex exists in the pattern and return it
    def find_vertex(new_vertex, vertices):
        for vertex in vertices:
            if vertex.x == new_vertex.x and vertex.y == new_vertex.y:
                return vertex
        return None

    # Helper function to check if an edge already exists in the pattern
    def edge_exists(new_edge, edges):
        for edge in edges:
            if (edge.start == new_edge.start and edge.end == new_edge.end) or \
               (edge.start == new_edge.end and edge.end == new_edge.start):
                return True
        return False

    # Add vertices from pattern1
    for vertex in pattern1.vertices:
        existing_vertex = find_vertex(vertex, composite_pattern.vertices)
        if not existing_vertex:
            composite_pattern.vertices.append(vertex)

    # Add vertices from pattern2
    for vertex in pattern2.vertices:
        existing_vertex = find_vertex(vertex, composite_pattern.vertices)
        if not existing_vertex:
            composite_pattern.vertices.append(vertex)

    # Add edges from pattern1, making sure vertices are reused when possible
    for edge in pattern1.lines:
        start = find_vertex(edge.start, composite_pattern.vertices)
        end = find_vertex(edge.end, composite_pattern.vertices)
        if not edge_exists(Line(start, end), composite_pattern.lines):
            composite_pattern.lines.append(Line(start, end))

    # Add edges from pattern2, making sure vertices are reused when possible
    for edge in pattern2.lines:
        start = find_vertex(edge.start, composite_pattern.vertices)
        end = find_vertex(edge.end, composite_pattern.vertices)
        if not edge_exists(Line(start, end), composite_pattern.lines):
            composite_pattern.lines.append(Line(start, end))

    return composite_pattern



