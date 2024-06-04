import numpy as np
import copy
from copy import deepcopy
import warnings
import random


PRECISION=4 # decimal places for storing coords
TOL = 10**(-PRECISION)

class Vertex:
    def __init__(self, x, y, name=None, boundary=False):
        self.x = x
        self.y = y
        self.precision = PRECISION
        if name is None:
            name = '({},{})'.format(np.round(x,2),np.round(y,2))
        self.name = name
        self.boundary = boundary
        self.lines = [] # A set to store lines connected to this vertex
        self.faces = []  # A set to store faces this vertex is part of
    
    def reset(self):
        self.lines = []
        self.faces = []

    def add_line(self, line):
        """Add a line to the set of lines connected to this vertex."""
        self.lines.append(line)

    def remove_line(self, line):
        """Remove a line from the set of lines connected to this vertex."""
        self.lines = [x for x in self.lines if id(x) != id(line)]

    def add_face(self, face):
        """Add a face to the set of faces this vertex is part of."""
        self.faces.append(face)

    def remove_face(self, face):
        """Remove a face from the set of faces this vertex is part of."""
        self.faces = [x for x in self.faces if id(x) != id(face)]

    def __hash__(self):
        #return(id(self))
        return hash((np.round(self.x, PRECISION), np.round(self.y, PRECISION)))

    def __eq__(self, other):
        # if not isinstance(other, type(self)): return NotImplemented
        #print(TOL)
        #return id(self)==id(other)
        return self.__hash__() == other.__hash__()
        #return np.allclose([self.x, self.y], [other.x, other.y], atol=TOL)
        #return (self.x, self.y) == (other.x, other.y)
    
    # Define the less-than method for sorting
    def __lt__(self, other):
        # if not isinstance(other, type(self)):
        #     return NotImplemented
        return (np.round(self.x, PRECISION), np.round(self.y,PRECISION)) < (np.round(other.x, PRECISION), np.round(other.y, PRECISION))


    def __repr__(self):
        return f"Vertex({np.round(self.x, PRECISION)}, {np.round(self.y, PRECISION)})"
    
    def rotate(self, angle, center=None):
        """Rotate the vertex around a given center by the angle (in radians)."""
        if center is None:
            center = Vertex(0, 0)

        # Translate vertex to origin
        x_translated = self.x - center.x
        y_translated = self.y - center.y

        # Perform rotation
        x_rotated = x_translated * np.cos(angle) - y_translated * np.sin(angle)
        y_rotated = x_translated * np.sin(angle) + y_translated * np.cos(angle)

        # Translate back
        self.x = x_rotated + center.x
        self.y = y_rotated + center.y
        #self.snap()

    def dilate(self, factor, center=None):
        """Dilate the vertex from a given center by the factor."""
        if center is None:
            center = Vertex(0, 0)

        self.x = center.x + (self.x - center.x) * factor
        self.y = center.y + (self.y - center.y) * factor
    
    def snap(self):
        # snap to precision
        self.x = round(self.x, PRECISION)
        self.y = round(self.y, PRECISION)

    def norm(self):
        return np.sqrt(self.x**2 + self.y**2)

    def __deepcopy__(self, memo):
        # Create a new Vertex instance without copying lines and faces
        copied_vertex = Vertex(self.x, self.y,
                                self.name, self.boundary)
        memo[id(self)] = copied_vertex
        return copied_vertex
    

class Line:
    def __init__(self, start, end, line_type='crease', name=None, boundary=False):
        if name is None:
            self.name = '({},{})'.format(start.name, end.name)
        else:
            self.name = name

        self.start = start
        self.end = end

        self.type = line_type  # 'mountain', 'valley', 'crease', 'boundary', etc.
        self.length = np.sqrt((start.x-end.x)**2+(start.y-end.y)**2)
        self.boundary = boundary
        self.faces = [] # Initialize an empty set to store faces this line belongs to

        # self.start.add_line(self)
        # self.end.add_line(self)
    
    def get_length(self):
        return np.sqrt((self.start.x-self.end.x)**2+(self.start.y-self.end.y)**2)
    
    def reset(self):
        self.faces = set()

    def remove(self):
        self.start.remove_line(self)
        self.end.remove_line(self)

    def add_face(self, face):
        """Add a face to the set of faces that this line belongs to."""
        self.faces.append(face)

    def remove_face(self, face):
        """Remove a face from the set of faces that this line belongs to."""
        self.faces = [x for x in self.faces if id(x) != id(face)] 
        
    def __eq__(self, other):
        # two lines are the same only if they connect the exact same two vertices
        return (self.start is other.start) & (self.end is other.end)

        # self_low, self_high = sorted([self.start, self.end], key=lambda v: (v.x, v.y))
        # other_low, other_high = sorted([other.start, other.end], key=lambda v: (v.x, v.y))

        # return (self_low, self_high) == (other_low, other_high)

    def overlaps(self, other):
        return ((self.start == other.start) & (self.end == other.end)) | ((self.start == other.end) & (self.end == other.start))

    def __hash__(self):
        low, high = sorted([self.start, self.end], key=lambda v: (v.x, v.y))
        return hash((low,high))
        
    def rotate(self, angle, center=None):
        """Rotate the line around a given center by the angle (in radians)."""
        self.start.rotate(angle, center)
        self.end.rotate(angle, center)

    def dilate(self, factor, center=None):
        """Dilate the line from a given center by the factor."""
        self.start.dilate(factor, center)
        self.end.dilate(factor, center)
    
    def touches_boundary(self):
        return (self.start.boundary or self.end.boundary)
    
    @property
    def angle(self, degrees=False):

        return absolute_angle(self.start, self.end)
    
    def __repr__(self):
        return f"Line({self.start}, {self.end})"
    
    def __deepcopy__(self, memo):
        # Deepcopy start and end vertices. This will not bring their lines and faces.
        copied_start = copy.deepcopy(self.start, memo)
        copied_end = copy.deepcopy(self.end, memo)
        
        # Name and type can be directly copied since they are immutable or do not reference other complex objects
        copied_line = Line(copied_start, copied_end, self.type, self.name, self.boundary)
        memo[id(self)] = copied_line
        return copied_line
    

class Face:
    def __init__(self, vertices, lines=None, type='face', boundary=False):
        """
        Initializes a Face instance with a list of vertices in counter-clockwise order.
        """

        if vertices is None:
            vertices = []

        self.vertices = vertices
        self.boundary = boundary
        self.neighbors = [] # neighboring faces
        self.type = type

        if lines is None:
            self.lines = []
            # Create Line objects for each pair of consecutive vertices
            for i in range(len(vertices)):
                start_vertex = vertices[i]
                end_vertex = vertices[(i + 1) % len(vertices)]  # Wrap around to the first vertex
                self.lines.append(Line(start_vertex, end_vertex))
            self.type = type
        else:
            self.lines = lines

        # for vertex in self.vertices:
        #     vertex.add_face(self)

        # for line in self.lines:
        #     line.add_face(self)
        #     for face in line.faces:
        #         if face != self:
        #             self.neighbors.add(face)

    def update_neighbors(self):
        # recompute adjacent faces, and tell neighbors about self
        self.neighbors = []
        for line in self.lines:
            for face in line.faces:
                if face != self:
                    self.neighbors.append(face)
                    if self not in face.neighbors:
                        face.neighbors.append(self)

    def remove(self):
        for vertex in self.vertices:
            vertex.remove_face(self)
        for line in self.lines:
            line.remove_face(self)
            for face in line.faces:
                face.neighbors = [f for f in face.neighbors if id(f) != id(self)]

    def __eq__(self, other):
        """
        Checks if two faces are equal, meaning they have the same vertices in the same cyclic order.
        """


        verts_same = set([id(v) for v in self.vertices]) == set([id(v) for v in other.vertices])
        lines_same = set([id(l) for l in self.lines]) == set([id(l) for l in other.lines])

        return verts_same & lines_same

        # return id(self) == id(other)
        # #return self.canonical_representation() == other.canonical_representation()
    
    def centroid(self):
        mean_x = np.mean([v.x for v in self.vertices])
        mean_y = np.mean([v.y for v in self.vertices])
        return(Vertex(x=mean_x, y=mean_y))
    
    def canonical_representation(self):

        """
        Finds the canonical (rotation-invariant) representation of the face
        by selecting the rotation whose starting vertex is lexicographically smallest
        """
        rotations = [tuple([self.vertices[i - j] for i in range(len(self.vertices))]) for j in range(len(self.vertices))]


        rotations_r = [list(rot) for rot in rotations.copy()]
        for rot in rotations_r:
            rot.reverse()
        rotations_r = [tuple(rot) for rot in rotations_r]
        
        # Assuming each vertex can be represented as a hashable and comparable entity
        sorted_rotations = sorted(rotations) #, key=lambda rotation:rotation[0]  # sort rotations by first vertex
        sorted_rotations_r = sorted(rotations_r)#, key=lambda rotation:rotation[0])

        best_rot = sorted_rotations[0]
        best_rot_r = sorted_rotations_r[0]

        res = tuple(sorted([best_rot, best_rot_r])[0])

        return res

    def dilate(self, factor, center=None):
        for vert in self.vertices:
            vert.dilate(factor, center)


    def __hash__(self):
        """
        Computes the hash value of the face based on its canonical representation.
        """
        return hash(frozenset([id(x) for x in self.vertices]).union(frozenset([id(x) for x in self.lines])))
        # return id(self)
        #return hash(self.canonical_representation())

    def intersection(self, other):
            """
            Returns a list of shared vertices between self and other,
            ensuring the vertices are in clockwise order.
            """
            shared_vertices_indices = [self.vertices.index(v) for v in self.vertices if v in other.vertices]

            # Ensure shared vertices are consecutive; if not, cycle them.
            for i in range(len(shared_vertices_indices) - 1):
                if abs(shared_vertices_indices[i] - shared_vertices_indices[i + 1]) != 1:
                    # Found non-consecutive indices, cycle the shared vertices
                    shared_vertices_indices = shared_vertices_indices[i + 1:] + shared_vertices_indices[:i + 1]
                    break

            shared_vertices = [self.vertices[i] for i in shared_vertices_indices]
            
            return shared_vertices

    def __repr__(self):
        """
        String representation of the Face instance for debugging purposes.
        """
        return f"Face({', '.join([str(v) for v in sorted(self.vertices, key = lambda x: (x.x, x.y))])})"
    
    def __deepcopy__(self, memo):
        # Copy vertices and lines explicitly. Assuming vertices and lines have __deepcopy__ implemented.
        copied_vertices = deepcopy(self.vertices, memo)
        copied_lines = deepcopy(self.lines, memo)
        
        # Create a new Face instance with the copied details
        new_face = Face(copied_vertices, copied_lines, self.type, self.boundary)
        memo[id(self)] = new_face

        # Neighbors will need to be updated in the context of the entire structure they belong to,
        # as this requires knowledge about how faces are connected which is not available in isolation.
        # Therefore, neighbors are not copied here but should be handled after all faces are copied.
        
        return new_face
    
    

def angle_bisector(v1, v2, v3):
    """
    Calculate the unit vector representing the direction of the angle bisector 
    of the angle formed by three vertices v1-v2-v3 with v2 being the vertex.

    Args:
        v1, v2, v3: Vertex instances, each with .x and .y attributes.

    Returns:
        A tuple (bx, by) representing the unit vector in the direction of the bisector.
    """
    # Convert vertices to numpy arrays for vector operations
    vec1 = np.array([v1.x - v2.x, v1.y - v2.y])
    vec2 = np.array([v3.x - v2.x, v3.y - v2.y])

    # Normalize the vectors
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)

    # The bisector vector is the sum of the normalized vectors
    bisector_vec = vec1_norm + vec2_norm

    # Normalize the bisector vector to get a unit vector
    bisector_unit_vec = bisector_vec / np.linalg.norm(bisector_vec)

    return bisector_unit_vec[0], bisector_unit_vec[1]

def find_intersection(v1, d1, v2, d2):
    """
    Find the intersection of two lines given by points v1, v2 and direction vectors d1, d2.

    Args:
        v1, v2: Vertex instances representing points on each line.
        d1, d2: Tuples representing the direction vectors of each line.

    Returns:
        A Vertex instance representing the intersection point of the two lines.
    """
    # Line 1 represented as p1 + t * d1 = (x1, y1) + t * (dx1, dy1)
    # Line 2 represented as p2 + s * d2 = (x2, y2) + s * (dx2, dy2)

    # Convert direction vectors to numpy arrays
    d1_arr = np.array(d1)
    d2_arr = np.array(d2)

    # Matrix A containing direction vector components
    A = np.array([d1_arr, -d2_arr]).T  # Need to transpose to align with [t, s] column vector

    # Vector b containing the differences of the coordinates
    b = np.array([v2.x - v1.x, v2.y - v1.y])

    # Solve for [t, s] where A * [t, s] = b
    try:
        t, s = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        warnings.warn('Lines are parallel! Returning None')
        return None  # Lines are parallel or coincident

    # Calculate the intersection point using one of the line equations
    intersection_x = v1.x + t * d1_arr[0]
    intersection_y = v1.y + t * d1_arr[1]

    return (intersection_x, intersection_y)

def euclidean_dist(v1, v2):

    return(np.sqrt((v1.x-v2.x)**2+(v1.y-v2.y)**2))


def find_centroid(vertices):
    """Calculate the centroid of a list of vertices."""
    x_sum = sum(vertex.x for vertex in vertices)
    y_sum = sum(vertex.y for vertex in vertices)
    n = len(vertices)
    return Vertex(x_sum / n, y_sum / n)

def find_incenter(v1, v2, v3):
    # Lengths of sides opposite the vertices v1, v2, v3
    a = euclidean_dist(v2, v3)
    b = euclidean_dist(v1, v3)
    c = euclidean_dist(v1, v2)

    # Incenter calculation
    px = (a * v1.x + b * v2.x + c * v3.x) / (a + b + c)
    py = (a * v1.y + b * v2.y + c * v3.y) / (a + b + c)
    incenter = Vertex(px, py)

    return incenter



def scale_polygon(vertices, scale_factor):
    """Scale a polygon by a given factor towards/away from its centroid."""
    centroid = find_centroid(vertices)
    scaled_vertices = []
    for vertex in vertices:
        new_x = centroid.x + (vertex.x - centroid.x) * scale_factor
        new_y = centroid.y + (vertex.y - centroid.y) * scale_factor
        scaled_vertices.append(Vertex(new_x, new_y))
    return scaled_vertices

def find_angle(v1, v2, v3, degrees=False):
    # Create vectors
    vec1 = np.array([v1.x - v2.x, v1.y - v2.y])
    vec2 = np.array([v3.x - v2.x, v3.y - v2.y])
    
    # Calculate angle using the dot product and magnitude of vectors
    angle_rad = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    # Convert angle to degrees
    angle_deg = np.degrees(angle_rad)
    
    if degrees:
        return angle_deg
    else:
        return angle_rad

def find_angle_clockwise(v1, v2, v3):
    """
    Computes the clockwise angle from line v1-v2 to line v2-v3.
    
    Parameters:
        v1, v2, v3: Vertex instances, each with .x and .y attributes.
    
    Returns:
        Clockwise angle in radians between the two lines.
    """
    # Calculate direction vectors from v1 to v2 and from v2 to v3
    vector_a = (v1.x - v2.x, v1.y - v2.y)
    vector_b = (v3.x - v2.x, v3.y - v2.y)
    
    # Calculate angles of vector_a and vector_b from the X-axis
    angle_a = np.arctan2(vector_a[1], vector_a[0])
    angle_b = np.arctan2(vector_b[1], vector_b[0])
    
    # Calculate the clockwise angle from vector_a to vector_b
    angle_clockwise = angle_a - angle_b
    
    # If the angle is negative, add 2pi to get the clockwise angle
    if angle_clockwise < 0:
        angle_clockwise += 2 * np.pi
    
    return angle_clockwise

def find_next_edge_clockwise(start_vertex, end_vertex, edges=None):
    min_angle = 2 * np.pi  # Initialize with the maximum possible angle
    next_edge = None

    if edges is None:
        edges = end_vertex.lines

    for edge in edges:
        # Skip the incoming edge itself
        if (edge.start == start_vertex) or (edge.end == start_vertex):
            continue
        
        # Determine the other vertex of the edge
        if edge.start == end_vertex:
            other_vertex = edge.end
        elif edge.end == end_vertex:
            other_vertex = edge.start
        else: # edge is not incident
            continue
            
        # Calculate the clockwise angle from incoming_edge to the current edge
        angle = find_angle_clockwise(start_vertex, end_vertex, other_vertex)

        # Find the edge with the smallest clockwise angle (the next edge in the clockwise direction)
        if angle < min_angle:
            min_angle = angle
            next_edge = edge
    
    return next_edge


# def find_next_edge_clockwise(incoming_edge, current_vertex):
#     """
#     Finds the next edge clockwise from the incoming_edge at the current_vertex.
    
#     Parameters:
#         edges (list of Line objects): All edges connected to current_vertex.
#         incoming_edge (Line object): The edge coming into current_vertex.
#         current_vertex (Vertex object): The vertex from which to find the next edge.
    
#     Returns:
#         The next edge in the clockwise direction from incoming_edge.
#     """
#     min_angle = 2 * np.pi  # Initialize with the maximum possible angle
#     next_edge = None

#     edges = current_vertex.lines
    
#     for edge in edges:
#         # Skip the incoming edge itself
#         if edge == incoming_edge:
#             continue
        
#         # Determine the other vertex of the edge
#         if edge.start == current_vertex:
#             other_vertex = edge.end
#         elif edge.end == current_vertex:
#             other_vertex = edge.start
#         else: # edge is not incident
#             continue
            
#         # Calculate the clockwise angle from incoming_edge to the current edge
#         if incoming_edge.start == current_vertex:
#             angle = find_angle_clockwise(incoming_edge.end, current_vertex, other_vertex)
#         else:
#             angle = find_angle_clockwise(incoming_edge.start, current_vertex, other_vertex)
        
#         # Find the edge with the smallest clockwise angle (the next edge in the clockwise direction)
#         if angle < min_angle:
#             min_angle = angle
#             next_edge = edge
    
#     return next_edge

def absolute_angle(start, end, degrees=False):
    """ Angle with the positive x axis"""

    # Vector from start to end
    vector = (end.x - start.x, end.y - start.y)
    
    # Compute the angle in radians
    angle_rad = np.arctan2(vector[1], vector[0])
    
    # Optionally, convert the angle to degrees
    angle_deg = np.degrees(angle_rad)
    
    # make between 0 and 2pi
    if angle_rad < 0:
        angle_rad += 2*np.pi
        
    if angle_deg < 0:
        angle_deg += 360

    if degrees:
        return angle_deg
    else:
        return angle_rad
    

def rotate(vertices, lines, center_of_rotation, rho):
    #TODO these should be functions of vertices and lines...

    # Assuming rho is in radians
    rotation_matrix = np.array([[np.cos(rho), -np.sin(rho)], 
                                [np.sin(rho), np.cos(rho)]])

    # Deep copy the pattern to avoid modifying the original
    rotated_vertices = copy.deepcopy(vertices)
    rotated_lines = []
    rotation_map = {}

    for i,vertex in enumerate(rotated_vertices):
        # Translate vertex position relative to the center of rotation
        relative_pos = np.array([vertex.x - center_of_rotation.x, vertex.y - center_of_rotation.y])

        # Rotate and update vertex position
        rotated_pos = rotation_matrix.dot(relative_pos)
        vertex.x, vertex.y = rotated_pos + np.array([center_of_rotation.x, center_of_rotation.y])
        rotation_map[vertices[i]] = vertex
    
    for line in lines:
        rotated_line = Line(rotation_map[line.start], rotation_map[line.end])
        rotated_lines.append(rotated_line)

    return rotated_vertices, rotated_lines



def dilate(vertices, lines, center, factor):
    # Deep copy the pattern to avoid modifying the original
    dilated_vertices = copy.deepcopy(vertices)
    dilated_lines = []
    dilation_map = {}
    for i,vertex in enumerate(dilated_vertices):
        # Translate vertex position relative to the center
        relative_pos = np.array([vertex.x - center.x, vertex.y - center.y])
        # Dilate and update vertex position
        dilated_pos = relative_pos * factor

        vertex.x, vertex.y = dilated_pos + np.array([center.x, center.y])
        dilation_map[vertices[i]] = vertex

    for line in lines:
        dilated_line = Line(dilation_map[line.start], dilation_map[line.end])
        dilated_lines.append(dilated_line)

    return dilated_vertices, dilated_lines


def find_circumcenter(vertex1, vertex2, vertex3):
    # Calculate the midpoints of two sides
    mid_point1 = Vertex((vertex1.x + vertex2.x) / 2, (vertex1.y + vertex2.y) / 2)
    mid_point2 = Vertex((vertex2.x + vertex3.x) / 2, (vertex2.y + vertex3.y) / 2)
    
    # Calculate the slopes of the two sides
    slope1 = (vertex2.y - vertex1.y) / (vertex2.x - vertex1.x) if vertex2.x != vertex1.x else float('inf')
    slope2 = (vertex3.y - vertex2.y) / (vertex3.x - vertex2.x) if vertex3.x != vertex2.x else float('inf')
    
    # Calculate the slopes of the perpendicular bisectors
    perp_slope1 = -1 / slope1 if slope1 != 0 else float('inf')
    perp_slope2 = -1 / slope2 if slope2 != 0 else float('inf')
    
    # Solve for the circumcenter coordinates (x, y)
    # If the slopes are infinite, use the vertical formula directly
    if perp_slope1 == float('inf'):
        cx = mid_point1.x
        cy = perp_slope2 * (cx - mid_point2.x) + mid_point2.y
    elif perp_slope2 == float('inf'):
        cx = mid_point2.x
        cy = perp_slope1 * (cx - mid_point1.x) + mid_point1.y
    else:
        cx = (perp_slope1 * mid_point1.x - perp_slope2 * mid_point2.x + mid_point2.y - mid_point1.y) / (perp_slope1 - perp_slope2)
        cy = perp_slope1 * (cx - mid_point1.x) + mid_point1.y
    
    return Vertex(cx, cy)

def find_incident_lines(vertex, lines, return_index=False):
    """Find all edges incident to a given vertex."""
    if return_index:
        return [(i,edge) for i,edge in enumerate(lines) if edge.start == vertex or edge.end == vertex]
    else:
        return [edge for edge in lines if edge.start == vertex or edge.end == vertex]

def walk_around_face(lines, start_line, reverse=False):
    if not reverse:
        forward_pair = (start_line.start, start_line.end)
    else:
        forward_pair = (start_line.end, start_line.start)
    start_vertex = forward_pair[0]
    current_vertex = forward_pair[0]
    next_vertex = forward_pair[1]
    face = [start_vertex]
    face_lines = [start_line]
    while True:
        face.append(next_vertex)

        next_edge = find_next_edge_clockwise(lines, Line(current_vertex, next_vertex),current_vertex=next_vertex)
        face_lines.append(next_edge)


        if next_edge.start == next_vertex:
            current_vertex, next_vertex = next_vertex, next_edge.end
        elif next_edge.end == next_vertex:
            current_vertex, next_vertex = next_vertex, next_edge.start
        else:
            raise ValueError('issue with find_next_edge_clockwise')
        
        if next_vertex == start_vertex: # back around
            break
    return(Face(vertices=face, lines=face_lines))




def enumerate_faces(lines, return_boundary=False):
    visited_edges = set() # with orientation
    faces = []

    for start_edge in lines:
        #print(start_edge)
        forward_pair = (start_edge.start, start_edge.end)
        reverse_pair = (start_edge.end, start_edge.start)

        if forward_pair not in visited_edges:
            start_vertex = forward_pair[0]
            current_vertex = forward_pair[0]
            next_vertex = forward_pair[1]
            face = [start_vertex]
            while True:
                visited_edges.add(tuple([current_vertex, next_vertex]))

  

                face.append(next_vertex)

                next_edge = find_next_edge_clockwise(lines, Line(current_vertex, next_vertex),current_vertex=next_vertex)

                if next_edge.start == next_vertex:
                    current_vertex, next_vertex = next_vertex, next_edge.end
                elif next_edge.end == next_vertex:
                    current_vertex, next_vertex = next_vertex, next_edge.start
                else:
                    raise ValueError('issue with find_next_edge_clockwise')
                
                if next_vertex == start_vertex: # back around
                    break
                
            faces.append(Face(vertices=face))

        if reverse_pair not in visited_edges:
            start_vertex = reverse_pair[0]
            current_vertex = reverse_pair[0]
            next_vertex = reverse_pair[1]
            face = [start_vertex]
            while True:
                visited_edges.add(tuple([current_vertex, next_vertex]))

                face.append(next_vertex)

                next_edge = find_next_edge_clockwise(lines, Line(current_vertex, next_vertex),current_vertex=next_vertex)

                if next_edge.start == next_vertex:
                    current_vertex, next_vertex = next_vertex, next_edge.end
                elif next_edge.end == next_vertex:
                    current_vertex, next_vertex = next_vertex, next_edge.start
                else:
                    raise ValueError('issue with find_next_edge_clockwise')
                if next_vertex == start_vertex: # back around
                    break
            faces.append(Face(vertices=face))


    # find boundary face
    boundary = find_boundary_face(faces)
    interior_faces = list(set([x for x in faces if x != boundary]))
    
    if return_boundary:
        return interior_faces, boundary
    else:
        return interior_faces

def find_perpendicular(vertex, line):
    # Calculate the slope of the line
    dx = line.end.x - line.start.x
    dy = line.end.y - line.start.y
    if dx == 0:  # Vertical line
        perpendicular_x = line.start.x
        perpendicular_y = vertex.y
    elif dy == 0:  # Horizontal line
        perpendicular_x = vertex.x
        perpendicular_y = line.start.y
    else:
        slope = dy / dx
        perp_slope = -1 / slope
        # y - yp = m(x - xp), solve for y = mx + b, b = y - mx
        b_line = line.start.y - slope * line.start.x
        b_perp = vertex.y - perp_slope * vertex.x
        # Intersection of y = slope*x + b_line and y = perp_slope*x + b_perp
        perpendicular_x = (b_perp - b_line) / (slope - perp_slope)
        perpendicular_y = slope * perpendicular_x + b_line

    intersection = Vertex(perpendicular_x, perpendicular_y)
    perpendicular_line = Line(vertex, intersection)
    return perpendicular_line

def find_boundary(pattern):

    faces = enumerate_faces(pattern.lines)
    edge_count = {}

    # Count occurrences of each edge in all faces
    for face in faces:
        for i in range(len(face.vertices)):
            v1 = face.vertices[i]
            v2 = face.vertices[(i + 1) % len(face.vertices)]  # Wrap around to the first vertex
            edge = Line(v1, v2)
            if edge in edge_count:
                edge_count[edge] += 1
            else:
                edge_count[edge] = 1

    # Identify boundary edges
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

    return boundary_edges


def find_boundary_face(faces):
    vertices = []
    for f in faces:
        for v in f.vertices:
            vertices.append(v)
    
    directions = [np.array([random.uniform(-1, 1), random.uniform(-1, 1)]) for _ in range(50)]  # 15 random directions
    directions = [d / np.linalg.norm(d) for d in directions]  # Normalize
    
    boundary_candidates = []

    for d in directions:
        projections = [(np.dot(np.array([v.x, v.y]), d), v) for v in vertices]
        min_proj, max_proj = min(projections)[0], max(projections)[0]
        
        # Faces containing vertices that tie for min/max projection
        min_faces = [face for face in faces if any(v for v in face.vertices if np.dot(np.array([v.x, v.y]), d) == min_proj)]
        max_faces = [face for face in faces if any(v for v in face.vertices if np.dot(np.array([v.x, v.y]), d) == max_proj)]
        
        # Intersection of min_faces and max_faces gives potential boundary faces for this direction
        boundary_candidates.extend(list(set(min_faces) & set(max_faces)))

    # The true boundary face should be among the candidates for all directions
    # Count occurrences and return the most common face, handling ties as needed
    face_counts = {face: boundary_candidates.count(face) for face in set(boundary_candidates)}
    boundary_face = max(face_counts, key=face_counts.get)
    
    return boundary_face

def generate_face_adjacency_graph(faces):
    adjacency_graph = {i: [] for i in range(len(faces))}  # Initialize empty adjacency list for each face

    for i, face1 in enumerate(faces):
        for j, face2 in enumerate(faces):
            if i != j:
                # Count shared vertices between face1 and face2
                if len(face1.intersection(face2))>=2:
                    adjacency_graph[i].append(j)

    return adjacency_graph


def find_intersection_with_slope(point, slope, line):
    # Unpack the given point and slope
    px, py = point.x, point.y
    m1 = slope

    # Calculate the slope and intercept of the existing line
    dx = line.end.x - line.start.x
    dy = line.end.y - line.start.y

    if dx == 0:  # Vertical line case
        # The intersection x is simply the x of the line, y is calculated from m1
        intersect_x = line.start.x
        intersect_y = m1 * (intersect_x - px) + py
    elif dy == 0:  # Horizontal line case
        # The intersection y is simply the y of the line, x is solved from the equation of the new line
        intersect_y = line.start.y
        intersect_x = (intersect_y - py + m1 * px) / m1
    else:
        m2 = dy / dx
        b2 = line.start.y - m2 * line.start.x

        # Intersection of y = m1*(x - px) + py and y = m2*x + b2
        if m1 == m2:  # Parallel lines
            return None  # No intersection, or infinite intersections if b2 == py - m1 * px

        # Solve for x and y
        intersect_x = (py - b2 - m1 * px) / (m2 - m1)
        intersect_y = m1 * (intersect_x - px) + py

    intersection = Vertex(intersect_x, intersect_y)
    return intersection



