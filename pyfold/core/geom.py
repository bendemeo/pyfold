import numpy as np
import copy
import warnings


PRECISION=3 # decimal places for storing coords
TOL = 10**(-PRECISION)

class Vertex:
    def __init__(self, x, y, name='Vertex', boundary=False):
        self.x = round(x, PRECISION)
        self.y = round(y, PRECISION)
        self.precision = PRECISION
        self.name = name
        self.boundary = boundary

    def __hash__(self):
        return hash((self.x, self.y)    )

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return (self.x, self.y) == (other.x, other.y)

    def __repr__(self):
        return f"Vertex({self.x}, {self.y})"

class Line:
    def __init__(self, start_vertex, end_vertex, line_type='crease', name=None, boundary=False):
        if name is None:
            self.name = '({},{})'.format(start_vertex.name, end_vertex.name)
        else:
            self.name = name

        self.start, self.end = sorted([start_vertex, end_vertex], key=lambda v: np.linalg.norm([v.x, v.y]))

        self.type = line_type  # 'mountain', 'valley', 'crease', 'boundary', etc.
        self.length = np.sqrt((start_vertex.x-end_vertex.x)**2+(start_vertex.y-end_vertex.y)**2)
        self.boundary = boundary
        
    def __eq__(self, other):
        return (self.start, self.end) == (other.start, other.end)

    def __hash__(self):
        return hash((self.start, self.end))
        
    
    @property
    def angle(self, degrees=False):

        return absolute_angle(self.start, self.end)
    
    def __repr__(self):
        return f"Line({self.start}, {self.end})"
    
    



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
    

def rotate_pattern(pattern, center_of_rotation, rho):
    # Assuming rho is in radians
    rotation_matrix = np.array([[np.cos(rho), -np.sin(rho)], 
                                [np.sin(rho), np.cos(rho)]])

    # Deep copy the pattern to avoid modifying the original
    rotated_pattern = copy.deepcopy(pattern)

    for vertex in rotated_pattern.vertices:
        # Translate vertex position relative to the center of rotation
        relative_pos = np.array([vertex.x - center_of_rotation.x, vertex.y - center_of_rotation.y])

        # Rotate and update vertex position
        rotated_pos = rotation_matrix.dot(relative_pos)
        vertex.x, vertex.y = rotated_pos + np.array([center_of_rotation.x, center_of_rotation.y])

    return rotated_pattern


def dilate_pattern(pattern, center, factor):
    # Deep copy the pattern to avoid modifying the original
    dilated_pattern = copy.deepcopy(pattern)

    for vertex in dilated_pattern.vertices:
        # Translate vertex position relative to the center
        relative_pos = np.array([vertex.x - center.x, vertex.y - center.y])

        # Dilate and update vertex position
        dilated_pos = relative_pos * factor
        vertex.x, vertex.y = dilated_pos + np.array([center.x, center.y])

    return dilated_pattern


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


