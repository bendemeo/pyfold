import numpy as np
from .geom import Vertex, Line

class Pattern:
    def __init__(self):
        self.vertices = []
        self.lines = []
        self.command_history = []  # For undo functionality

    def add_vertex(self, x, y, name='Vertex', boundary=False):
        vertex = Vertex(x, y, name=name, boundary=boundary)
        self.vertices.append(vertex)
        return vertex

    def add_line(self, start_vertex, end_vertex, line_type='crease', name=None):
        line = Line(start_vertex, end_vertex, line_type, name)
        self.lines.append(line)
        return line

    def delete_line(self, line_id):
        # Assumes each line has a unique identifier
        line_to_delete = None
        for line in self.lines:
            if line.id == line_id:
                line_to_delete = line
                break
        if line_to_delete:
            self.lines.remove(line_to_delete)
            return line_to_delete
        else:
            return None  # Or raise an exception/error

    def find_line_by_id(self, line_id):
        for line in self.lines:
            if line.id == line_id:
                return line
        return None  # Or raise an exception/error

    def remove_isolated_vertices(self):
        # Identify all vertices that are part of at least one line
        connected_vertices = set()
        for line in self.lines:
            connected_vertices.add(line.start)
            connected_vertices.add(line.end)

        # Filter the vertices list to only include those that are connected
        self.vertices = [vertex for vertex in self.vertices if vertex in connected_vertices]

    def perform_command(self, command):
        command.execute()
        self.command_history.append(command)

    def undo_last_command(self):
        if self.command_history:
            command = self.command_history.pop()
            command.undo()

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