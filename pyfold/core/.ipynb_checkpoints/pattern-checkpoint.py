class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Line:
    def __init__(self, start_vertex, end_vertex, line_type='crease'):
        self.start = start_vertex
        self.end = end_vertex
        self.type = line_type  # 'mountain', 'valley', 'crease', 'boundary', etc.

class Pattern:
    def __init__(self):
        self.vertices = []
        self.lines = []
        self.command_history = []  # For undo functionality

    def add_vertex(self, x, y):
        vertex = Vertex(x, y)
        self.vertices.append(vertex)
        return vertex

    def add_line(self, start_vertex, end_vertex, line_type='crease'):
        line = Line(start_vertex, end_vertex, line_type)
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

    def perform_command(self, command):
        command.execute()
        self.command_history.append(command)

    def undo_last_command(self):
        if self.command_history:
            command = self.command_history.pop()
            command.undo()