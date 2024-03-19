

class Command:
    def __init__(self, pattern):
        self.pattern = pattern

    def execute(self):
        raise NotImplementedError("Execute method must be implemented by subclass")

    def undo(self):
        raise NotImplementedError("Undo method must be implemented by subclass")
    
class AddVertexCommand(Command):
    def __init__(self, pattern, x, y):
        super().__init__(pattern)
        self.x = x
        self.y = y
        self.vertex = None  # Will hold the created vertex

    def execute(self):
        self.vertex = self.pattern.add_vertex(self.x, self.y)

    def undo(self):
        self.pattern.vertices.remove(self.vertex)
    

class AddLineCommand(Command):
    def __init__(self, pattern, start_vertex, end_vertex, line_type='crease'):
        super().__init__(pattern)
        self.start_vertex = start_vertex
        self.end_vertex = end_vertex
        self.line_type = line_type
        self.line = None

    def execute(self):
        self.line = self.pattern.add_line(self.start_vertex, self.end_vertex, self.line_type)

    def undo(self):
        self.pattern.lines.remove(self.line)

class DeleteLineCommand(Command):
    def __init__(self, pattern, line):
        super().__init__(pattern)
        self.line = line
        self.index = None  # To remember the position for undo

    def execute(self):
        if self.line in self.pattern.lines:
            self.index = self.pattern.lines.index(self.line)
            self.pattern.lines.remove(self.line)

    def undo(self):
        if self.line and self.index is not None:
            self.pattern.lines.insert(self.index, self.line)

