from .geom import Line, Vertex, Face, find_incenter, find_perpendicular
from .pattern import Pattern
import numpy as np
from scipy.spatial import Voronoi, Delaunay
import warnings
from ..utils.power import get_edges


def voronoi(verts):

    vor = Voronoi(np.array([[p.x, p.y] for p in verts]))

    vor_vertices = [Vertex(*v) for v in vor.vertices]

    res = Pattern()

    for v in vor_vertices:
        res.append_vertex(v)


    ridge_lines = []
    for vert_idx in vor.ridge_vertices:
        if np.min(vert_idx)>=0:
            start, end = [vor_vertices[i] for i in vert_idx]
            newline = Line(start, end, line_type='reference')
            res.append_line(newline, adjust_faces=False)
            #print(newline in abird.lines)
            ridge_lines.append(newline)
            #print([x in abird.lines for x in ridge_lines])
    res.reconstruct_faces()

    return(res)


def power_diagram(verts, radii):
    # make a power diagram
    res = Pattern()
    pts = np.array([[v.x, v.y] for v in verts])
    edges = get_edges(pts, radii) # from power.py
    
    for p1, p2 in edges:
        start_vert = Vertex(p1[0], p1[1])
        end_vert = Vertex(p2[0], p2[1])
        if start_vert in res.vertices:
            start_vert = res.vertices[res.vertices.index(start_vert)]
        if end_vert in res.vertices:
            end_vert = res.vertices[res.vertices.index(end_vert)]
        
        res.add_line(start_vert, end_vert, adjust_faces=False)
    
    return(res)