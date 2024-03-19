import numpy as np
from .geom import Vertex, Line, angle_bisector, find_intersection, euclidean_dist

def sawhorse_molecule(v1, v2, v3, v4, line_type='valley'):
    """Given four vertices describing a quadrilateral, fill in the sawhorse molecule"""

    vertices = [v1, v2, v3, v4]
    # first find all the angle bisectors
    vext = [v4,v1,v2,v3,v4,v1]
    bisectors = []
    bisectornames = []
    for i,v in enumerate(vext[:-2]):
        bisector_i = angle_bisector(vext[i], vext[i+1], vext[i+2])
        bisectors.append(bisector_i)
        bisectornames.append((vext[i].name, vext[i+1].name, vext[i+2].name))
        
        
    b1, b2, b3, b4 = bisectors

        
    # for each bisector, find where it intersects its neighbors and draw a line to the nearest
    bext = [b4,b1,b2,b3,b4,b1]

    newlines = []
    
    newverts = []

    for i, (v, d) in list(enumerate(list(zip(vext, bext))))[1:-1]:
        
        n1 = vext[i+1]
        nd1 = bext[i+1]

        n2 = vext[i-1]
        nd2 = bext[i-1]


        i1 = Vertex(*find_intersection(v, d, n1, nd1))
        i2 = Vertex(*find_intersection(v, d, n2, nd2))
        dist1 = euclidean_dist(v, i1)
        dist2 = euclidean_dist(v, i2)

        if dist1<=dist2:
            newline = Line(v, i1, line_type=line_type)
            newvert = i1
        else:
            newline = Line(v, i2, line_type=line_type)
            newvert = i2

        
        newlines.append(newline)
        newverts.append(newvert)
    
    newverts = list(set(newverts))
    connector = Line(*newverts)
    newlines.append(connector)
    
    return((newverts, newlines))
        
