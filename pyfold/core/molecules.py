from .geom import Line, Vertex, Face, find_incenter, find_perpendicular
from .pattern import Pattern
import numpy as np
from scipy.spatial import Voronoi, Delaunay
import warnings

def rabbit_ear(verts, line_type='valley'):
    inc = find_incenter(*verts)
    bisector_lines = []
    for v in verts:
        new_line = Line(v, inc, line_type=line_type)
        bisector_lines.append(new_line)

    res = Pattern()
    res.append_vertex(inc)
    for line in bisector_lines:
        res.append_line(line)

    return res


def delaunay(verts):
    deln = Delaunay(np.array([[p.x, p.y] for p in verts]))
    del_vertices = [Vertex(*v) for v in deln.points]

    res = Pattern()

    for v in del_vertices:
        res.append_vertex(v)

    for v1, v2, v3 in deln.simplices:
        if v1 < v2:
            res.add_line(del_vertices[v1], del_vertices[v2], adjust_faces=False)
        if v2 < v3:
            res.add_line(del_vertices[v2], del_vertices[v3], adjust_faces=False)
        if v3 < v1:
            res.add_line(del_vertices[v3], del_vertices[v1], adjust_faces=False)
    
    res.reconstruct_faces()
    return(res)





def angry_bird(poly, ridges_only=False):

    def in_hull(p, hull):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        from scipy.spatial import Delaunay
        if not isinstance(hull,Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p)>=0

    vor = Voronoi(np.array([[p.x, p.y] for p in poly]))

    vor_vertices = [Vertex(*v) for v in vor.vertices]

    abird = Pattern()
    poly_coords = np.array([[p.x, p.y] for p in poly])

    for v in vor_vertices:
        abird.append_vertex(v)

    poly_verts = []
    for p in poly:
        poly_verts.append(abird.add_vertex(p.x, p.y))
        
    poly_lines = []
    for i,p in enumerate(poly_verts):
        poly_lines.append(abird.add_line(poly_verts[i], poly_verts[(i+1)% len(poly_verts)],
                    line_type='boundary'))

    for i, p in enumerate(poly_verts):
        region = vor.regions[vor.point_region[i]]
        for vor_i in region:
            if vor_i >= 0:
                if in_hull(vor.vertices[vor_i], poly_coords):
                    line = Line(p, vor_vertices[vor_i], line_type='mountain')
                    abird.append_line(line)
                else:
                    print('vertex not in conv hull!')
                    line = Line(p, vor_vertices[vor_i], line_type='mountain')
                    abird.append_line(line)

    ridge_lines = []
    for vert_idx in vor.ridge_vertices:
        if np.min(vert_idx)>=0:
            start, end = [vor_vertices[i] for i in vert_idx]
            newline = Line(start, end, line_type='mountain')
            abird.append_line(newline)
            #print(newline in abird.lines)
            ridge_lines.append(newline)
            #print([x in abird.lines for x in ridge_lines])

    if ridges_only:
        lines_to_remove = [l for l in abird.lines if l.type=='boundary']

        for line in lines_to_remove:
            abird.remove_line(line)
        return(abird)
    
    #abird.reconstruct_faces()
    triangles = [f for f in abird.faces if (len(f.vertices )==3) and not f.boundary]
    for t in triangles:
        verts = t.vertices
        ear = rabbit_ear(verts)
        inc = ear.vertices[0]
        abird = abird.merge(ear, copy=False)

        for l in t.lines:
            if l.type == 'boundary':
                perp = find_perpendicular(inc, l)
                perp.type='mountain'
                abird.append_line(perp)


    #abird.reconstruct_faces()
    for line in ridge_lines:
        ridge = abird.lines[abird.lines.index(line)]
        face1 = abird.walk_around_face(ridge, reverse=False)
        face2 = abird.walk_around_face(ridge, reverse=True)
        # print(face1)
        # print(face2)

        if not (face1 and face2):
            warnings.warn('issue with ridge line: {}'.format(ridge))
            continue
        #assert face1 and face2, 'ridge faces not properly found'
        # if len(ridge.faces) != 2:
        #     warnings.warn('issue with ridge line: {}'.format(ridge))
        #     continue
        # face1, face2 = ridge.faces
        shared_line = line

        v1 = [v for v in face1.vertices if v not in face2.vertices][0]
        v2 = [v for v in face2.vertices if v not in face1.vertices][0]

        abird.add_line(v1, v2, line_type='mountain')
        abird.remove_line(shared_line)

    lines_to_remove = [l for l in abird.lines if l.type=='boundary']



    for line in lines_to_remove:
        abird.remove_line(line)
    # for line in abird.lines:
    #     if line.type=='boundary':
    #         print('removing')
    #         abird.remove_line(line)
    #         line.remove()
            

    
    return abird
    