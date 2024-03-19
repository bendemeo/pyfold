from .geom import find_angle, find_incident_edges, absolute_angle

def compute_angle_deficit(pattern):
    for vertex in pattern.vertices:
        incident_edges = find_incident_edges(vertex, pattern.lines)
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
        vertex.angle_deficit = angle_deficit