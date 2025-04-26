import math
import pygame
import random

import global_vars

###############################
# Graph Class
###############################

class Graph:
    def __init__(self):
        self.nodes = []           # list of (x, y)
        self.adjacency_list = {}  # node_index -> [neighbor_indices]
        self.edge_slopes = {}     # (n1, n2) -> slope in radians

    def add_node(self, pos):
        idx = len(self.nodes)
        self.nodes.append(pos)
        self.adjacency_list[idx] = []
        return idx
    
    def remove_node(self, pos):
        if not self.nodes:
            return  # Nothing to remove

        # Find the closest node
        closest_idx = min(range(len(self.nodes)),
                        key=lambda i: (self.nodes[i][0] - pos[0])**2 + (self.nodes[i][1] - pos[1])**2)

        # Remove from nodes
        self.nodes.pop(closest_idx)

        # Remove from adjacency list
        if closest_idx in self.adjacency_list:
            del self.adjacency_list[closest_idx]

        # Remove edges incident to this node
        to_delete = []
        for n, neighbors in self.adjacency_list.items():
            if closest_idx in neighbors:
                neighbors.remove(closest_idx)
            # Also adjust any neighbors with index > closest_idx (because indices shifted)
            self.adjacency_list[n] = [nbr - 1 if nbr > closest_idx else nbr for nbr in neighbors]

        # Remove edge slopes related to this node
        new_edge_slopes = {}
        for (n1, n2), slope in self.edge_slopes.items():
            if closest_idx in (n1, n2):
                continue  # Skip edges connected to the removed node
            # Shift indices if necessary
            n1_new = n1 - 1 if n1 > closest_idx else n1
            n2_new = n2 - 1 if n2 > closest_idx else n2
            new_edge_slopes[(n1_new, n2_new)] = slope
        self.edge_slopes = new_edge_slopes

        # Shift adjacency list keys
        new_adj_list = {}
        for n, neighbors in self.adjacency_list.items():
            n_new = n - 1 if n > closest_idx else n
            new_adj_list[n_new] = neighbors
        self.adjacency_list = new_adj_list

    def add_edge(self, n1, n2):
        # Creates an undirected edge between nodes n1 and n2 and computes its slope
        if n2 not in self.adjacency_list[n1]:
            self.adjacency_list[n1].append(n2)
        if n1 not in self.adjacency_list[n2]:
            self.adjacency_list[n2].append(n1)
        dx = self.nodes[n2][0] - self.nodes[n1][0]
        dy = self.nodes[n2][1] - self.nodes[n1][1]
        slope = math.atan2(dy, dx)
        self.edge_slopes[(n1, n2)] = slope
        self.edge_slopes[(n2, n1)] = math.atan2(-dy, -dx)

###############################
# Adjacency-Based Cycle Detection (DFS)
###############################

def detect_cycles_in_graph(g):
    # Returns a list of cycles, each a list of node indices in adjacency order.
    
    visited = set()
    stack = []
    final_cycles = []
    seen_sets = set()

    def dfs(node, parent):
        stack.append(node)
        visited.add(node)
        for nbr in g.adjacency_list[node]:
            if nbr == parent:
                continue
            if nbr not in visited:
                dfs(nbr, node)
            else:
                if nbr in stack:
                    cyc_start = stack.index(nbr)
                    cyc_path = stack[cyc_start:]
                    if len(cyc_path) > 1 and cyc_path[-1] == cyc_path[0]:
                        cyc_path = cyc_path[:-1]
                    cyc_set = frozenset(cyc_path)
                    if len(cyc_path) >= 3 and cyc_set not in seen_sets:
                        final_cycles.append(cyc_path[:])
                        seen_sets.add(cyc_set)
        stack.pop()

    for n in g.adjacency_list:
        if n not in visited:
            dfs(n, None)

    return final_cycles

###############################
# Intersection-Based Fill
###############################

def approximate_curve(start, end, curve_intensity, segments=20):
    # Returns a list of points approximating the curve from 'start' to 'end'
    # using the same midpoint + perpendicular logic as draw_jagged_or_curved_edge

    (x1, y1) = start
    (x2, y2) = end

    # no curve required, straight line
    if curve_intensity <= 0:
        return [start, end]

    mx = 0.5*(x1 + x2)
    my = 0.5*(y1 + y2)
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if abs(length) < 1e-9:
        return [start, end]

    perp = (-dy/length, dx/length)
    cx = mx + perp[0]*curve_intensity
    cy = my + perp[1]*curve_intensity

    # Generate points along the quadratic Bezier defined by start->ctrl->end
    # Parametric form: (1-t)^2 * start + 2(1-t)t * ctrl + t^2 * end
    points = []
    for i in range(segments+1):
        t = i / segments
        x_ = (1 - t)**2 * x1 + 2*(1 - t)*t*cx + t**2*x2
        y_ = (1 - t)**2 * y1 + 2*(1 - t)*t*cy + t**2*y2
        points.append((x_, y_))
    return points

def line_segment_intersect(p1, p2, p3, p4):
    # Returns (True, (ix, iy)) if line segments p1->p2 and p3->p4 intersect
    
    (x1, y1), (x2, y2) = p1, p2
    (x3, y3), (x4, y4) = p3, p4

    denom = (y4 - y3)*(x2 - x1) - (x4 - x3)*(y2 - y1)
    if abs(denom) < 1e-9:
        return (False, None)

    # parametric positions along each segment
    ua = ((x4 - x3)*(y1 - y3) - (y4 - y3)*(x1 - x3)) / denom
    ub = ((x2 - x1)*(y1 - y3) - (y2 - y1)*(x1 - x3)) / denom

    # Intersection is valid only if ua and ub lie in [0, 1]
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        ix = x1 + ua*(x2 - x1)
        iy = y1 + ua*(y2 - y1)
        return (True, (ix, iy))
    else:
        return (False, None)

def fill_composite_cycles_with_intersections(layer, surface, panel_rect,
                                            scale_3d, offset_x, offset_y):

    # Renders filled cycles (polygons) resulting from a set of 'composite edges'
    # in 3D space after projecting them into 2D. 
    
    # Intersections are detected and used to subdivide edges, 
    # ensuring that any polygons formed by these edges can be filled.
    
    transformed_nodes = []
    # Convert each composite edge to 2D global_vars.screen coords
    for (xx, yy) in layer.compositeGraph.nodes:

        # apply translate/offset
        xx_ = xx + layer.comp_offset_x
        yy_ = yy + layer.comp_offset_y

        # rotate yaw
        x_yaw = xx_*math.cos(layer.camera_yaw) - 0*math.sin(layer.camera_yaw)
        z_yaw = xx_*math.sin(layer.camera_yaw) + 0*math.cos(layer.camera_yaw)

        # rotate pitch
        y_pitch = yy_*math.cos(layer.camera_pitch) - z_yaw*math.sin(layer.camera_pitch)
        z_pitch = yy_*math.sin(layer.camera_pitch) + z_yaw*math.cos(layer.camera_pitch)

        # perspective divide
        denom = global_vars.PERSPECTIVE_DISTANCE - z_pitch
        if abs(denom) < 1e-9:
            denom = 1e-9  # Avoid division by zero
        pf = global_vars.PERSPECTIVE_DISTANCE/denom
        x_eff = x_yaw*pf
        y_eff = y_pitch*pf

        # scale and offset 
        sx = x_eff*scale_3d + offset_x
        sy = y_eff*scale_3d + offset_y
        transformed_nodes.append((sx, sy))

    segments_2d = []
    for (i, j) in layer.composite_edges:
        p1 = transformed_nodes[i]
        p2 = transformed_nodes[j]
        if i < j:
            segments_2d.append(((p1, p2), (i, j)))
        else:
            segments_2d.append(((p2, p1), (j, i)))

    # Prepare for intersection processing:
    # - vertex_list holds all 2D points (original + intersection points).
    # - adjacency_2d is a list of lists, storing edges as adjacency among vertices.
    vertex_list = list(transformed_nodes)
    adjacency_2d = [[] for _ in range(len(vertex_list))]

    from collections import defaultdict

    # maps edge key to list of intersection data
    splits = defaultdict(list)

    def add_vertex(pos):
        # Appends a new vertex 'pos' to the global vertex_list and adjacency_2d, returning its new index
        vertex_list.append(pos)
        adjacency_2d.append([])
        return len(vertex_list) - 1

    # Detect intersections among all edges. For each intersection, store its parametric 't' along the edge
    n_seg = len(segments_2d)
    for s1_idx in range(n_seg):
        ((p1a, p1b), (n1a, n1b)) = segments_2d[s1_idx]
        for s2_idx in range(s1_idx+1, n_seg):
            ((p2a, p2b), (n2a, n2b)) = segments_2d[s2_idx]
            shared = {n1a, n1b}.intersection({n2a, n2b})

            # If edges share a node, skip (we don't need to create an intersection for endpoints that are already connected).
            if shared:
                continue
            (intersects, ipos) = line_segment_intersect(p1a, p1b, p2a, p2b)
            if intersects and ipos:
                # Compute param t1 along edge1, t2 along edge2, used for correct ordering
                denom1 = math.hypot(p1b[0]-p1a[0], p1b[1]-p1a[1])
                t1 = 0 if denom1 < 1e-9 else ((ipos[0]-p1a[0])*(p1b[0]-p1a[0]) + (ipos[1]-p1a[1])*(p1b[1]-p1a[1])) / (denom1**2)
                denom2 = math.hypot(p2b[0]-p2a[0], p2b[1]-p2a[1])
                t2 = 0 if denom2 < 1e-9 else ((ipos[0]-p2a[0])*(p2b[0]-p2a[0]) + (ipos[1]-p2a[1])*(p2b[1]-p2a[1])) / (denom2**2)
                splits[(n1a, n1b)].append((t1, ipos))
                splits[(n2a, n2b)].append((t2, ipos))

    # For each edge with intersection splits, insert the new vertices and create sub-edges in adjacency_2d
    for (edge_key, sp_list) in splits.items():
        i, j = edge_key
        pA = transformed_nodes[i]
        pB = transformed_nodes[j]
        denom = math.hypot(pB[0]-pA[0], pB[1]-pA[1])
        if denom < 1e-9:
            continue
        sp_list_sorted = sorted(sp_list, key=lambda x: x[0])
        sub_verts = [(0, i)]
        for (tval, ipos) in sp_list_sorted:
            if 0 <= tval <= 1:
                new_idx = add_vertex(ipos)
                sub_verts.append((tval, new_idx))
        sub_verts.append((1, j))
        for sidx in range(len(sub_verts)-1):
            vA_idx = sub_verts[sidx][1]
            vB_idx = sub_verts[sidx+1][1]
            adjacency_2d[vA_idx].append(vB_idx)
            adjacency_2d[vB_idx].append(vA_idx)

    for ((p1, p2), (i, j)) in segments_2d:
        if (i, j) not in splits and (j, i) not in splits:
            adjacency_2d[i].append(j)
            adjacency_2d[j].append(i)

    visited_edges = set()
    polygons = []

    # Find all polygon cycles in the planar graph via DFS
    def walk_cycle(start_v, prev_v, path):
        current_v = path[-1]
        for nxt in adjacency_2d[current_v]:
            if nxt == prev_v:
                continue
            edge_key = tuple(sorted((current_v, nxt)))
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            if nxt == start_v and len(path) > 2:
                polygons.append(path[:])
            elif nxt not in path:
                path.append(nxt)
                walk_cycle(start_v, current_v, path)
                path.pop()

    for v_start in range(len(vertex_list)):
        for v_next in adjacency_2d[v_start]:
            visited_edges.add(tuple(sorted((v_start, v_next))))
            path = [v_start, v_next]
            walk_cycle(v_start, v_start, path)

    # Filter out duplicates or degenerate polygons and finalize
    final_polygons = []
    seen_sets = set()
    for poly in polygons:
        sset = frozenset(poly)
        if len(poly) >= 3 and sset not in seen_sets:
            seen_sets.add(sset)
            final_polygons.append(poly)

     # Create a temporary surface to draw polygons
    cycle_surf = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
    cycle_surf.fill((0, 0, 0, 0))

    # Draw each polygon (cycle) on the temporary surface
    for cyc in final_polygons:
        pts = []
        for vidx in cyc:
            vx, vy = vertex_list[vidx]
            px = vx - panel_rect.x
            py = vy - panel_rect.y
            pts.append((px, py))
        if len(pts) >= 3:
            pygame.draw.polygon(cycle_surf, tuple(layer.cycle_color), pts)

    # Finally, blit the temporary surface onto the main surface
    surface.blit(cycle_surf, (panel_rect.x, panel_rect.y))


###############################
# Edge Helper Functions
###############################

def recalc_edge_slopes(g):
    for n1, pos in enumerate(g.nodes):
        for n2 in g.adjacency_list.get(n1, []):
            dx = g.nodes[n2][0] - pos[0]
            dy = g.nodes[n2][1] - pos[1]
            slope = math.atan2(dy, dx)
            g.edge_slopes[(n1, n2)] = slope
            g.edge_slopes[(n2, n1)] = math.atan2(-dy, -dx)

def get_edge_points(start, end, noise_intensity, curve_intensity, segments=20):
    # Computes a list of points along an edge between start and end.
    # Also applies a curve (via a control point) and random noise.
    if curve_intensity > 0:
        mid = ((start[0]+end[0])*0.5, (start[1]+end[1])*0.5)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        perp = (-dy/length, dx/length) if length != 0 else (0, 0)
        control = (mid[0] + perp[0]*curve_intensity, mid[1] + perp[1]*curve_intensity)
    else:
        control = None

    pts = []
    for i in range(segments+1):
        t = i / segments
        if control is not None:
            x = (1 - t)**2 * start[0] + 2*(1 - t)*t*control[0] + t**2*end[0]
            y = (1 - t)**2 * start[1] + 2*(1 - t)*t*control[1] + t**2*end[1]
        else:
            x = start[0] + t*(end[0]-start[0])
            y = start[1] + t*(end[1]-start[1])
        if noise_intensity > 0:
            seed = hash((round(start[0],2), round(start[1],2), round(end[0],2), round(end[1],2), i))
            rng = random.Random(seed)
            x += rng.uniform(-noise_intensity, noise_intensity)
            y += rng.uniform(-noise_intensity, noise_intensity)
        pts.append((int(x), int(y)))
    return pts

def draw_jagged_or_curved_edge(surface, color, start, end, noise_intensity, curve_intensity, thickness=2):
    # Draws an edge from start to end with jagged/curved effects based on the given intensities
    pts = get_edge_points(start, end, noise_intensity, curve_intensity, segments=20)
    for i in range(len(pts)-1):
        pygame.draw.line(surface, color, pts[i], pts[i+1], thickness)