import pygame
import sys
import math
import json
import random
from PIL import Image
import cv2
import noise
import numpy as np

pygame.init()

###############################
# Global Setup Variables
###############################

WIDTH, HEIGHT = 1400, 900 # Total window dimensions.
TOP_PANEL_HEIGHT = 30     # Height of the top instructions panel.

# Left GUI panel for sliders/buttons
GUI_PANEL_WIDTH = 350

# Panels:
MAIN_PANEL_WIDTH = 450      # primary (base) graph
RIGHT_PANEL_WIDTH = 600     # composite view

screen = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("GrammarScape: Interactive Graph Editor")

BG_COLOR = (30, 30, 30)
GRID_COLOR = (50, 50, 50)
GRID_SIZE = 20

NODE_SIZE = 20

pygame.font.init()
font = pygame.font.SysFont(None, 18)
instr_font = pygame.font.SysFont(None, 16)

perspective_distance = 300

# Row 1: Clear, Save, New Layer
clear_btn_rect = pygame.Rect(10, TOP_PANEL_HEIGHT+10, 90, 25)
capture_btn_rect = pygame.Rect(110, TOP_PANEL_HEIGHT+10, 90, 25)
new_layer_btn_rect = pygame.Rect(210, TOP_PANEL_HEIGHT+10, 120, 25)

# Row 2: Capture Right Panel and Load JSON
second_row_y = TOP_PANEL_HEIGHT + 45
load_json_btn_rect = pygame.Rect(10, second_row_y, 165, 25)
save_btn_rect  = pygame.Rect(185, second_row_y, 155, 25)

running = True
selected_node = None

right_panel_left_dragging = False
right_panel_middle_dragging = False
right_panel_right_dragging = False

last_mouse_left = (0, 0)
last_mouse_middle = (0, 0)
last_mouse_right = (0, 0)

minx = 0
miny = 0

canvas = cv2.imread('canvas.jpg')
if canvas is None:
    raise FileNotFoundError("Canvas texture 'canvas.jpg' not found. Please check your path.")
# Convert from BGR to RGB for consistency
canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

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
    # Convert each composite edge to 2D screen coords
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
        denom = perspective_distance - z_pitch
        if abs(denom) < 1e-9:
            denom = 1e-9  # Avoid division by zero
        pf = perspective_distance/denom
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
# Checkbox Class
###############################

class Checkbox:
    def __init__(self, x, y, label, value=False):
        self.rect = pygame.Rect(x, y, 16, 16)
        self.label = label
        self.value = value

    def draw(self, surf, font, mouse_pos):
        color = (200, 200, 0) if self.rect.collidepoint(mouse_pos) else (180, 180, 180)
        pygame.draw.rect(surf, color, self.rect, 2)
        if self.value:
            pygame.draw.line(surf, color, self.rect.topleft, self.rect.bottomright, 2)
            pygame.draw.line(surf, color, self.rect.topright, self.rect.bottomleft, 2)

        # Split label into two lines if there's a space
        parts = self.label.split()
        if len(parts) == 2:
            label1 = font.render(parts[0], True, (255, 255, 255))
            label2 = font.render(parts[1], True, (255, 255, 255))
            label_x = self.rect.right + 6
            surf.blit(label1, (label_x, self.rect.y - 2))
            surf.blit(label2, (label_x, self.rect.y + 10))
        else:
            label_surf = font.render(self.label, True, (255, 255, 255))
            surf.blit(label_surf, (self.rect.right + 6, self.rect.y - 2))

    def process_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.value = not self.value

###############################
# Slider Class
###############################
    
class Slider:
    # A simple horizontal slider
    def __init__(self, x, y, w, h, min_val, max_val, value, label="", is_int=False):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.is_int = is_int
        self.label = label
        self.handle_radius = h // 2
        self.dragging = False
        self._inline_offset = 0
        self._inline_width = self.rect.width


    def draw(self, surf, mouse_pos, font, show_label=True, inline=False):
        if inline and show_label:
            val_text = str(int(self.value) if self.is_int else round(self.value, 2))
            label_with_val = f"{self.label} ({val_text})"
            label_surf = font.render(label_with_val, True, (255, 255, 255))
            label_width = label_surf.get_width() + 8

            # Update internal tracking info for hit test
            self._inline_offset = self.rect.x + label_width
            self._inline_width = self.rect.width - label_width

            track_y = self.rect.y + self.rect.height // 2
            surf.blit(label_surf, (self.rect.x, self.rect.y - 1))

            pygame.draw.line(surf, (120, 120, 120),
                            (self._inline_offset, track_y),
                            (self._inline_offset + self._inline_width, track_y), 3)

            denom = (self.max_val - self.min_val) or 1
            t = (self.value - self.min_val) / denom
            handle_x = self._inline_offset + int(t * self._inline_width)
            handle_y = track_y
            handle_color = (200, 200, 0) if (self.dragging or self.handle_hit_test(mouse_pos)) else (180, 180, 180)
            pygame.draw.circle(surf, handle_color, (handle_x, handle_y), self.handle_radius)


        else:
            # Old layout
            self._inline_offset = self.rect.x
            self._inline_width = self.rect.width

            track_y = self.rect.y + self.rect.height // 2
            pygame.draw.line(surf, (120, 120, 120),
                            (self.rect.x, track_y),
                            (self.rect.x + self.rect.width, track_y), 3)

            denom = (self.max_val - self.min_val) or 1
            t = (self.value - self.min_val) / denom
            handle_x = self.rect.x + int(t * self.rect.width)
            handle_y = track_y
            handle_color = (200, 200, 0) if (self.dragging or self.handle_hit_test(mouse_pos)) else (180, 180, 180)
            pygame.draw.circle(surf, handle_color, (handle_x, handle_y), self.handle_radius)

            if show_label:
                val_text = str(int(self.value) if self.is_int else round(self.value, 2))
                label_text = f"{self.label}: {val_text}"
                surf.blit(font.render(label_text, True, (255, 255, 255)), (self.rect.x, self.rect.y - 18))


    def handle_hit_test(self, mouse_pos):
        denom = (self.max_val - self.min_val) or 1
        t = (self.value - self.min_val) / denom
        handle_x = self._inline_offset + int(t * self._inline_width)
        handle_y = self.rect.y + self.rect.height // 2
        return math.dist(mouse_pos, (handle_x, handle_y)) <= self.handle_radius + 2


    def process_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.handle_hit_test(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            rel_x = event.pos[0] - self._inline_offset
            t = rel_x / self._inline_width
            new_val = self.min_val + t*(self.max_val - self.min_val)

            if new_val < self.min_val:
                new_val = self.min_val
            if new_val > self.max_val:
                new_val = self.max_val
            if self.is_int:
                new_val = int(round(new_val))
            self.value = new_val

###############################
# Post-Process Effects
###############################

def apply_painterly_effect(post_group, top_group, post_process_intensity,
                           panel_width, panel_height, bg_color):
    # Applies an oil painting effect (with an added canvas blend) to the post_group surface.
    # Then it combines the processed post_group with the top_group and returns a new surface.
    
    if post_process_intensity > 0:
        rgb_post = pygame.Surface((panel_width, panel_height))
        rgb_post.fill(bg_color)
        rgb_post.blit(post_group, (0, 0))
        
        # Convert the post_group surface to a NumPy array
        arr = pygame.surfarray.array3d(rgb_post).transpose(1, 0, 2)
        # Convert from RGB (pygame) to BGR (OpenCV format)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        
        # Apply the oil painting effect and blend with canvas
        stylized_bgr = cv2.xphoto.oilPainting(bgr, 7, 2)
        canvas_resized = cv2.resize(canvas, (stylized_bgr.shape[1], stylized_bgr.shape[0]))
        stylized_bgr = cv2.addWeighted(stylized_bgr, 0.85, canvas_resized, 0.15, 0)
        
        # Convert back to RGB and then to a Pygame surface
        stylized_rgb = cv2.cvtColor(stylized_bgr, cv2.COLOR_BGR2RGB)
        stylized_surface = pygame.surfarray.make_surface(stylized_rgb.transpose(1, 0, 2))
        
        blend_factor = post_process_intensity / 10.0
        original_arr = pygame.surfarray.array3d(rgb_post).transpose(1, 0, 2)
        stylized_arr = pygame.surfarray.array3d(stylized_surface).transpose(1, 0, 2)
        blended_arr = cv2.addWeighted(original_arr, 1 - blend_factor, stylized_arr, blend_factor, 0)
        processed_post = pygame.surfarray.make_surface(blended_arr.transpose(1, 0, 2))
        
        new_surf = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        new_surf.blit(processed_post, (0, 0))
        new_surf.blit(top_group, (0, 0))
        return new_surf
    else:
        new_surf = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        new_surf.blit(post_group, (0, 0))
        new_surf.blit(top_group, (0, 0))
        return new_surf


def apply_composite_paint_splatters(surface, active_layer, base_scale,
                                    minx, miny, bw, bh, perspective_distance,
                                    panel_width, panel_height):
    # Computes transformed composite node positions from the active layer and uses them
    # to determine a deterministic seed and palette for drawing paint splatters.
    # It then calls add_paint_splatters to draw the splatters onto the given surface.
    
    scale_3d_active = base_scale * active_layer.camera_zoom
    offset_x_active = ((panel_width - bw * scale_3d_active) * 0.5 - minx * scale_3d_active +
                       active_layer.camera_offset[0])
    offset_y_active = ((panel_height - bh * scale_3d_active) * 0.5 - miny * scale_3d_active +
                       active_layer.camera_offset[1])
    comp_transformed = []
    for (xx, yy) in active_layer.compositeGraph.nodes:
        xx_ = xx + active_layer.comp_offset_x
        yy_ = yy + active_layer.comp_offset_y
        x_yaw = xx_ * math.cos(active_layer.camera_yaw)
        z_yaw = xx_ * math.sin(active_layer.camera_yaw)
        y_pitch = yy_ * math.cos(active_layer.camera_pitch) - z_yaw * math.sin(active_layer.camera_pitch)
        z_pitch = yy_ * math.sin(active_layer.camera_pitch) + z_yaw * math.cos(active_layer.camera_pitch)
        denom = perspective_distance - z_pitch
        if abs(denom) < 1e-6:
            denom = 1e-6
        pf = perspective_distance / denom
        comp_transformed.append((int(x_yaw * pf * scale_3d_active + offset_x_active),
                                 int(y_pitch * pf * scale_3d_active + offset_y_active)))
    
    # Compute a deterministic seed from the composite positions.
    splat_seed = int(sum(x + y for (x, y) in comp_transformed))
    # Define a palette using the active layer's colors.
    palette = [tuple(active_layer.edge_color),
               tuple(active_layer.node_color),
               tuple(active_layer.cycle_color[:3])]
    
    # Draw the splatters on the given surface.
    add_paint_splatters(surface, comp_transformed, palette, splatter_count=20,
                          splat_seed=splat_seed, splatter_value=active_layer.splatters)
    return surface

def add_paint_splatters(surface, transformed_nodes, palette, splatter_count=20, splat_seed=None, splatter_value=5):
    # Draws paint splatters on the given surface using a deterministic random generator.
    if not transformed_nodes or splatter_value == 0:
        return surface

    # Use a local random generator with a fixed seed for reproducibility.
    rng = random.Random(splat_seed) if splat_seed is not None else random

    xs = [p[0] for p in transformed_nodes]
    ys = [p[1] for p in transformed_nodes]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    # Add a 10% margin around the bounding box.
    margin_x = int(width * 0.1)
    margin_y = int(height * 0.1)

    # Adjust the frequency and size of splatters based on the splatter_value (0-100).
    adjusted_splatter_count = int(splatter_value * 0.3) + 3  # Number of splatters
    adjusted_radius_range = (int(splatter_value * 0.35) + 5, int(splatter_value * 0.35) + 5)  # Size of splatters

    for _ in range(adjusted_splatter_count):
        x = rng.randint(min_x - margin_x, max_x + margin_x)
        y = rng.randint(min_y - margin_y, max_y + margin_y)
        color = rng.choice(palette)
        radius = rng.randint(*adjusted_radius_range)  # Size based on splatter_value
        pygame.draw.circle(surface, color, (x, y), radius)
        # Draw several smaller droplets around for a splatter effect.
        for _ in range(rng.randint(1, 7)):
            dx = rng.randint(-radius, radius)
            dy = rng.randint(-radius, radius)
            small_radius = rng.randint(1, max(1, radius // 2))
            splatter_color = rng.choice(palette)
            pygame.draw.circle(surface, splatter_color, (x + dx, y + dy), small_radius)

###############################
# Utility Functions
###############################

def recalc_edge_slopes(g):
    for n1, pos in enumerate(g.nodes):
        for n2 in g.adjacency_list.get(n1, []):
            dx = g.nodes[n2][0] - pos[0]
            dy = g.nodes[n2][1] - pos[1]
            slope = math.atan2(dy, dx)
            g.edge_slopes[(n1, n2)] = slope
            g.edge_slopes[(n2, n1)] = math.atan2(-dy, -dx)

def load_project(filename):
    global layers, active_layer_index
    try:
        with open(filename, "r") as f:
            project_data = json.load(f)
    except Exception as e:
        print("Failed to load project:", e)
        return

    layers = []  # Clear current layers
    for layer_data in project_data["layers"]:
        layer = Layer(layer_data["name"])
        # Restore graph information with key conversion
        layer.graph.nodes = layer_data["graph"]["nodes"]
        layer.graph.adjacency_list = {int(k): [int(n) for n in v] 
                                      for k, v in layer_data["graph"]["adjacency_list"].items()}
        # Recalculate edge slopes so that composite graph building works properly.
        recalc_edge_slopes(layer.graph)
        
        # Restore settings
        settings = layer_data["settings"]
        layer.edge_color = settings["edge_color"]
        layer.cycle_color = settings["cycle_color"]
        layer.node_color = settings["node_color"]
        layer.edge_noise = settings["edge_noise"]
        layer.edge_curve = settings["edge_curve"]
        layer.edge_thickness = settings["edge_thickness"]
        layer.numIterations = settings["numIterations"]
        layer.composite_seed = settings["composite_seed"]
        layer.composite_length_seed = settings["composite_length_seed"]
        layer.composite_tolerance = settings["composite_tolerance"]
        layer.connection_length = settings["connection_length"]
        layer.merge_threshold = settings["merge_threshold"]
        layer.draw_composite_nodes = settings["draw_composite_nodes"]
        layer.use_duplicate_mode = settings["use_duplicate_mode"]
        layer.fill_cycles = settings["fill_cycles"]
        layer.post_process_intensity = settings["post_process_intensity"]
        layer.splatters = settings["splatters"]
        layer.camera_offset = settings["camera_offset"]
        layer.camera_zoom = settings["camera_zoom"]
        layer.camera_yaw = settings["camera_yaw"]
        layer.camera_pitch = settings["camera_pitch"]

        # Update slider values to reflect loaded settings
        layer.sliders[0].value = layer.edge_color[0]
        layer.sliders[1].value = layer.edge_color[1]
        layer.sliders[2].value = layer.edge_color[2]
        layer.sliders[3].value = layer.cycle_color[0]
        layer.sliders[4].value = layer.cycle_color[1]
        layer.sliders[5].value = layer.cycle_color[2]
        layer.sliders[6].value = layer.node_color[0]
        layer.sliders[7].value = layer.node_color[1]
        layer.sliders[8].value = layer.node_color[2]
        layer.sliders[9].value = layer.edge_noise
        layer.sliders[10].value = layer.edge_curve
        layer.sliders[11].value = layer.edge_thickness
        layer.sliders[12].value = layer.numIterations
        layer.sliders[13].value = layer.composite_seed
        layer.sliders[14].value = layer.composite_length_seed
        layer.checkboxes[0].value = layer.draw_composite_nodes
        layer.checkboxes[1].value = layer.use_duplicate_mode
        layer.checkboxes[2].value = layer.fill_cycles
        layer.sliders[15].value = layer.post_process_intensity
        layer.sliders[16].value = layer.splatters

        # Rebuild the composite graph now that the base graph is fully restored
        layer.build_composite_graph()
        layers.append(layer)
    active_layer_index = project_data.get("active_layer_index", 0)
    print("Project loaded successfully from", filename)

def snap_to_grid(pos):
    # Snaps a given position to the nearest grid intersection
    # The graph editor used to create the example graph leverages a grid 
    # We need to do this because the composite graph extends the graph based on edge slopes. 
    GRID_SIZE = 20
    x, y = pos
    return (round(x / GRID_SIZE) * GRID_SIZE, round(y / GRID_SIZE) * GRID_SIZE)

def draw_button(surf, text, rect, mouse_pos, font):
    color = (80, 80, 80)
    hover = (120, 120, 120)
    if rect.collidepoint(mouse_pos):
        pygame.draw.rect(surf, hover, rect)
    else:
        pygame.draw.rect(surf, color, rect)
    txt_surf = font.render(text, True, (255, 255, 255))
    surf.blit(txt_surf, (rect.x+5, rect.y+5))

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

###############################
# Layer Class
###############################

class Layer:
    """
    Each layer has:
      - A base Graph
      - Sliders for color, noise, etc.
      - Composite builder and cycle detection.
      - Per-layer camera transforms.
      - Additional composite offset.
      - Toggle for node circles.
      - Option to use duplicate mode.
      - Toggle for filling cycles.
      - A new slider for post-process effects.
    """
    def __init__(self, name):
        self.name = name
        self.graph = Graph()

        # Colors / edge effects
        self.edge_color = [200, 200, 200]
        self.cycle_color = [200, 200, 200, 150]
        self.node_color = [255, 255, 255]
        self.edge_noise = 0
        self.edge_curve = 0

        # Edge thickness
        self.edge_thickness = 2

        # Composite logic parameters
        self.numIterations = 3
        self.composite_seed = 0
        self.composite_length_seed = 0
        self.composite_tolerance = 0.0873
        self.connection_length = 100
        self.merge_threshold = 10

        # Composite offset
        self.comp_offset_x = 0
        self.comp_offset_y = 0
        # Toggle node circles
        self.draw_composite_nodes = 1
        # Duplicate mode flag
        self.use_duplicate_mode = 0
        # Fill cycles toggle
        self.fill_cycles = 0

        # Cycle storage
        self.base_cycles = []
        self.composite_nodes = []
        self.composite_edges = []
        self.compositeGraph = Graph()
        self.composite_cycles = []

        # Per-layer camera transforms
        self.camera_offset = [0, 0]
        self.camera_zoom = 1.0
        self.camera_yaw = 0.0
        self.camera_pitch = 0.0

        # Post-process Paint Effects
        self.post_process_intensity = 0
        self.splatters = 0

        # Checkboxes ----------------------

        slider_x = 10
        slider_y = TOP_PANEL_HEIGHT + 175
        slider_w = GUI_PANEL_WIDTH - 25
        slider_gap = 30
        smaller_gap = 20
        s_height = 16

        checkbox_x = 10
        checkbox_y = slider_y
        gap = 105
        self.checkboxes = [
            Checkbox(checkbox_x, checkbox_y, "Toggle Nodes", bool(self.draw_composite_nodes)),
            Checkbox(checkbox_x + gap, checkbox_y, "Copy Graph", bool(self.use_duplicate_mode)),
            Checkbox(checkbox_x + 2 * gap, checkbox_y, "Fill Cycles", bool(self.fill_cycles))
        ]
        slider_y += 85  # Move the Y cursor down after checkbox row

        # Sliders ------------------------

        # Build sliders (GUI)

        self.sliders = []
        # 1) Edge Color (R, G, B) – label them simply "R", "G", "B"
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height, 0, 255,
                                   self.edge_color[0], "R", True))
        slider_y += smaller_gap
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height, 0, 255,
                                   self.edge_color[1], "G", True))
        slider_y += smaller_gap
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height, 0, 255,
                                   self.edge_color[2], "B", True))
        slider_y += slider_gap + 10

        # 2) Cycle Color (R, G, B)
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height, 0, 255,
                                   self.cycle_color[0], "R", True))
        slider_y += smaller_gap
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height, 0, 255,
                                   self.cycle_color[1], "G", True))
        slider_y += smaller_gap
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height, 0, 255,
                                   self.cycle_color[2], "B", True))
        slider_y += slider_gap + 10

        # 3) Node Color (R, G, B)
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height, 0, 255,
                                   self.node_color[0], "R", True))
        slider_y += smaller_gap
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height, 0, 255,
                                   self.node_color[1], "G", True))
        slider_y += smaller_gap
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height, 0, 255,
                                   self.node_color[2], "B", True))
        slider_y += slider_gap + 35

        # 4) Noise & Curve
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height, 0, 50,
                                   self.edge_noise, "Noise", True))
        slider_y += slider_gap
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height, 0, 100,
                                   self.edge_curve, "Curve", True))
        slider_y += slider_gap

        # Edge thickness
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height, 0, 15,
                                   self.edge_thickness, "Thickness", True))
        slider_y += slider_gap + 35

        # Iterations, seeds
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height, 1, 10,
                                   self.numIterations, "Iterations", True))
        slider_y += slider_gap
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height, 0, 1000,
                                   self.composite_seed, "Seed - Composite", True))
        slider_y += slider_gap
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height, 0, 1000,
                                   self.composite_length_seed, "Seed - Edge Lengths", True))
        slider_y += slider_gap + 35

        # Post Process slider: 0..10
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height,
                                   0, 10, 0, "Painterly", True))
        slider_y += slider_gap

        # Splatter size and frequency: 0..25
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height,
                                   0, 25, 0, "Splatters", True))
        slider_y += slider_gap

    def process_sliders(self, events):
        for s in self.sliders:
            for e in events:
                s.process_event(e)
        for cb in self.checkboxes:
            for e in events:
                cb.process_event(e)


    def draw_sliders(self, surface, mouse_pos):

        # Tooltips setup
        tooltips = []

        # Draw checkboxes
        for cb in self.checkboxes:
            cb.draw(surface, font, mouse_pos)

        color_opts_label = font.render("Graph Color Palette", True, (255, 255, 0))
        color_opts_pos = (self.sliders[0].rect.x, self.sliders[0].rect.y - 45)
        color_opts_rect = color_opts_label.get_rect(topleft=color_opts_pos)
        surface.blit(color_opts_label, color_opts_pos)

        if color_opts_rect.collidepoint(mouse_pos):
            tooltips.append("Edit the colors of the graph\n in the current layer.")

        # 1) Edge color - sliders [0,1,2]
        ex, ey = self.sliders[0].rect.x, self.sliders[0].rect.y
        # Single label: "Edge RGB: R, G, B"
        edge_label = (
            f"Edge (RGB: {int(self.edge_color[0])}, "
            f"{int(self.edge_color[1])}, {int(self.edge_color[2])}"
            f")"
        )
        surface.blit(font.render(edge_label, True, (255, 255, 255)), (ex, ey - 20))
        # Draw the 3 sliders with show_label=False
        for i in range(0, 3):
            self.sliders[i].draw(surface, mouse_pos, font, show_label=False)

        # 2) Cycle color - sliders [3,4,5]
        cx, cy = self.sliders[3].rect.x, self.sliders[3].rect.y
        cycle_label = (
            f"Cycle (RGB: {int(self.cycle_color[0])}, "
            f"{int(self.cycle_color[1])}, {int(self.cycle_color[2])}"
            f")"
        )
        surface.blit(font.render(cycle_label, True, (255, 255, 255)), (cx, cy - 20))
        for i in range(3, 6):
            self.sliders[i].draw(surface, mouse_pos, font, show_label=False)

        # 3) Node color - sliders [6,7,8]
        nx, ny = self.sliders[6].rect.x, self.sliders[6].rect.y
        node_label = (
            f"Node (RGB: {int(self.node_color[0])}, "
            f"{int(self.node_color[1])}, {int(self.node_color[2])}"
            f")"
        )
        surface.blit(font.render(node_label, True, (255, 255, 255)), (nx, ny - 20))
        for i in range(6, 9):
            self.sliders[i].draw(surface, mouse_pos, font, show_label=False)

        # 4) The rest of the sliders are drawn normally with their existing labels

        # --- Edge Options Section ---
        edge_opts_label = font.render("Edge Options", True, (255, 255, 0))
        edge_opts_pos = (self.sliders[9].rect.x, self.sliders[9].rect.y - 25)
        edge_opts_rect = edge_opts_label.get_rect(topleft=edge_opts_pos)
        surface.blit(edge_opts_label, edge_opts_pos)

        if edge_opts_rect.collidepoint(mouse_pos):
            tooltips.append("Noise adds randomness. \nCurve bends edges. \nThickness controls width.")

        for i in range(9, 12):
            self.sliders[i].draw(surface, mouse_pos, font, show_label=True, inline=True)

        # --- Composite Graph Section ---
        comp_graph_label = font.render("Composite Graph", True, (255, 255, 0))
        comp_graph_pos = (self.sliders[12].rect.x, self.sliders[12].rect.y - 25)
        comp_graph_rect = comp_graph_label.get_rect(topleft=comp_graph_pos)
        surface.blit(comp_graph_label, comp_graph_pos)

        if comp_graph_rect.collidepoint(mouse_pos):
            tooltips.append("Controls how the composite graph\nis generated and expanded.")

        for i in range(12, 15):
            self.sliders[i].draw(surface, mouse_pos, font, show_label=True, inline=True)

        # --- Post-process Effects Section ---
        post_label = font.render("Post-process Effects", True, (255, 255, 0))
        post_label_pos = (self.sliders[15].rect.x, self.sliders[15].rect.y - 25)
        post_rect = post_label.get_rect(topleft=post_label_pos)
        surface.blit(post_label, post_label_pos)

        if post_rect.collidepoint(mouse_pos):
            tooltips.append("Painterly = oil paint effect. \nSplatters = paint droplets.")

        for i in range(15, len(self.sliders)):
            self.sliders[i].draw(surface, mouse_pos, font, show_label=True, inline=True)

        # --- Show tooltip if hovering ---
        for i, tip in enumerate(tooltips):
            tip_surf = font.render(tip, True, (255, 255, 255))
            bg_rect = tip_surf.get_rect()
            bg_rect.topleft = (mouse_pos[0] + 12, mouse_pos[1] + 12 + i*20)
            pygame.draw.rect(surface, (20, 20, 20), bg_rect.inflate(6, 4))
            pygame.draw.rect(surface, (255, 255, 0), bg_rect.inflate(6, 4), 1)
            surface.blit(tip_surf, bg_rect.topleft)

    def update_from_sliders(self):
        # Update internal values from all sliders
        # (No changes here—only referencing the updated color sliders)
        self.edge_color[0] = self.sliders[0].value
        self.edge_color[1] = self.sliders[1].value
        self.edge_color[2] = self.sliders[2].value

        self.cycle_color[0] = self.sliders[3].value
        self.cycle_color[1] = self.sliders[4].value
        self.cycle_color[2] = self.sliders[5].value

        self.node_color[0] = self.sliders[6].value
        self.node_color[1] = self.sliders[7].value
        self.node_color[2] = self.sliders[8].value

        self.edge_noise = self.sliders[9].value
        self.edge_curve = self.sliders[10].value

        new_thick = self.sliders[11].value
        if new_thick != self.edge_thickness:
            self.edge_thickness = new_thick

        new_iter = self.sliders[12].value
        new_cseed = self.sliders[13].value
        new_lseed = self.sliders[14].value

        changed = False
        if new_iter != self.numIterations:
            self.numIterations = new_iter
            changed = True
        if new_cseed != self.composite_seed:
            self.composite_seed = new_cseed
            changed = True
        if new_lseed != self.composite_length_seed:
            self.composite_length_seed = new_lseed
            changed = True

        new_show = self.checkboxes[0].value
        if new_show != self.draw_composite_nodes:
            self.draw_composite_nodes = int(new_show)

        new_dup = self.checkboxes[1].value
        if new_dup != self.use_duplicate_mode:
            self.use_duplicate_mode = int(new_dup)
            changed = True

        new_fill = self.checkboxes[2].value
        if new_fill != self.fill_cycles:
            self.fill_cycles = int(new_fill)


        new_post = self.sliders[15].value
        self.post_process_intensity = new_post

        new_splat = self.sliders[16].value
        self.splatters = new_splat

        if changed:
            self.build_composite_graph()

    def build_composite_graph(self):
        # Builds a composite graph by iteratively selecting candidate nodes and connecting them 
        # based on edges (slopes) from a primary graph. Two random generators (seeded by 
        # composite_seed and composite_length_seed) control candidate selection and edge lengths.
        
        self.base_cycles = detect_cycles_in_graph(self.graph)
        self.composite_nodes = []
        self.composite_edges = []
        self.compositeGraph = Graph()
        self.composite_cycles = []

        if not self.graph.nodes:
            return

        # set composite graph to duplicate of self.graph
        if self.use_duplicate_mode == 1:
            for i, (bx, by) in enumerate(self.graph.nodes):
                self.composite_nodes.append({
                    "pos": (bx, by),
                    "primary": i,
                    "available": []
                })
            for n1, nbrs in self.graph.adjacency_list.items():
                for n2 in nbrs:
                    if n2 > n1:
                        self.composite_edges.append((n1, n2))
            self.compositeGraph.nodes = [nd["pos"] for nd in self.composite_nodes]
            for i in range(len(self.composite_nodes)):
                self.compositeGraph.adjacency_list[i] = []
            for (i, j) in self.composite_edges:
                self.compositeGraph.adjacency_list[i].append(j)
                self.compositeGraph.adjacency_list[j].append(i)
            for i, nbrs in self.compositeGraph.adjacency_list.items():
                for j in nbrs:
                    dx = self.compositeGraph.nodes[j][0] - self.compositeGraph.nodes[i][0]
                    dy = self.compositeGraph.nodes[j][1] - self.compositeGraph.nodes[i][1]
                    sp = math.atan2(dy, dx)
                    self.compositeGraph.edge_slopes[(i, j)] = sp
                    self.compositeGraph.edge_slopes[(j, i)] = math.atan2(-dy, -dx)
            self.composite_cycles = detect_cycles_in_graph(self.compositeGraph)
            return

        # Create two random generators: one for candidate selection and one for determining edge lengths
        rng = random.Random(self.composite_seed)
        rng_len = random.Random(self.composite_length_seed)

        if 0 not in self.graph.adjacency_list:
            return

        available_angles = []
        for nbr in self.graph.adjacency_list[0]:
            angle = self.graph.edge_slopes.get((0, nbr))
            if angle is not None:
                available_angles.append(angle)

        # Start the composite graph using the primary node with index 0.
        # Set its position to (0,0), store the primary node index, and record available edge slopes from node 0
        self.composite_nodes.append({
            "pos": (0, 0),
            "primary": 0,
            "available": available_angles
        })
        # Initialize the frontier with tuples of (node index, slope) for each available edge from the starting node.
        frontier = [(0, a) for a in available_angles]

        # Iterate a fixed number of times to expand the composite graph.
        for _ in range(self.numIterations):
            new_frontier = [] # New frontier to store edges from newly added or merged nodes.

            # Process each edge in the current frontier.
            for (comp_i, slope_val) in frontier:

                # Calculate the required slope in the opposite direction (s + π), normalized to [0, 2π)
                req = (slope_val + math.pi) % (2*math.pi)
                candidates = []

                # Iterate over every candidate node in the primary graph
                for cid in range(len(self.graph.nodes)):
                    half_edges = []

                    # Gather all edge slopes (if defined) for the candidate node
                    for nbr in self.graph.adjacency_list.get(cid, []):
                        sp = self.graph.edge_slopes.get((cid, nbr))
                        if sp is not None:
                            half_edges.append(sp)

                    # Check each slope of the candidate node to see if it matches the required slope within a tolerance
                    for h in half_edges:
                        diff = abs((h - req + math.pi) % (2*math.pi) - math.pi)
                        if diff < self.composite_tolerance:
                            candidates.append((cid, h, half_edges))
                if not candidates:
                    continue

                # If matching candidate nodes were found:

                # randomly select one from list
                cand, used, half_ = rng.choice(candidates) 

                # Get the base position from the composite node from which we're expanding
                base_pos = self.composite_nodes[comp_i]["pos"]

                # Randomize the connection length based on a base length and a factor between 0.5 and 1.5
                length_ = self.connection_length * rng_len.uniform(0.5, 1.5)

                # Compute the new position using trigonometry, based on the slope
                new_pos = (
                    base_pos[0] + length_ * math.cos(slope_val),
                    base_pos[1] + length_ * math.sin(slope_val)
                )

                # Determine new available edge slopes for the new node,
                # filtering out the edge used for connection
                new_avail = []
                for x in half_:
                    diff = abs(((x - used + math.pi) % (2*math.pi)) - math.pi)
                    if diff >= self.composite_tolerance:
                        new_avail.append(x)

                # Create a candidate new node with its position, primary node, and available slopes
                candidate_node = {
                    "pos": new_pos,
                    "primary": cand,
                    "available": new_avail
                }

                # Check if there is an existing node close enough to merge with
                merge_idx = None
                for i, ex in enumerate(self.composite_nodes):
                    dx = ex["pos"][0] - new_pos[0]
                    dy = ex["pos"][1] - new_pos[1]
                    if math.hypot(dx, dy) < self.merge_threshold:
                        merge_idx = i
                        break
                if merge_idx is not None:

                    # Merge candidate_node into the existing node
                    # Unite (union) the available edges lists, avoiding duplicates
                    exist_set = set(self.composite_nodes[merge_idx]["available"])
                    new_set = set(candidate_node["available"])
                    self.composite_nodes[merge_idx]["available"] = list(exist_set.union(new_set))

                    # Record the edge between the current node and the merged node, if not already recorded
                    e = (comp_i, merge_idx)
                    if e not in self.composite_edges and (e[1], e[0]) not in self.composite_edges:
                        self.composite_edges.append(e)
                    
                    # Add new frontier entries for the merged node using its updated available slopes
                    for vv in candidate_node["available"]:
                        new_frontier.append((merge_idx, vv))
                else:
                    
                    # No merge candidate found: add the new node as usual
                    new_idx = len(self.composite_nodes)
                    self.composite_nodes.append(candidate_node)

                    # Record an edge connecting the current node to the new node
                    self.composite_edges.append((comp_i, new_idx))

                    # Add each available edge from the new node to the frontier for further expansion
                    for vv in candidate_node["available"]:
                        new_frontier.append((new_idx, vv))

            # Update the frontier with the new edges for the next iteration
            frontier = new_frontier

        # Build compositeGraph
        self.compositeGraph.nodes = [nd["pos"] for nd in self.composite_nodes]
        for i in range(len(self.composite_nodes)):
            self.compositeGraph.adjacency_list[i] = []
        for (i, j) in self.composite_edges:
            self.compositeGraph.adjacency_list[i].append(j)
            self.compositeGraph.adjacency_list[j].append(i)

        # compute slopes
        for i, nbrs in self.compositeGraph.adjacency_list.items():
            for j in nbrs:
                dx = self.compositeGraph.nodes[j][0] - self.compositeGraph.nodes[i][0]
                dy = self.compositeGraph.nodes[j][1] - self.compositeGraph.nodes[i][1]
                sp = math.atan2(dy, dx)
                self.compositeGraph.edge_slopes[(i, j)] = sp
                self.compositeGraph.edge_slopes[(j, i)] = math.atan2(-dy, -dx)
        self.composite_cycles = detect_cycles_in_graph(self.compositeGraph)

###############################
# Global Multi-Layer Setup
###############################
layers = []
active_layer_index = 0
columns = 4
max_layers = 8
tab_w = 75
tab_h = 30
x0 = 10
y0 = TOP_PANEL_HEIGHT + 90
gap_x = 5
gap_y = 5

def create_new_layer(name):
    layer = Layer(name)
    layers.append(layer)

def switch_to_layer(i):
    global active_layer_index
    active_layer_index = i
    if active_layer_index < 0:
        active_layer_index = 0
    if active_layer_index >= len(layers):
        active_layer_index = len(layers) - 1

def draw_layer_tabs(surface, mouse_pos):
    for i, layer in enumerate(layers):
        if i >= max_layers:
            break
        row = i // columns
        col = i % columns
        tab_x = x0 + col * (tab_w + gap_x)
        tab_y = y0 + row * (tab_h + gap_y)
        rect = pygame.Rect(tab_x, tab_y, tab_w, tab_h)
        color = (70, 70, 70) if i == active_layer_index else (50, 50, 50)
        pygame.draw.rect(surface, color, rect)
        pygame.draw.rect(surface, (120, 120, 120), rect, 2)
        txt_surf = font.render(layer.name, True, (255, 255, 255))
        surface.blit(txt_surf, (rect.x + 5, rect.y + 5))

def check_tab_click(mouse_pos):
    for i, layer in enumerate(layers):
        if i >= max_layers:
            break
        row = i // columns
        col = i % columns
        tab_x = x0 + col * (tab_w + gap_x)
        tab_y = y0 + row * (tab_h + gap_y)
        rect = pygame.Rect(tab_x, tab_y, tab_w, tab_h)
        if rect.collidepoint(mouse_pos):
            switch_to_layer(i)

###############################
# Initialize a few layers
###############################
create_new_layer("Layer 1")
create_new_layer("Layer 2")
create_new_layer("Layer 3")

###############################
# Main Loop
###############################
while running:
    events = pygame.event.get()
    mouse_pos = pygame.mouse.get_pos()
    mouse_click = False

    for event in events:
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_click = True
                check_tab_click(event.pos)
                main_rect = pygame.Rect(GUI_PANEL_WIDTH, TOP_PANEL_HEIGHT,
                                        MAIN_PANEL_WIDTH, HEIGHT - TOP_PANEL_HEIGHT)
                if main_rect.collidepoint(event.pos):
                    local_x = event.pos[0] - main_rect.x
                    local_y = event.pos[1] - main_rect.y
                    L = layers[active_layer_index]
                    n_idx = None
                    for i, (nx, ny) in enumerate(L.graph.nodes):
                        if math.dist((nx, ny), (local_x, local_y)) < NODE_SIZE:
                            n_idx = i
                            break
                    if n_idx is None:
                        gx, gy = snap_to_grid((local_x, local_y))
                        L.graph.add_node((gx, gy))
                        L.build_composite_graph()
                    else:
                        selected_node = n_idx

                right_rect = pygame.Rect(GUI_PANEL_WIDTH+MAIN_PANEL_WIDTH, TOP_PANEL_HEIGHT,
                                         RIGHT_PANEL_WIDTH, HEIGHT-TOP_PANEL_HEIGHT)
                if right_rect.collidepoint(event.pos):
                    right_panel_left_dragging = True
                    last_mouse_left = event.pos

            elif event.button == 2:
                right_rect = pygame.Rect(GUI_PANEL_WIDTH+MAIN_PANEL_WIDTH, TOP_PANEL_HEIGHT,
                                         RIGHT_PANEL_WIDTH, HEIGHT-TOP_PANEL_HEIGHT)
                if right_rect.collidepoint(event.pos):
                    right_panel_middle_dragging = True
                    last_mouse_middle = event.pos

            elif event.button == 3:
                right_rect = pygame.Rect(GUI_PANEL_WIDTH+MAIN_PANEL_WIDTH, TOP_PANEL_HEIGHT,
                                         RIGHT_PANEL_WIDTH, HEIGHT-TOP_PANEL_HEIGHT)
                if right_rect.collidepoint(event.pos):
                    right_panel_right_dragging = True
                    last_mouse_right = event.pos

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                right_panel_left_dragging = False
                main_rect = pygame.Rect(GUI_PANEL_WIDTH, TOP_PANEL_HEIGHT,
                                        MAIN_PANEL_WIDTH, HEIGHT-TOP_PANEL_HEIGHT)
                if main_rect.collidepoint(event.pos) and selected_node is not None:
                    L = layers[active_layer_index]
                    local_x = event.pos[0] - main_rect.x
                    local_y = event.pos[1] - main_rect.y
                    tgt = None
                    for i, (nx, ny) in enumerate(L.graph.nodes):
                        if math.dist((nx, ny), (local_x, local_y)) < NODE_SIZE:
                            tgt = i
                            break
                    if tgt is not None and tgt != selected_node:
                        L.graph.add_edge(selected_node, tgt)
                        L.build_composite_graph()
                    selected_node = None
            elif event.button == 2:
                right_panel_middle_dragging = False
            elif event.button == 3:
                right_panel_right_dragging = False

        elif event.type == pygame.MOUSEMOTION:
            if right_panel_middle_dragging:
                dx = event.pos[0] - last_mouse_middle[0]
                dy = event.pos[1] - last_mouse_middle[1]
                L = layers[active_layer_index]
                L.camera_offset[0] += dx
                L.camera_offset[1] += dy
                last_mouse_middle = event.pos
            if right_panel_left_dragging:
                dy = event.pos[1] - last_mouse_left[1]
                L = layers[active_layer_index]
                L.camera_pitch += dy * 0.005
                last_mouse_left = event.pos
            if right_panel_right_dragging:
                dx = event.pos[0] - last_mouse_right[0]
                L = layers[active_layer_index]
                L.camera_yaw += dx * 0.005
                last_mouse_right = event.pos

        elif event.type == pygame.MOUSEWHEEL:
            L = layers[active_layer_index]
            L.camera_zoom *= (1 + event.y * 0.1)

    if mouse_click:
        if clear_btn_rect.collidepoint(mouse_pos):
            L = layers[active_layer_index]
            L.graph = Graph()
            L.build_composite_graph()
        if save_btn_rect.collidepoint(mouse_pos):
            # --- Save the entire project as JSON ---
            project_data = {"active_layer_index": active_layer_index, "layers": []}
            for layer in layers:
                layer_data = {
                    "name": layer.name,
                    "graph": {
                        "nodes": layer.graph.nodes,
                        "adjacency_list": layer.graph.adjacency_list
                    },
                    "settings": {
                        "edge_color": layer.edge_color,
                        "cycle_color": layer.cycle_color,
                        "node_color": layer.node_color,
                        "edge_noise": layer.edge_noise,
                        "edge_curve": layer.edge_curve,
                        "edge_thickness": layer.edge_thickness,
                        "numIterations": layer.numIterations,
                        "composite_seed": layer.composite_seed,
                        "composite_length_seed": layer.composite_length_seed,
                        "composite_tolerance": layer.composite_tolerance,
                        "connection_length": layer.connection_length,
                        "merge_threshold": layer.merge_threshold,
                        "draw_composite_nodes": layer.draw_composite_nodes,
                        "use_duplicate_mode": layer.use_duplicate_mode,
                        "fill_cycles": layer.fill_cycles,
                        "post_process_intensity": layer.post_process_intensity,
                        "splatters": layer.splatters,
                        "camera_offset": layer.camera_offset,
                        "camera_zoom": layer.camera_zoom,
                        "camera_yaw": layer.camera_yaw,
                        "camera_pitch": layer.camera_pitch,
                    }
                }
                project_data["layers"].append(layer_data)
            fname = "jsons/project.json"
            with open(fname, "w") as f:
                json.dump(project_data, f, indent=4)
            print("Saved project JSON:", fname)

        if capture_btn_rect.collidepoint(mouse_pos):
            # Capture image of the right panel
            right_rect = pygame.Rect(
                GUI_PANEL_WIDTH + MAIN_PANEL_WIDTH,
                TOP_PANEL_HEIGHT,
                RIGHT_PANEL_WIDTH,
                HEIGHT - TOP_PANEL_HEIGHT
            )
            capture_fname = f"captures/{layers[active_layer_index].name}_capture.png"
            subsurf = screen.subsurface(right_rect)
            pygame.image.save(subsurf, capture_fname)
            print("Captured right panel image:", capture_fname)
        if load_json_btn_rect.collidepoint(mouse_pos):
            load_project("jsons/project.json")
        if new_layer_btn_rect.collidepoint(mouse_pos):
            nm = f"Layer {len(layers)+1}"
            create_new_layer(nm)

    layers[active_layer_index].process_sliders(events)
    layers[active_layer_index].update_from_sliders()

    screen.fill(BG_COLOR)
    top_rect = pygame.Rect(0, 0, WIDTH, TOP_PANEL_HEIGHT)
    pygame.draw.rect(screen, (50, 50, 50), top_rect)
    info_txt = (
        "Left Panel: GUI || "
        "Middle Panel: Graph Editor ||  "
        "Right Panel: Painting & Camera Manipulation || "
        "Ready to Load jsons/project.json || "
    )
    screen.blit(instr_font.render(info_txt, True, (230, 230, 230)), (10, 10))

    gui_rect = pygame.Rect(0, TOP_PANEL_HEIGHT, GUI_PANEL_WIDTH, HEIGHT - TOP_PANEL_HEIGHT)
    pygame.draw.rect(screen, (40, 40, 40), gui_rect)
    pygame.draw.rect(screen, (80, 80, 80), gui_rect, 2)

    # --- Drawing the Top GUI Rows ---
    
    # Row 1: Control buttons
    draw_button(screen, "Clear", clear_btn_rect, mouse_pos, font)
    draw_button(screen, "Capture", capture_btn_rect, mouse_pos, font)
    draw_button(screen, "New Layer", new_layer_btn_rect, mouse_pos, font)
    
    # Row 2: New buttons for capturing and loading JSON
    draw_button(screen, "Load JSON", load_json_btn_rect, mouse_pos, font)
    draw_button(screen, "Save JSON", save_btn_rect, mouse_pos, font)
    
    # Row 3: Layer tabs (adjusted to use new y-coordinate) 
    draw_layer_tabs(screen, mouse_pos)
    layers[active_layer_index].draw_sliders(screen, mouse_pos)

    main_panel_rect = pygame.Rect(GUI_PANEL_WIDTH, TOP_PANEL_HEIGHT, MAIN_PANEL_WIDTH, HEIGHT - TOP_PANEL_HEIGHT)
    pygame.draw.rect(screen, BG_COLOR, main_panel_rect)

    for gx in range(GUI_PANEL_WIDTH, GUI_PANEL_WIDTH+MAIN_PANEL_WIDTH, GRID_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (gx, TOP_PANEL_HEIGHT), (gx, HEIGHT), 1)
    for gy in range(TOP_PANEL_HEIGHT, HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (GUI_PANEL_WIDTH, gy), (GUI_PANEL_WIDTH+MAIN_PANEL_WIDTH, gy), 1)

    L = layers[active_layer_index]

    if L.fill_cycles == 1 and L.base_cycles:
        cycle_surf_main = pygame.Surface((MAIN_PANEL_WIDTH, HEIGHT - TOP_PANEL_HEIGHT), pygame.SRCALPHA)
        cycle_surf_main.fill((0, 0, 0, 0))
        for cyc in L.base_cycles:
            if len(cyc) < 3:
                continue
            pts = []
            for n_idx in cyc:
                bx, by = L.graph.nodes[n_idx]
                pts.append((bx, by))
            pygame.draw.polygon(cycle_surf_main, tuple(L.cycle_color), pts)
        screen.blit(cycle_surf_main, (GUI_PANEL_WIDTH, TOP_PANEL_HEIGHT))

    for n1 in L.graph.adjacency_list:
        for n2 in L.graph.adjacency_list[n1]:
            if n2 > n1:
                x1, y1 = L.graph.nodes[n1]
                x2, y2 = L.graph.nodes[n2]
                st = (x1 + GUI_PANEL_WIDTH, y1 + TOP_PANEL_HEIGHT)
                en = (x2 + GUI_PANEL_WIDTH, y2 + TOP_PANEL_HEIGHT)
                draw_jagged_or_curved_edge(screen, L.edge_color, st, en,
                                           L.edge_noise, L.edge_curve, L.edge_thickness)

    for i, (nx, ny) in enumerate(L.graph.nodes):
        sx = nx + GUI_PANEL_WIDTH
        sy = ny + TOP_PANEL_HEIGHT
        pygame.draw.circle(screen, L.node_color, (sx, sy), NODE_SIZE // 2)
        pygame.draw.circle(screen, L.node_color, (sx, sy), NODE_SIZE // 2, 1)

    # --- Right panel: Composite view ---
    right_rect = pygame.Rect(GUI_PANEL_WIDTH + MAIN_PANEL_WIDTH, TOP_PANEL_HEIGHT,
                             RIGHT_PANEL_WIDTH, HEIGHT - TOP_PANEL_HEIGHT)
    pygame.draw.rect(screen, BG_COLOR, right_rect)
    pygame.draw.rect(screen, (200, 200, 200), right_rect, 2)
    
    right_panel_surf = pygame.Surface((RIGHT_PANEL_WIDTH, right_rect.height), pygame.SRCALPHA)
    temp_rect = pygame.Rect(0, 0, RIGHT_PANEL_WIDTH, right_rect.height)
    
    post_group = pygame.Surface((RIGHT_PANEL_WIDTH, right_rect.height), pygame.SRCALPHA)
    top_group = pygame.Surface((RIGHT_PANEL_WIDTH, right_rect.height), pygame.SRCALPHA)
    
    # Determine overall bounding box for composite nodes (across all layers)
    all_xs, all_ys = [], []
    for (xx, yy) in layers[0].compositeGraph.nodes:
        all_xs.append(xx)
        all_ys.append(yy)
    if all_xs and all_ys:
        minx, maxx = min(all_xs), max(all_xs)
        miny, maxy = min(all_ys), max(all_ys)
        bw = maxx - minx or 1
        bh = maxy - miny or 1
    else:
        bw = bh = 1

    base_scale = min((RIGHT_PANEL_WIDTH - 20) / bw, (right_rect.height - 20) / bh)

    for i, ly in enumerate(layers):
        scale_3d = base_scale * ly.camera_zoom
        offset_x = ((RIGHT_PANEL_WIDTH - bw * scale_3d) * 0.5 - minx * scale_3d + ly.camera_offset[0])
        offset_y = ((right_rect.height - bh * scale_3d) * 0.5 - miny * scale_3d + ly.camera_offset[1])
        
        layer_surf = pygame.Surface((RIGHT_PANEL_WIDTH, right_rect.height), pygame.SRCALPHA)
        if ly.fill_cycles == 1 and len(ly.compositeGraph.nodes) > 1 and len(ly.composite_edges) > 0:
            fill_composite_cycles_with_intersections(ly, layer_surf, temp_rect, scale_3d, offset_x, offset_y)
    
        transformed = []
        for (xx, yy) in ly.compositeGraph.nodes:
            xx_ = xx + ly.comp_offset_x
            yy_ = yy + ly.comp_offset_y
            x_yaw = xx_ * math.cos(ly.camera_yaw)
            z_yaw = xx_ * math.sin(ly.camera_yaw)
            y_pitch = yy_ * math.cos(ly.camera_pitch) - z_yaw * math.sin(ly.camera_pitch)
            z_pitch = yy_ * math.sin(ly.camera_pitch) + z_yaw * math.cos(ly.camera_pitch)
            denom = perspective_distance - z_pitch
            if abs(denom) < 1e-6:
                denom = 1e-6
            pf = perspective_distance / denom
            x_eff = x_yaw * pf
            y_eff = y_pitch * pf
            transformed.append((x_eff, y_eff))
    
        for (i1, j) in ly.composite_edges:
            p1 = transformed[i1]
            p2 = transformed[j]
            st = (int(p1[0] * scale_3d + offset_x), int(p1[1] * scale_3d + offset_y))
            en = (int(p2[0] * scale_3d + offset_x), int(p2[1] * scale_3d + offset_y))
            draw_jagged_or_curved_edge(layer_surf, ly.edge_color, st, en,
                                       ly.edge_noise, ly.edge_curve, ly.edge_thickness)
    
        if ly.draw_composite_nodes == 1:
            for (xx_eff, yy_eff) in transformed:
                sx = int(xx_eff * scale_3d + offset_x)
                sy = int(yy_eff * scale_3d + offset_y)
                pygame.draw.circle(layer_surf, ly.node_color, (sx, sy), NODE_SIZE // 2)
                pygame.draw.circle(layer_surf, ly.node_color, (sx, sy), NODE_SIZE // 2, 1)

        # Apply paint splatters for this layer.
        layer_surf = apply_composite_paint_splatters(layer_surf, ly, base_scale,
                                                     minx, miny, bw, bh,
                                                     perspective_distance,
                                                     RIGHT_PANEL_WIDTH, right_rect.height)
    
        if i <= active_layer_index:
            post_group.blit(layer_surf, (0, 0))
        else:
            top_group.blit(layer_surf, (0, 0))

    right_panel_surf.blit(post_group, (0, 0))
    right_panel_surf.blit(top_group, (0, 0))

    # --- Apply painterly effect ---
    right_panel_surf = apply_painterly_effect(post_group, top_group,
                                              L.post_process_intensity,
                                              RIGHT_PANEL_WIDTH, right_rect.height,
                                              BG_COLOR)
    
    # --- Add paint splatters using colors from the active layer ---
    active_ly = layers[active_layer_index]
    right_panel_surf = apply_composite_paint_splatters(right_panel_surf, active_ly, base_scale,
                                                       minx, miny, bw, bh,
                                                       perspective_distance,
                                                       RIGHT_PANEL_WIDTH, right_rect.height)
    
    # Finally, blit composite view onto the screen.
    screen.blit(right_panel_surf, (right_rect.x, right_rect.y))
    pygame.display.flip()

pygame.quit()
sys.exit()