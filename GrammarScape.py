import pygame
import sys
import math
import json
import random
from PIL import Image
import cv2
import noise
import numpy as np

import graph
import ui_comps
import global_vars
import postprocess

pygame.init()

# Global Setup 

# Row 1: Clear, Save, New Layer
clear_btn_rect = pygame.Rect(10, global_vars.TOP_PANEL_HEIGHT+10, 90, 25)
capture_btn_rect = pygame.Rect(110, global_vars.TOP_PANEL_HEIGHT+10, 90, 25)
new_layer_btn_rect = pygame.Rect(210, global_vars.TOP_PANEL_HEIGHT+10, 120, 25)

# Row 2: Capture Right Panel and Load JSON
second_row_y = global_vars.TOP_PANEL_HEIGHT + 45
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
        layer = ui_comps.Layer(layer_data["name"])
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
        layer.blur_amount = settings["blur_amount"]
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
        layer.sliders[17].value = layer.blur_amount

        # Rebuild the composite graph now that the base graph is fully restored
        layer.build_composite_graph()
        layers.append(layer)
    active_layer_index = project_data.get("active_layer_index", 0)
    print("Project loaded successfully from", filename)

def snap_to_grid(pos):
    # Snaps a given position to the nearest grid intersection
    # The graph editor used to create the example graph leverages a grid 
    # We need to do this because the composite graph extends the graph based on edge slopes. 
    global_vars.GRID_SIZE = 20
    x, y = pos
    return (round(x / global_vars.GRID_SIZE) * global_vars.GRID_SIZE, round(y / global_vars.GRID_SIZE) * global_vars.GRID_SIZE)

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
# Global Multi-Layer Setup
###############################
layers = []
active_layer_index = 0
columns = 4
max_layers = 8
tab_w = 75
tab_h = 30
x0 = 10
y0 = global_vars.TOP_PANEL_HEIGHT + 90
gap_x = 5
gap_y = 5

def create_new_layer(name):
    layer = ui_comps.Layer(name)
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
        txt_surf = global_vars.FONT.render(layer.name, True, (255, 255, 255))
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
                main_rect = pygame.Rect(global_vars.GUI_PANEL_WIDTH, global_vars.TOP_PANEL_HEIGHT,
                                        global_vars.MAIN_PANEL_WIDTH, global_vars.HEIGHT - global_vars.TOP_PANEL_HEIGHT)
                if main_rect.collidepoint(event.pos):
                    local_x = event.pos[0] - main_rect.x
                    local_y = event.pos[1] - main_rect.y
                    L = layers[active_layer_index]
                    n_idx = None
                    for i, (nx, ny) in enumerate(L.graph.nodes):
                        if math.dist((nx, ny), (local_x, local_y)) < global_vars.NODE_SIZE:
                            n_idx = i
                            break
                    if n_idx is None:
                        gx, gy = snap_to_grid((local_x, local_y))
                        L.graph.add_node((gx, gy))
                        L.build_composite_graph()
                    else:
                        selected_node = n_idx

                right_rect = pygame.Rect(global_vars.GUI_PANEL_WIDTH + global_vars.MAIN_PANEL_WIDTH, global_vars.TOP_PANEL_HEIGHT,
                                         global_vars.RIGHT_PANEL_WIDTH, global_vars.HEIGHT - global_vars.TOP_PANEL_HEIGHT)
                if right_rect.collidepoint(event.pos):
                    right_panel_left_dragging = True
                    last_mouse_left = event.pos

            elif event.button == 2:
                right_rect = pygame.Rect(global_vars.GUI_PANEL_WIDTH + global_vars.MAIN_PANEL_WIDTH, global_vars.TOP_PANEL_HEIGHT,
                                         global_vars.RIGHT_PANEL_WIDTH, global_vars.HEIGHT - global_vars.TOP_PANEL_HEIGHT)
                if right_rect.collidepoint(event.pos):
                    right_panel_middle_dragging = True
                    last_mouse_middle = event.pos

            elif event.button == 3:
                right_rect = pygame.Rect(global_vars.GUI_PANEL_WIDTH + global_vars.MAIN_PANEL_WIDTH, global_vars.TOP_PANEL_HEIGHT,
                                         global_vars.RIGHT_PANEL_WIDTH, global_vars.HEIGHT - global_vars.TOP_PANEL_HEIGHT)
                if right_rect.collidepoint(event.pos):
                    right_panel_right_dragging = True
                    last_mouse_right = event.pos

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                right_panel_left_dragging = False
                main_rect = pygame.Rect(global_vars.GUI_PANEL_WIDTH, global_vars.TOP_PANEL_HEIGHT,
                                        global_vars.MAIN_PANEL_WIDTH, global_vars.HEIGHT - global_vars.TOP_PANEL_HEIGHT)
                if main_rect.collidepoint(event.pos) and selected_node is not None:
                    L = layers[active_layer_index]
                    local_x = event.pos[0] - main_rect.x
                    local_y = event.pos[1] - main_rect.y
                    tgt = None
                    for i, (nx, ny) in enumerate(L.graph.nodes):
                        if math.dist((nx, ny), (local_x, local_y)) < global_vars.NODE_SIZE:
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
            L.graph = graph.Graph()
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
                global_vars.GUI_PANEL_WIDTH + global_vars.MAIN_PANEL_WIDTH,
                global_vars.TOP_PANEL_HEIGHT,
                global_vars.RIGHT_PANEL_WIDTH,
                global_vars.HEIGHT - global_vars.TOP_PANEL_HEIGHT
            )
            capture_fname = f"captures/{layers[active_layer_index].name}_capture.png"
            subsurf = global_vars.screen.subsurface(right_rect)
            pygame.image.save(subsurf, capture_fname)
            print("Captured right panel image:", capture_fname)
        if load_json_btn_rect.collidepoint(mouse_pos):
            load_project("jsons/project.json")
        if new_layer_btn_rect.collidepoint(mouse_pos):
            nm = f"Layer {len(layers)+1}"
            create_new_layer(nm)

    layers[active_layer_index].process_sliders(events)
    layers[active_layer_index].update_from_sliders()

    global_vars.screen.fill(global_vars.BG_COLOR)
    top_rect = pygame.Rect(0, 0, global_vars.WIDTH, global_vars.TOP_PANEL_HEIGHT)
    pygame.draw.rect(global_vars.screen, (50, 50, 50), top_rect)
    info_txt = (
        "Left Panel: GUI || "
        "Middle Panel: Graph Editor ||  "
        "Right Panel: Painting & Camera Manipulation || "
        "Ready to Load jsons/project.json || "
    )
    global_vars.screen.blit(global_vars.INSTR_FONT.render(info_txt, True, (230, 230, 230)), (10, 10))

    gui_rect = pygame.Rect(0, global_vars.TOP_PANEL_HEIGHT, global_vars.GUI_PANEL_WIDTH, global_vars.HEIGHT - global_vars.TOP_PANEL_HEIGHT)
    pygame.draw.rect(global_vars.screen, (40, 40, 40), gui_rect)
    pygame.draw.rect(global_vars.screen, (80, 80, 80), gui_rect, 2)

    # --- Drawing the Top GUI Rows ---
    
    # Row 1: Control buttons
    draw_button(global_vars.screen, "Clear", clear_btn_rect, mouse_pos, global_vars.FONT)
    draw_button(global_vars.screen, "Capture", capture_btn_rect, mouse_pos, global_vars.FONT)
    draw_button(global_vars.screen, "New Layer", new_layer_btn_rect, mouse_pos, global_vars.FONT)
    
    # Row 2: New buttons for capturing and loading JSON
    draw_button(global_vars.screen, "Load JSON", load_json_btn_rect, mouse_pos, global_vars.FONT)
    draw_button(global_vars.screen, "Save JSON", save_btn_rect, mouse_pos, global_vars.FONT)
    
    # Row 3: Layer tabs (adjusted to use new y-coordinate) 
    draw_layer_tabs(global_vars.screen, mouse_pos)
    layers[active_layer_index].draw_sliders(global_vars.screen, mouse_pos)

    main_panel_rect = pygame.Rect(global_vars.GUI_PANEL_WIDTH, global_vars.TOP_PANEL_HEIGHT, global_vars.MAIN_PANEL_WIDTH, global_vars.HEIGHT - global_vars.TOP_PANEL_HEIGHT)
    pygame.draw.rect(global_vars.screen, global_vars.BG_COLOR, main_panel_rect)

    for gx in range(global_vars.GUI_PANEL_WIDTH, global_vars.GUI_PANEL_WIDTH+global_vars.MAIN_PANEL_WIDTH, global_vars.GRID_SIZE):
        pygame.draw.line(global_vars.screen, global_vars.GRID_COLOR, (gx, global_vars.TOP_PANEL_HEIGHT), (gx, global_vars.HEIGHT), 1)
    for gy in range(global_vars.TOP_PANEL_HEIGHT, global_vars.HEIGHT, global_vars.GRID_SIZE):
        pygame.draw.line(global_vars.screen, global_vars.GRID_COLOR, (global_vars.GUI_PANEL_WIDTH, gy), (global_vars.GUI_PANEL_WIDTH+global_vars.MAIN_PANEL_WIDTH, gy), 1)

    L = layers[active_layer_index]

    if L.fill_cycles == 1 and L.base_cycles:
        cycle_surf_main = pygame.Surface((global_vars.MAIN_PANEL_WIDTH, global_vars.HEIGHT - global_vars.TOP_PANEL_HEIGHT), pygame.SRCALPHA)
        cycle_surf_main.fill((0, 0, 0, 0))
        for cyc in L.base_cycles:
            if len(cyc) < 3:
                continue
            pts = []
            for n_idx in cyc:
                bx, by = L.graph.nodes[n_idx]
                pts.append((bx, by))
            pygame.draw.polygon(cycle_surf_main, tuple(L.cycle_color), pts)
        global_vars.screen.blit(cycle_surf_main, (global_vars.GUI_PANEL_WIDTH, global_vars.TOP_PANEL_HEIGHT))

    for n1 in L.graph.adjacency_list:
        for n2 in L.graph.adjacency_list[n1]:
            if n2 > n1:
                x1, y1 = L.graph.nodes[n1]
                x2, y2 = L.graph.nodes[n2]
                st = (x1 + global_vars.GUI_PANEL_WIDTH, y1 + global_vars.TOP_PANEL_HEIGHT)
                en = (x2 + global_vars.GUI_PANEL_WIDTH, y2 + global_vars.TOP_PANEL_HEIGHT)
                draw_jagged_or_curved_edge(global_vars.screen, L.edge_color, st, en,
                                           L.edge_noise, L.edge_curve, L.edge_thickness)

    for i, (nx, ny) in enumerate(L.graph.nodes):
        sx = nx + global_vars.GUI_PANEL_WIDTH
        sy = ny + global_vars.TOP_PANEL_HEIGHT
        pygame.draw.circle(global_vars.screen, L.node_color, (sx, sy), global_vars.NODE_SIZE // 2)
        pygame.draw.circle(global_vars.screen, L.node_color, (sx, sy), global_vars.NODE_SIZE // 2, 1)

    # --- Right panel: Composite view ---
    right_rect = pygame.Rect(global_vars.GUI_PANEL_WIDTH + global_vars.MAIN_PANEL_WIDTH, global_vars.TOP_PANEL_HEIGHT,
                             global_vars.RIGHT_PANEL_WIDTH, global_vars.HEIGHT - global_vars.TOP_PANEL_HEIGHT)
    pygame.draw.rect(global_vars.screen, global_vars.BG_COLOR, right_rect)
    pygame.draw.rect(global_vars.screen, (200, 200, 200), right_rect, 2)
    
    right_panel_surf = pygame.Surface((global_vars.RIGHT_PANEL_WIDTH, right_rect.height), pygame.SRCALPHA)
    temp_rect = pygame.Rect(0, 0, global_vars.RIGHT_PANEL_WIDTH, right_rect.height)
    
    post_group = pygame.Surface((global_vars.RIGHT_PANEL_WIDTH, right_rect.height), pygame.SRCALPHA)
    top_group = pygame.Surface((global_vars.RIGHT_PANEL_WIDTH, right_rect.height), pygame.SRCALPHA)
    
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

    base_scale = min((global_vars.RIGHT_PANEL_WIDTH - 20) / bw, (right_rect.height - 20) / bh)

    for i, ly in enumerate(layers):
        scale_3d = base_scale * ly.camera_zoom
        offset_x = ((global_vars.RIGHT_PANEL_WIDTH - bw * scale_3d) * 0.5 - minx * scale_3d + ly.camera_offset[0])
        offset_y = ((right_rect.height - bh * scale_3d) * 0.5 - miny * scale_3d + ly.camera_offset[1])
        
        layer_surf = pygame.Surface((global_vars.RIGHT_PANEL_WIDTH, right_rect.height), pygame.SRCALPHA)
        if ly.fill_cycles == 1 and len(ly.compositeGraph.nodes) > 1 and len(ly.composite_edges) > 0:
            graph.fill_composite_cycles_with_intersections(ly, layer_surf, temp_rect, scale_3d, offset_x, offset_y)
    
        transformed = []
        for (xx, yy) in ly.compositeGraph.nodes:
            xx_ = xx + ly.comp_offset_x
            yy_ = yy + ly.comp_offset_y
            x_yaw = xx_ * math.cos(ly.camera_yaw)
            z_yaw = xx_ * math.sin(ly.camera_yaw)
            y_pitch = yy_ * math.cos(ly.camera_pitch) - z_yaw * math.sin(ly.camera_pitch)
            z_pitch = yy_ * math.sin(ly.camera_pitch) + z_yaw * math.cos(ly.camera_pitch)
            denom = global_vars.PERSPECTIVE_DISTANCE - z_pitch
            if abs(denom) < 1e-6:
                denom = 1e-6
            pf = global_vars.PERSPECTIVE_DISTANCE / denom
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
                pygame.draw.circle(layer_surf, ly.node_color, (sx, sy), global_vars.NODE_SIZE // 2)
                pygame.draw.circle(layer_surf, ly.node_color, (sx, sy), global_vars.NODE_SIZE // 2, 1)

        # Apply paint splatters for this layer.
        layer_surf = postprocess.apply_composite_paint_splatters(layer_surf, ly, base_scale,
                                                     minx, miny, bw, bh,
                                                     global_vars.PERSPECTIVE_DISTANCE,
                                                     global_vars.RIGHT_PANEL_WIDTH, right_rect.height)
        
        if i <= active_layer_index and ly.blur_amount > 0:
            layer_surf = postprocess.gaussian_blur(layer_surf, ly.blur_amount)
    
        if i <= active_layer_index:
            post_group.blit(layer_surf, (0, 0))
        else:
            top_group.blit(layer_surf, (0, 0))

    right_panel_surf.blit(post_group, (0, 0))
    right_panel_surf.blit(top_group, (0, 0))

    # --- Apply painterly effect ---
    right_panel_surf = postprocess.apply_painterly_effect(post_group, top_group,
                                              L.post_process_intensity,
                                              global_vars.RIGHT_PANEL_WIDTH, right_rect.height,
                                              global_vars.BG_COLOR, canvas)
    
    # Finally, blit composite view onto the global_vars.screen.
    global_vars.screen.blit(right_panel_surf, (right_rect.x, right_rect.y))
    pygame.display.flip()

pygame.quit()
sys.exit()