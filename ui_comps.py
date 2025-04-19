import pygame
import math
import random

import graph
import global_vars

###############################
# Button Functions
###############################

def draw_button(surf, text, rect, mouse_pos, font):
    color = (80, 80, 80)
    hover = (120, 120, 120)
    if rect.collidepoint(mouse_pos):
        pygame.draw.rect(surf, hover, rect)
    else:
        pygame.draw.rect(surf, color, rect)
    txt_surf = font.render(text, True, (255, 255, 255))
    surf.blit(txt_surf, (rect.x+5, rect.y+5))


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
        self.graph = graph.Graph()

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
        self.compositeGraph = graph.Graph()
        self.composite_cycles = []

        # Per-layer camera transforms
        self.camera_offset = [0, 0]
        self.camera_zoom = 1.0
        self.camera_yaw = 0.0
        self.camera_pitch = 0.0

        # Post-process Paint Effects
        self.post_process_intensity = 0
        self.splatters = 0
        self.blur_amount = 0

        # Checkboxes ----------------------

        slider_x = 10
        slider_y = global_vars.TOP_PANEL_HEIGHT + 175
        slider_w = global_vars.GUI_PANEL_WIDTH - 25
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
        slider_y += slider_gap + 30

        # 4) Noise & Curve
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height, 0, 50,
                                   self.edge_noise, "Noise", True))
        slider_y += slider_gap
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height, 0, 100,
                                   self.edge_curve, "Curve", True))
        slider_y += slider_gap

        # Edge thickness
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height, 0, 30,
                                   self.edge_thickness, "Thickness", True))
        slider_y += slider_gap + 30

        # Iterations, seeds
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height, 1, 10,
                                   self.numIterations, "Iterations", True))
        slider_y += slider_gap
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height, 0, 1000,
                                   self.composite_seed, "Seed - Composite", True))
        slider_y += slider_gap
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height, 0, 1000,
                                   self.composite_length_seed, "Seed - Edge Lengths", True))
        slider_y += slider_gap + 30

        # Post Process slider: 0..10
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height,
                                   0, 10, 0, "Painterly", True))
        slider_y += slider_gap

        # Splatter size and frequency: 0..25
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height,
                                   0, 25, 0, "Splatters", True))
        slider_y += slider_gap

        # Blur effect slider: 0...10
        self.sliders.append(Slider(slider_x, slider_y, slider_w, s_height,
                           0, 10, 0, "Blur", True))
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
            cb.draw(surface, global_vars.FONT, mouse_pos)

        color_opts_label = global_vars.FONT.render("Graph Color Palette", True, (255, 255, 0))
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
        surface.blit(global_vars.FONT.render(edge_label, True, (255, 255, 255)), (ex, ey - 20))
        # Draw the 3 sliders with show_label=False
        for i in range(0, 3):
            self.sliders[i].draw(surface, mouse_pos, global_vars.FONT, show_label=False)

        # 2) Cycle color - sliders [3,4,5]
        cx, cy = self.sliders[3].rect.x, self.sliders[3].rect.y
        cycle_label = (
            f"Cycle (RGB: {int(self.cycle_color[0])}, "
            f"{int(self.cycle_color[1])}, {int(self.cycle_color[2])}"
            f")"
        )
        surface.blit(global_vars.FONT.render(cycle_label, True, (255, 255, 255)), (cx, cy - 20))
        for i in range(3, 6):
            self.sliders[i].draw(surface, mouse_pos, global_vars.FONT, show_label=False)

        # 3) Node color - sliders [6,7,8]
        nx, ny = self.sliders[6].rect.x, self.sliders[6].rect.y
        node_label = (
            f"Node (RGB: {int(self.node_color[0])}, "
            f"{int(self.node_color[1])}, {int(self.node_color[2])}"
            f")"
        )
        surface.blit(global_vars.FONT.render(node_label, True, (255, 255, 255)), (nx, ny - 20))
        for i in range(6, 9):
            self.sliders[i].draw(surface, mouse_pos, global_vars.FONT, show_label=False)

        # 4) The rest of the sliders are drawn normally with their existing labels

        # --- Edge Options Section ---
        edge_opts_label = global_vars.FONT.render("Edge Options", True, (255, 255, 0))
        edge_opts_pos = (self.sliders[9].rect.x, self.sliders[9].rect.y - 25)
        edge_opts_rect = edge_opts_label.get_rect(topleft=edge_opts_pos)
        surface.blit(edge_opts_label, edge_opts_pos)

        if edge_opts_rect.collidepoint(mouse_pos):
            tooltips.append("Noise adds randomness. \nCurve bends edges. \nThickness controls width.")

        for i in range(9, 12):
            self.sliders[i].draw(surface, mouse_pos, global_vars.FONT, show_label=True, inline=True)

        # --- Composite Graph Section ---
        comp_graph_label = global_vars.FONT.render("Composite Graph", True, (255, 255, 0))
        comp_graph_pos = (self.sliders[12].rect.x, self.sliders[12].rect.y - 25)
        comp_graph_rect = comp_graph_label.get_rect(topleft=comp_graph_pos)
        surface.blit(comp_graph_label, comp_graph_pos)

        if comp_graph_rect.collidepoint(mouse_pos):
            tooltips.append("Controls how the composite graph\nis generated and expanded.")

        for i in range(12, 15):
            self.sliders[i].draw(surface, mouse_pos, global_vars.FONT, show_label=True, inline=True)

        # --- Post-process Effects Section ---
        post_label = global_vars.FONT.render("Post-Process Effects", True, (255, 255, 0))
        post_label_pos = (self.sliders[15].rect.x, self.sliders[15].rect.y - 25)
        post_rect = post_label.get_rect(topleft=post_label_pos)
        surface.blit(post_label, post_label_pos)

        if post_rect.collidepoint(mouse_pos):
            tooltips.append("Painterly = oil paint effect. \nSplatters = paint droplets.\nBlur = resolution-based blend.")

        for i in range(15, len(self.sliders)):
            self.sliders[i].draw(surface, mouse_pos, global_vars.FONT, show_label=True, inline=True)

        # --- Show tooltip if hovering ---
        for i, tip in enumerate(tooltips):
            tip_surf = global_vars.FONT.render(tip, True, (255, 255, 255))
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

        self.blur_amount = self.sliders[17].value  

        if changed:
            self.build_composite_graph()

    def build_composite_graph(self):
        # Builds a composite graph by iteratively selecting candidate nodes and connecting them 
        # based on edges (slopes) from a primary graph. Two random generators (seeded by 
        # composite_seed and composite_length_seed) control candidate selection and edge lengths.
        
        self.base_cycles = graph.detect_cycles_in_graph(self.graph)
        self.composite_nodes = []
        self.composite_edges = []
        self.compositeGraph = graph.Graph()
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
            self.composite_cycles = graph.detect_cycles_in_graph(self.compositeGraph)
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
        self.composite_cycles = graph.detect_cycles_in_graph(self.compositeGraph)