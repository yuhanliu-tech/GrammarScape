import math
import numpy as np
import cv2
import pygame
import random

###############################
# Post-Process Effects
###############################

def apply_painterly_effect(post_group, top_group, post_process_intensity,
                           panel_width, panel_height, bg_color, canvas):
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
    draw_paint_splatters(surface, comp_transformed, palette, splatter_count=20,
                          splat_seed=splat_seed, splatter_value=active_layer.splatters)
    return surface

def draw_paint_splatters(surface, transformed_nodes, palette, splatter_count=20, splat_seed=None, splatter_value=5):
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

def gaussian_blur(surface, blur_amount):
    if blur_amount <= 0:
        return surface

    arr_rgb = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
    arr_alpha = pygame.surfarray.array_alpha(surface).T

    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)

    # Kernel size must be odd and > 1
    ksize = max(3, int(blur_amount * 3) * 2 + 1)

    # Apply blur to RGB and Alpha separately
    bgr_blurred = cv2.GaussianBlur(bgr, (ksize, ksize), sigmaX=0)
    alpha_blurred = cv2.GaussianBlur(arr_alpha, (ksize, ksize), sigmaX=0)

    # Convert back to RGB for Pygame
    rgb_blurred = cv2.cvtColor(bgr_blurred, cv2.COLOR_BGR2RGB)

    # Merge RGB and Alpha
    final_arr = np.dstack((rgb_blurred, alpha_blurred))

    # Convert to Pygame surface
    surf = pygame.image.frombuffer(final_arr.tobytes(), surface.get_size(), "RGBA")
    return surf.convert_alpha()
