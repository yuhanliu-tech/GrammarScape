import pygame

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
FONT = pygame.font.SysFont(None, 18)
INSTR_FONT = pygame.font.SysFont(None, 16)

PERSPECTIVE_DISTANCE = 300