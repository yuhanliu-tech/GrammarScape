from google import genai
import pygame
import concurrent.futures

import global_vars
import graph

# DO NOT COMMIT API KEY
client = "INSERT HERE"

###############################
# Text Box Setup for Graph Description
###############################
# Position the text box at the top of the middle panel.
# Adjust coordinates as needed. 
text_box_rect = pygame.Rect(global_vars.GUI_PANEL_WIDTH + 10, global_vars.TOP_PANEL_HEIGHT + 10, 300, 30)
text_box_active = False

def parse_graph_string(graph_str, graph_obj):

    tokens = []  # This list will store tokens which are either numbers (as strings) or parenthesized pairs.
    i = 0
    n = len(graph_str)
    
    # Iterate over the string to extract tokens.
    while i < n:
        # Skip any whitespace or commas.
        if graph_str[i].isspace() or graph_str[i] == ',':
            i += 1
            continue
        if graph_str[i] == '(':
            # Capture everything until the corresponding ')'
            i += 1  # Skip the '(' character.
            start = i
            while i < n and graph_str[i] != ')':
                i += 1
            token = graph_str[start:i]
            tokens.append("(" + token + ")")
            i += 1  # Skip the closing ')'
        else:
            # Capture a number, allowing for negative numbers.
            start = i
            # Advance while the character is a digit or a minus sign.
            while i < n and (graph_str[i].isdigit() or graph_str[i] == '-'):
                i += 1
            token = graph_str[start:i]
            tokens.append(token)

    # Expecting tokens structure:
    # tokens[0]      : number of nodes (string representation)
    # tokens[1 .. node_count]   : node coordinate pairs, e.g., "(n1, n2)"
    # tokens[node_count+1] : number of edges
    # tokens[node_count+2 .. end] : edge pairs, e.g., "(e1, e2)"
    
    # Parse the node count.
    node_count = int(tokens[0])

    if node_count > 20: 
        print("Graph is too complex, aborting.")
    else:
    
        # Parse the edge count. It is the token right after the node tokens.
        edge_count = int(tokens[node_count + 1])

        if len(tokens) - (node_count + 2) != edge_count:
            raise ValueError("Edge token count does not match the parsed edge count.")
        
        # Process node tokens.
        for token in tokens[1:1 + node_count]:
            # Remove the enclosing parentheses.
            inner = token[1:-1]
            parts = inner.split(',')
            x = int(parts[0].strip())
            y = int(parts[1].strip())
            graph_obj.graph.add_node((x, y))
        
        # Process edge tokens.
        edge_tokens = tokens[1 + node_count + 1:]
        for token in edge_tokens:
            inner = token[1:-1]
            parts = inner.split(',')
            src = int(parts[0].strip())
            dest = int(parts[1].strip())
            graph_obj.graph.add_edge(src, dest)

def cut_before_integer(s):
    for index, char in enumerate(s):
        if char.isdigit():
            return s[index:]
    return s

def generate_and_draw_graph(description, layers, active_layer_index):
    
    L = layers[active_layer_index]
    # Reset the graph on the active layer
    L.graph = graph.Graph()
    
    # Example logic: create a specific pattern based on keywords
    desc = description.lower()

    bot_message = "Draw an image of a " + desc + " as a graph in the form of nodes and edges. Format your response in the following way: numNodes, (n1, n2), (n3, n4), numEdges, (e1, e2), (e3, e4) where all values are integers. Do not include anything else in your response as it will be used as a string input to code. For example, to draw a triangle, return: 3, (100,100), (150,50), (200,100), 3, (0,1), (1,2), (2,0)"
    timeout = 10

    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=bot_message
    )

    input_str = cut_before_integer(response.text)

    # Run the parsing function in a separate thread with a timeout.
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_parse = executor.submit(parse_graph_string, input_str, L)
            # Wait for parse_graph_string() to complete with a timeout.
            future_parse.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        print("Error: Parsing the graph string took too long. Aborting function.")
        return
    except Exception as e:
        print("An error occurred while parsing the graph string:", e)
        return

    '''
    if "triangle" in desc:
        # Create a triangular graph
        L.graph.add_node((100, 100))
        L.graph.add_node((150, 50))
        L.graph.add_node((200, 100))
        L.graph.add_edge(0, 1)
        L.graph.add_edge(1, 2)
        L.graph.add_edge(2, 0)
    elif "square" in desc:
        # Create a square graph
        L.graph.add_node((100, 100))
        L.graph.add_node((100, 200))
        L.graph.add_node((200, 200))
        L.graph.add_node((200, 100))
        L.graph.add_edge(0, 1)
        L.graph.add_edge(1, 2)
        L.graph.add_edge(2, 3)
        L.graph.add_edge(3, 0)
    '''
        
    # Rebuild composite graph after generating new graph.
    L.build_composite_graph()