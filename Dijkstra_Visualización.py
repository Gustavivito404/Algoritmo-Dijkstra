import heapq            # Para la cola de prioridad (priority queue)
import networkx as nx   # Para manejar y dibujar grafos
import matplotlib.pyplot as plt  # Para la parte gráfica


# ----------------------------------------
# 1. Algoritmo de Dijkstra (distancias mínimas)
# ----------------------------------------
def dijkstra(graph, start):
    """
    Implementación del algoritmo de Dijkstra.
    - graph: diccionario de adyacencia, por ejemplo:
        {
            "A": {"B": 3, "C": 3},
            "B": {"A": 3, "D": 3.5}
        }
    - start: nodo origen (por ejemplo "B")

    Regresa:
    - distances: diccionario con la distancia mínima desde start a cada nodo
    - previous: diccionario con el "anterior" de cada nodo en el camino óptimo
    """

    # Inicializamos todas las distancias en infinito
    distances = {node: float("inf") for node in graph}
    # Y el nodo de origen en 0
    distances[start] = 0

    # Diccionario para reconstruir caminos
    previous = {node: None for node in graph}

    # Cola de prioridad: (distancia_actual, nodo)
    priority_queue = [(0, start)]

    while priority_queue:
        # Sacamos el nodo con menor distancia conocida
        current_dist, current_node = heapq.heappop(priority_queue)

        # Si esta distancia ya no es la mejor conocida, la ignoramos
        if current_dist > distances[current_node]:
            continue

        # Recorremos vecinos del nodo actual
        for neighbor, weight in graph[current_node].items():
            distance = current_dist + weight  # distancia acumulada

            # Si encontramos una mejor ruta hacia "neighbor"
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances, previous


# ----------------------------------------
# 2. Reconstruir el camino más corto
# ----------------------------------------
def reconstruir_camino(previous, start, goal):
    """
    Reconstruye el camino más corto desde 'start' hasta 'goal'
    usando el diccionario 'previous' que produjo Dijkstra.
    Regresa una lista de nodos en orden, o None si no hay camino.
    """
    if goal not in previous:
        return None

    path = []
    nodo = goal

    # Si el nodo destino no tiene anterior y no es el origen, no hay camino
    if previous[nodo] is None and nodo != start:
        return None

    while nodo is not None:
        path.append(nodo)
        if nodo == start:
            break
        nodo = previous[nodo]

    path.reverse()
    return path


# ----------------------------------------
# 3. Dibujar el grafo y resaltar el camino más corto
# ----------------------------------------
def dibujar_grafo(graph, shortest_path=None):
    """
    Dibuja el grafo usando networkx y matplotlib.
    - graph: diccionario de adyacencia
    - shortest_path: lista de nodos que forman el camino más corto (opcional)
    """

    # Creamos el grafo de networkx
    G = nx.Graph()

    # Agregamos nodos y aristas con pesos
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            # Evitar duplicar aristas en grafos no dirigidos
            if not G.has_edge(node, neighbor):
                G.add_edge(node, neighbor, weight=weight)

    # Layout (posición de los nodos en el dibujo)
    pos = nx.spring_layout(G, seed=42)  # seed para que siempre se dibuje parecido

    # Preparamos las aristas que forman parte del camino más corto
    path_edges = set()
    if shortest_path is not None:
        for u, v in zip(shortest_path[:-1], shortest_path[1:]):
            path_edges.add((u, v))
            path_edges.add((v, u))  # para grafo no dirigido

    # Colores y grosores de arista
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if (u, v) in path_edges:
            edge_colors.append("red")   # aristas del camino más corto
            edge_widths.append(3)
        else:
            edge_colors.append("gray")  # aristas normales
            edge_widths.append(1)

    # Colores de nodos
    if shortest_path is not None:
        path_nodes = set(shortest_path)
        node_colors = []
        for node in G.nodes():
            if node == shortest_path[0]:
                node_colors.append("lightgreen")  # origen
            elif node == shortest_path[-1]:
                node_colors.append("orange")      # destino
            elif node in path_nodes:
                node_colors.append("yellow")      # nodos intermedios del camino
            else:
                node_colors.append("lightblue")   # nodos normales
    else:
        node_colors = "lightblue"

    # Dibujar nodos y aristas
    plt.figure()
    nx.draw_networkx(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        edge_color=edge_colors,
        width=edge_widths,
        node_size=800,
        font_weight="bold",
    )

    # Dibujar etiquetas de pesos en las aristas
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Grafo con camino más corto (Dijkstra)")
    plt.axis("off")
    plt.show()


# ----------------------------------------
# 4. Ejemplo de uso
# ----------------------------------------
if __name__ == "__main__":
    # Grafo de ejemplo (similar al del tutorial)
    graph = {
        "A": {"B": 3, "C": 3},
        "B": {"A": 3, "D": 3.5, "E": 2.8},
        "C": {"A": 3, "E": 2.8, "F": 3.5},
        "D": {"B": 3.5, "E": 3.1, "G": 10},
        "E": {"B": 2.8, "C": 2.8, "D": 3.1, "G": 7},
        "F": {"C": 3.5, "G": 2.5},
        "G": {"F": 2.5, "E": 7, "D": 10},
    }

    origen = "B"
    destino = "F"

    # 1) Ejecutar Dijkstra
    distances, previous = dijkstra(graph, origen)

    # 2) Reconstruir el camino más corto
    shortest_path = reconstruir_camino(previous, origen, destino)

    print(f"Distancias mínimas desde {origen}:")
    for nodo, dist in distances.items():
        print(f"  {origen} -> {nodo} = {dist}")

    if shortest_path is None:
        print(f"No hay camino de {origen} a {destino}.")
    else:
        print(f"\nCamino más corto de {origen} a {destino}: {shortest_path}")
        print(f"Longitud total: {distances[destino]}")

    # 3) Dibujar el grafo y resaltar el camino
    dibujar_grafo(graph, shortest_path)
