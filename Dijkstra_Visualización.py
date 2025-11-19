import heapq            # Para la cola de prioridad (priority queue)
import networkx as nx   # Para manejar y dibujar grafos
import matplotlib.pyplot as plt  # Para la parte gráfica
import math

# ----------------------------------------
# Función auxiliar: imprimir tabla de distancias
# ----------------------------------------
def imprimir_tabla(distances, visited, previous):
    print("\nTabla actual de distancias:")
    print("{:<8} {:<12} {:<10} {:<10}".format("Nodo", "Distancia", "Visitado", "Previo"))
    print("-" * 45)
    for node in sorted(distances.keys()):
        dist = distances[node]
        dist_str = "∞" if dist == math.inf else f"{dist:.1f}"
        vis_str = "Sí" if node in visited else "No"
        prev_str = previous[node] if previous[node] is not None else "-"
        print("{:<8} {:<12} {:<10} {:<10}".format(node, dist_str, vis_str, prev_str))
    print("-" * 45)


# ----------------------------------------
# 1. Algoritmo de Dijkstra (NORMAL)
# ----------------------------------------
def dijkstra(graph, start):
    """
    Implementación del algoritmo de Dijkstra (versión normal).
    Regresa:
    - distances: diccionario con la distancia mínima desde start a cada nodo
    - previous: diccionario con el "anterior" de cada nodo en el camino óptimo
    """

    distances = {node: float("inf") for node in graph}
    distances[start] = 0

    previous = {node: None for node in graph}

    priority_queue = [(0, start)]

    while priority_queue:
        current_dist, current_node = heapq.heappop(priority_queue)

        if current_dist > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_dist + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances, previous


# ----------------------------------------
# 1B. Algoritmo de Dijkstra (SIMULADOR PASO A PASO)
# ----------------------------------------
def dijkstra_paso_a_paso(graph, start):
    """
    Misma lógica que dijkstra(), pero va imprimiendo
    paso a paso qué hace el algoritmo.
    """

    distances = {node: float("inf") for node in graph}
    distances[start] = 0

    previous = {node: None for node in graph}

    priority_queue = [(0, start)]
    visited = set()
    paso = 0

    print("\n=== INICIO DEL ALGORITMO DE DIJKSTRA ===")
    imprimir_tabla(distances, visited, previous)

    while priority_queue:
        paso += 1
        print(f"\n--- Paso {paso} ---")

        current_dist, current_node = heapq.heappop(priority_queue)

        # Si este nodo ya fue visitado con una mejor distancia, lo ignoramos
        if current_dist > distances[current_node]:
            print(f"Se extrae el nodo {current_node} con distancia {current_dist:.1f},")
            print("pero ya existe una mejor distancia registrada. Se ignora.")
            continue

        print(f"Se extrae el nodo {current_node} de la cola con distancia {current_dist:.1f}")
        visited.add(current_node)

        # Relajación de aristas
        for neighbor, weight in graph[current_node].items():
            print(f"  Vecino: {neighbor} (peso de arista = {weight})")
            distance = current_dist + weight
            print(f"    Distancia nueva propuesta hacia {neighbor}: {distance:.1f}")

            if distance < distances[neighbor]:
                print(f"    Mejora encontrada! Antes: {distances[neighbor]:.1f}, ahora: {distance:.1f}")
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
            else:
                print(f"    No mejora. Distancia actual hacia {neighbor} es {distances[neighbor]:.1f}")

        imprimir_tabla(distances, visited, previous)

    print("\n=== FIN DEL ALGORITMO DE DIJKSTRA ===")
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

    G = nx.Graph()

    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            if not G.has_edge(node, neighbor):
                G.add_edge(node, neighbor, weight=weight)

    pos = nx.spring_layout(G, seed=42)

    path_edges = set()
    if shortest_path is not None:
        for u, v in zip(shortest_path[:-1], shortest_path[1:]):
            path_edges.add((u, v))
            path_edges.add((v, u))

    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if (u, v) in path_edges:
            edge_colors.append("red")
            edge_widths.append(3)
        else:
            edge_colors.append("gray")
            edge_widths.append(1)

    if shortest_path is not None:
        path_nodes = set(shortest_path)
        node_colors = []
        for node in G.nodes():
            if node == shortest_path[0]:
                node_colors.append("lightgreen")
            elif node == shortest_path[-1]:
                node_colors.append("orange")
            elif node in path_nodes:
                node_colors.append("yellow")
            else:
                node_colors.append("lightblue")
    else:
        node_colors = "lightblue"

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

    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Grafo con camino más corto (Dijkstra)")
    plt.axis("off")
    plt.show()


# ----------------------------------------
# 4. Ejemplo de uso
# ----------------------------------------
if __name__ == "__main__":
    # Grafo de ejemplo
    graph = {
        "A": {"B": 3, "C": 3},
        "B": {"A": 3, "D": 3.5, "E": 2.8},
        "C": {"A": 3, "E": 2.8, "F": 3.5},
        "D": {"B": 3.5, "E": 3.1, "G": 10},
        "E": {"B": 2.8, "C": 2.8, "D": 3.1, "G": 7},
        "F": {"C": 3.5, "G": 2.5},
        "G": {"F": 2.5, "E": 7, "D": 10},
    }

    print("Nodos disponibles:", list(graph.keys()))
    origen = input("Ingresa el nodo ORIGEN: ").strip().upper()
    destino = input("Ingresa el nodo DESTINO: ").strip().upper()

    if origen not in graph or destino not in graph:
        print("Origen o destino no válidos.")
    else:
        # 1) Ejecutar Dijkstra con simulación paso a paso
        distances, previous = dijkstra_paso_a_paso(graph, origen)

        # 2) Reconstruir el camino más corto
        shortest_path = reconstruir_camino(previous, origen, destino)

        print(f"\nDistancias mínimas desde {origen}:")
        for nodo, dist in distances.items():
            print(f"  {origen} -> {nodo} = {dist}")

        if shortest_path is None:
            print(f"\nNo hay camino de {origen} a {destino}.")
        else:
            print(f"\nCamino más corto de {origen} a {destino}: {shortest_path}")
            print(f"Longitud total: {distances[destino]}")

            # 3) Dibujar grafo
            dibujar_grafo(graph, shortest_path)
