import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp 
from sklearn.cluster import KMeans
import numpy as np 

# Набор ребер: (a, b) - a --> b 
edges = [
        (1, 2), (1, 3), (1, 5), 
        (2, 1), (2, 3), (2, 5), (2, 4),
        (3, 1), (3, 2), (3, 4), (3, 21),
        (4, 2), (4, 3), (4, 5), (4, 11), 
        (5, 1), (5, 2), (5, 4), 
        (6, 7), (6, 8), (6, 10), 
        (7, 6), (7, 8), (7, 9), (7, 10), 
        (8, 6), (8, 7), (8, 9),
        (9, 7), (9, 8), (9, 10), (9, 16),
        (10, 6), (10, 7), (10, 9), (10, 11),
        (11, 4), (11, 10), (11, 13), (11, 14), (11, 15),
        (12, 13), (12, 14), (12, 15), 
        (13, 12), (13, 14), (13, 11),
        (14, 13), (14, 15), (14, 11), (14, 12),
        (15, 11), (15, 14), (15, 12),
        (16, 9), (16, 17), (16, 18), (16, 20),
        (17, 16), (17, 18), (17, 19), (17, 20),
        (18, 16), (18, 17), (18, 19), 
        (19, 17), (19, 18), (19, 20), (19, 23),
        (20, 16), (20, 17), (20, 19), 
        (21, 3), (21, 22), (21, 24), (21, 25),
        (22, 21), (22, 23), (22, 25),
        (23, 19), (23, 22), (23, 24), (23, 25),
        (24, 21), (24, 23), (24, 25), 
        (25, 21), (25, 22), (25, 23), (25, 24)
        ]

nodes = np.arange(1, 26)      # Набор вершин от 1 до 25
Graph = nx.DiGraph()

Graph.add_nodes_from(nodes)   # Добавляем вершины в граф
Graph.add_edges_from(edges)   # Добавляем ребра в граф 
 
colors = ('yellow', 'lightgreen', 'cyan', 'orange', 'violet')   # Набор цветов

# Функция кластеризации и графическое представление результатов
def Draw(k, colors):
    
    G = nx.Graph()             # Создайте один неориентированный граф
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    L = nx.laplacian_matrix(G).astype(float)      # матрица Лапласиан

    # Найдите собственные векторы и собственные значения
    eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(L, k, which='SM')

    # Кластеризация
    data = eigenvectors*eigenvalues
    kmeans = KMeans(init='k-means++', n_clusters=k, n_init=5)
    kmeans.fit_predict(data)

    clusters = kmeans.labels_   # Массив содержит метки данных после разделения на k кластеров.  
        
    colors = colors[:k]         # Массив содержит k цветов, соответствующих k кластеризации.
    node_colors = [colors[cluster] for cluster in clusters]   # Назначьте цвета точкам, соответствующим каждому кластеру

    Graph_2 = nx.DiGraph(G)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 9))
    plt.sca(axes[0])
    nx.draw(Graph, with_labels='True', node_color='red')                     # Нарисуйте исходный график

    plt.sca(axes[1])
    nx.draw(Graph_2, node_color=node_colors, with_labels=True, arrows=True)  # Нарисуйте график после кластеризации
    # Show graph 
    plt.show()

# Draw(2, colors)  # Для k = 2 
# Draw(3, colors)  # Для k = 3
# Draw(4, colors)  # Для k = 4
# Draw(5, colors)  # Для k = 5


