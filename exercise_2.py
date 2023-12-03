import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

edges = [
    (1, 2), (2, 3), (3, 4), (4, 5), (7, 8), (8, 9), (9, 10), (10, 11),
    (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11),
    (1, 7), (7, 1),
    (11, 5), (5, 11),
    (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (8, 8), (9, 9), (10, 10)
]

nodes = list(range(1, 12))
Graph = nx.DiGraph()
Graph.add_nodes_from(nodes)
Graph.add_edges_from(edges)

M = np.zeros((11, 11))  # Матрица М
p = np.zeros((11, 11))  # Матрица проверяет, есть ли ребро, идущее из вершины j в вершину i. 
m = np.zeros((11, 11))  # Матрица содержит общее количество стрелок, выходящих из вершины j.

for i in range(11):
    for j in range(11):
        m[i][j] = len(Graph.out_edges(j + 1))
        if Graph.has_edge(j + 1, i + 1):
            p[i][j] = 1
            if m[i][j] != 0:
                M[i][j] = 1 / m[i][j]
        else:
            p[i][j] = 0

fig = plt.figure(figsize=(10, 7))
nx.draw(Graph, node_color='red', with_labels=True)
# plt.show()

def Pagerank(M, epsilon, d):
    """
    параметр M:  матрица N строк N столбцов
    параметр epsilon: Чем он меньше, тем выше точность
    параметр d: отсутствие затухания
    
    return p: numpy.ndarray - вектор рейтинга N строк 1 столбец << чем выше рейтинг, тем выше >>
    """
    # Получить размер квадратной матрицы M
    N = len(M)

    # Инициализируйте p случайных начальных позиций и другой случайный вектор, чтобы контролировать сходимость вектора p.
    p = np.random.dirichlet(np.ones(N),size=1).reshape(N,1)
    last_p = np.random.dirichlet(np.ones(N),size=1).reshape(N,1)
    
    # Вычислить матрицу G
    G = (1 - d) / float(N)*np.ones((N, N), dtype=np.float32) + d*M

    # Повторять до схождения
    while np.linalg.norm(p - last_p, 2) > epsilon:
        last_p = p
        p = G @ p   #  << @: Скалярное умножение >>

    # Возвращает результат
    return p


# Показать результаты
p = Pagerank(M, 1, 1)
p = np.round(p, 4)
print(p)
