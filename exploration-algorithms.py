import numpy as np
import matplotlib.pyplot as plt
from queue import Queue
import heapq
import time
import sys
import random
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Grid:
    def __init__(self, grid):
        # Initialize the grid, start, and goal positions
        self.grid = grid
        self.start = None
        self.goal = None
        self.parse_grid()

    def parse_grid(self):
        # Parse the grid to locate the start ('S') and goal ('G') positions
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                if cell == 'S':
                    self.start = (i, j)
                elif cell == 'G':
                    self.goal = (i, j)

    def is_valid(self, position):
        # Check if a position is within grid bounds and not an obstacle ('X')
        x, y = position
        return 0 <= x < len(self.grid) and 0 <= y < len(self.grid[0]) and self.grid[x][y] != 'X'


# Algorithm Implementations
class SearchAlgorithms:
    @staticmethod
    def dfs(grid):
        # Depth-First Search algorithm
        start, goal = grid.start, grid.goal
        stack = [(start, [start])]
        visited = set()
        node_expansion_count = 0

        while stack:
            current, path = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            node_expansion_count += 1
            if current == goal:
                return path, node_expansion_count
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if grid.is_valid(neighbor) and neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
        return [], node_expansion_count

    @staticmethod
    def bfs(grid):
        # Breadth-First Search algorithm
        start, goal = grid.start, grid.goal
        queue = Queue()
        queue.put((start, [start]))
        visited = set()
        node_expansion_count = 0

        while not queue.empty():
            current, path = queue.get()
            if current in visited:
                continue
            visited.add(current)
            node_expansion_count += 1
            if current == goal:
                return path, node_expansion_count
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if grid.is_valid(neighbor) and neighbor not in visited:
                    queue.put((neighbor, path + [neighbor]))
        return [], node_expansion_count

    @staticmethod
    def a_star(grid, heuristic):
        # A* Search algorithm with a heuristic function
        start, goal = grid.start, grid.goal
        pq = []
        heapq.heappush(pq, (0, start, [start]))
        visited = set()
        node_expansion_count = 0

        while pq:
            cost, current, path = heapq.heappop(pq)
            if current in visited:
                continue
            visited.add(current)
            node_expansion_count += 1
            if current == goal:
                return path, node_expansion_count
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if grid.is_valid(neighbor) and neighbor not in visited:
                    g_cost = len(path)
                    h_cost = heuristic(neighbor, goal)
                    heapq.heappush(pq, (g_cost + h_cost, neighbor, path + [neighbor]))
        return [], node_expansion_count

# Heuristics
def manhattan_heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean_heuristic(a, b):
    # Euclidean distance
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

# Visualization
def visualize_path(grid, path, title):
    grid_data = np.array(grid.grid)
    for x, y in path:
        if grid_data[x][y] not in ['S', 'G']:
            grid_data[x][y] = '*'
    
    fig, ax = plt.subplots()
    ax.imshow(grid_data == 'X', cmap='binary', interpolation='none')

    for i in range(len(grid_data)):
        for j in range(len(grid_data[0])):
            if grid_data[i][j] == 'S':
                ax.text(j, i, 'S', ha='center', va='center', color='blue', fontweight='bold')  
            elif grid_data[i][j] == 'G':
                ax.text(j, i, 'G', ha='center', va='center', color='green', fontweight='bold')  
            elif grid_data[i][j] == '*':
                ax.text(j, i, '*', ha='center', va='center', color='red', fontweight='bold')  
            elif grid_data[i][j] == '.':
                ax.text(j, i, '.', ha='center', va='center', color='darkgray')  

    fig.suptitle(title, fontsize=14)
    return fig


# Performance Analysis
def analyze_performance(grid, algorithm, heuristic=None):
    start_time = time.time()
    if heuristic:
        path, nodes_expanded = algorithm(grid, heuristic)
    else:
        path, nodes_expanded = algorithm(grid)
    end_time = time.time()
    runtime = end_time - start_time
    memory_usage = sys.getsizeof(path) + sys.getsizeof(nodes_expanded)
    return path, runtime, memory_usage, nodes_expanded


def main():
    example_grid = [
        ['S', '.', '.', 'X', '.'],
        ['.', 'X', '.', '.', '.'],
        ['.', 'X', 'X', 'X', '.'],
        ['.', '.', '.', 'G', '.']
    ]

    grid = Grid(example_grid)

    def run_algorithm(algorithm, heuristic=None):
        if algorithm == "DFS":
            path, runtime, memory, nodes = analyze_performance(grid, SearchAlgorithms.dfs)
            title = f"DFS Path (Time: {runtime:.4f}s, Nodes: {nodes}, Memory: {memory} bytes)"
        elif algorithm == "BFS":
            path, runtime, memory, nodes = analyze_performance(grid, SearchAlgorithms.bfs)
            title = f"BFS Path (Time: {runtime:.4f}s, Nodes: {nodes}, Memory: {memory} bytes)"
        elif algorithm == "A*":
            path, runtime, memory, nodes = analyze_performance(grid, SearchAlgorithms.a_star, manhattan_heuristic)
            title = f"A* Path (Time: {runtime:.4f}s, Nodes: {nodes}, Memory: {memory} bytes)"
        else:
            return

        fig = visualize_path(grid, path, title)
        for widget in frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    # GUI Setup
    root = tk.Tk()
    root.title("Search Algorithms Visualization")

    root.configure(bg="white")

    frame = tk.Frame(root, bg="white")
    frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    btn_dfs = tk.Button(root, text="Run DFS", command=lambda: run_algorithm("DFS"), font=("Arial", 14), width=8, height=1, bg="light gray")
    btn_bfs = tk.Button(root, text="Run BFS", command=lambda: run_algorithm("BFS"), font=("Arial", 14), width=8, height=1, bg="light gray")
    btn_astar = tk.Button(root, text="Run A*", command=lambda: run_algorithm("A*"), font=("Arial", 14), width=8, height=1, bg="light gray")

    btn_dfs.pack(side=tk.LEFT, padx=75, pady=10)
    btn_bfs.pack(side=tk.LEFT, padx=75, pady=10)
    btn_astar.pack(side=tk.LEFT, padx=75, pady=10)

    run_algorithm("DFS")

    root.mainloop()

if __name__ == "__main__":
    main()
