### Search Algorithms Visualization

#### Overview
This project implements and visualizes search algorithms (DFS, BFS, and A*) for pathfinding in a grid-based environment using Python and Tkinter. The application provides a graphical interface to run and compare these algorithms.

### Features
* **Algorithms Implemented:**
  * Depth-First Search (DFS)
  * Breadth-First Search (BFS)
  * A* Search (with Manhattan heuristic)
* **Visualization:** The grid updates dynamically to display the path found by each algorithm.
* **Performance Metrics:** Execution time, memory usage, and node expansions are displayed.
* **User Interface:** Tkinter-based GUI with buttons to select and run different algorithms.

### Installation & Usage
1. **Install Dependencies:**
   ```sh
   pip install numpy matplotlib
   ```
2. **Run the Application:**
   ```sh
   python script.py
   ```

### How It Works
* The grid consists of:
  * 'S' (Start position)
  * 'G' (Goal position)
  * 'X' (Obstacles)
  * '.' (Walkable space)
* Algorithms traverse the grid and find the shortest path (if available).
* The path is displayed using Matplotlib within a Tkinter interface.

### License
This project is open-source and free to use.

