# PRM-based Pathfinding Algorithm for Obstacle-Avoidance
This project implements a Probabilistic Roadmap (PRM) for pathfinding in an environment with obstacles. It uses random sampling to create a network of nodes and connects them using the nearest neighbors algorithm. A shortest path between two points is then found using the A* search algorithm, avoiding obstacles along the way. The result is visualized on an image, showing both the graph network and the computed path.

## Features
Random Sampling: Random points are generated in a specified image (map) to represent valid points for navigation.
Obstacle Detection: Obstacles (shown as white areas in the image) are detected, and paths are only generated if they avoid these areas.
Graph Representation: A graph is created where each node is a point, and edges represent valid connections between neighboring points.
A Pathfinding*: The A* algorithm is used to find the shortest path between a start and goal point, taking into account obstacles.
Interactive Visualization: Users can click on the image to select start and goal points. The path and graph network are visualized on the image with the graph in red and the path in green.

## Requirements
 -Python 3.x
 -Required Python packages:
 -opencv-python (for image manipulation)
 -numpy (for numerical operations)
 -matplotlib (for visualization)
 -scikit-learn (for k-nearest neighbors)
 -heapq (for A* algorithm)


To install the required packages, you can use pip:
pip install opencv-python numpy matplotlib scikit-learn

## Usage
1. Clone the repository:
git clone https://github.com/Ishanned/PRM-Pathfinding.git

2. Prepare the map image:
Ensure you have a map image (e.g., new_map.png) in the project directory. The map should have obstacles represented as white areas, and valid paths should be black.

3. Run the algorithm:
Once the map is ready, run the script:
python prm_pathfinding.py

4. Select Start and Goal Points:
The program will display the map. Click on the image to select the start and goal points:
Left-click to select the start point.
Left-click again to select the goal point.

5. View the Results:
After selecting the start and goal points, the algorithm will find the shortest path between them, considering obstacles. The image will be displayed with:
Red lines representing the PRM graph network.
Green lines and markers representing the computed shortest path between the start and goal points.