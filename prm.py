import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import heapq

class PRM:
    def __init__(self, image_path, num_points=300, k=8):
        self.image = cv2.imread(image_path)
        self.num_points = num_points
        self.k = k
        self.point_list = []
        self.node_dict = {}
        self.start_point = None
        self.goal_point = None
        # self._create_obstacle()

    # def _create_obstacle(self):
    #     cv2.rectangle(self.image, pt1=(100, 120), pt2=(130, 150), color=(255, 255, 255), thickness=-1)

    def generate_random_points(self):
        i = 0
        while i < self.num_points:
            x = int(np.random.rand() * (self.image.shape[0] - 1))
            y = int(np.random.rand() * (self.image.shape[1] - 1))
            if np.all(self.image[x, y] == 0):
                self.image[x, y] = [0, 255, 255]
                point = (x, y)
                self.point_list.append(point)
                self.node_dict[point] = Node(x, y)
            i += 1

    def check_obstacles_in_line(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
        x_values = np.linspace(x1, x2, num_points, dtype=int)
        y_values = np.linspace(y1, y2, num_points, dtype=int)
        for x, y in zip(x_values, y_values):
            if np.all(self.image[x, y] == 255):
                return False
        return True

    def connect_neighbors(self):
        points = np.array(self.point_list)
        knn = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
        knn.fit(points)
        distances, indices = knn.kneighbors(points)

        for i, point in enumerate(self.point_list):
            nearest_neighbors_indices = indices[i]
            nearest_neighbors_points = points[nearest_neighbors_indices]

            for neighbor_point in nearest_neighbors_points:
                neighbor_point_tuple = (neighbor_point[0], neighbor_point[1])
                if self.check_obstacles_in_line(point, neighbor_point_tuple):
                    dist = np.linalg.norm(np.array(point) - np.array(neighbor_point_tuple))
                    self.node_dict[point].add_neighbor(self.node_dict[neighbor_point_tuple], dist)

    def draw_graph(self, ax):
        for point, node in self.node_dict.items():
            for neighbor, _ in zip(node.node_list, node.dist_list):
                ax.plot([point[1], neighbor.y], [point[0], neighbor.x], 'r-', alpha=0.5)

    def display_map(self, path=None):
        fig, ax = plt.subplots()
        ax.imshow(self.image)
        
        # Draw the PRM graph
        self.draw_graph(ax)

        # Draw the random points
        points = np.array(self.point_list)
        ax.scatter(points[:, 1], points[:, 0], c='b', marker='o', s=15)

        # Draw the shortest path if available
        if path:
            for i in range(len(path) - 1):
                node1 = path[i]
                node2 = path[i + 1]
                ax.plot([node1.y, node2.y], [node1.x, node2.x], 'g-', linewidth=2)
                ax.plot(node1.y, node1.x, 'go', markersize=5)
            ax.plot(path[-1].y, path[-1].x, 'go', markersize=5)

        plt.show()

    def heuristic(self, node, goal):
        return np.linalg.norm(np.array((node.x, node.y)) - np.array((goal.x, goal.y)))

    def a_star_search(self, start_point, goal_point):
        start_node = self.node_dict.get(start_point)
        goal_node = self.node_dict.get(goal_point)
        
        if not start_node or not goal_node:
            print("Start or goal point is not in the graph.")
            return []

        open_set = []
        heapq.heappush(open_set, (0, start_node))
        
        came_from = {}
        g_score = {node: float('inf') for node in self.node_dict.values()}
        g_score[start_node] = 0
        
        f_score = {node: float('inf') for node in self.node_dict.values()}
        f_score[start_node] = self.heuristic(start_node, goal_node)

        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal_node:
                return self.reconstruct_path(came_from, current)

            for neighbor, dist in zip(current.node_list, current.dist_list):
                tentative_g_score = g_score[current] + dist
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_node)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path

    def on_click(self, event):
        if event.xdata is not None and event.ydata is not None:
            point = (int(event.ydata), int(event.xdata))
            if self.start_point is None:
                self.start_point = point
                print(f"Start point selected: {self.start_point}")
            elif self.goal_point is None:
                self.goal_point = point
                print(f"Goal point selected: {self.goal_point}")
                plt.close()

    def find_nearest_point(self, point):
        points = np.array(self.point_list)
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(points)
        distance, index = knn.kneighbors([point])
        return tuple(points[index[0][0]])

    def add_connection_if_needed(self, point):
        if point not in self.node_dict:
            nearest_point = self.find_nearest_point(point)
            if self.check_obstacles_in_line(point, nearest_point):
                dist = np.linalg.norm(np.array(point) - np.array(nearest_point))
                self.node_dict[point] = Node(point[0], point[1])
                self.node_dict[point].add_neighbor(self.node_dict[nearest_point], dist)
                self.node_dict[nearest_point].add_neighbor(self.node_dict[point], dist)
                print(f"Connecting new point {point} to nearest point")

    def run(self):
        self.generate_random_points()
        self.connect_neighbors()

        # Capture user clicks for start and goal points
        fig, ax = plt.subplots()
        ax.imshow(self.image)
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

        if self.start_point and self.goal_point:
            self.add_connection_if_needed(self.start_point)
            self.add_connection_if_needed(self.goal_point)

            path = self.a_star_search(self.start_point, self.goal_point)
            if path:
                print("Path found!")
                self.display_map(path)
            else:
                print("No path found.")

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.node_list = []
        self.dist_list = []

    def add_neighbor(self, node, dist):
        self.node_list.append(node)
        self.dist_list.append(dist)

    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)

# Example usage
if __name__ == "__main__":
    prm = PRM(image_path='new_map.png', num_points=500, k=8)
    prm.run()
