import numpy as np
import heapq
from collections import defaultdict

# Run A* algorithm to find the path from start to goal
def run_astar(map_data, start, goal, occupied_positions=None):
    """
    A* search algorithm to find optimal path from start to goal
    
    Args:
        map_data: 2D grid representation of the environment
        start: Starting coordinates (x, y)
        goal: Goal coordinates (x, y)
        occupied_positions: Set of positions occupied by other robots to avoid
        
    Returns:
        action: Next action to take ('U', 'D', 'L', 'R', 'S')
        distance: Distance to goal
    """
    n_rows = len(map_data)
    n_cols = len(map_data[0])
    
    if occupied_positions is None:
        occupied_positions = set()
    
    # Heuristic function (Manhattan distance)
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    # A* algorithm
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, []))
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    open_set_hash = {start}
    
    while open_set:
        _, _, current, path = heapq.heappop(open_set)
        open_set_hash.remove(current)
        
        if current == goal:
            if not path:
                return 'S', 0
            first_step = path[0]
            dx, dy = first_step[0] - start[0], first_step[1] - start[1]
            
            # Convert direction to action
            if dx == -1 and dy == 0:
                return 'U', len(path)
            elif dx == 1 and dy == 0:
                return 'D', len(path)
            elif dx == 0 and dy == -1:
                return 'L', len(path)
            elif dx == 0 and dy == 1:
                return 'R', len(path)
            return 'S', len(path)
        
        # Check neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Check if the neighbor is valid
            if (neighbor[0] < 0 or neighbor[0] >= n_rows or 
                neighbor[1] < 0 or neighbor[1] >= n_cols or
                map_data[neighbor[0]][neighbor[1]] == 1 or
                neighbor in occupied_positions):
                continue
            
            tentative_g_score = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                
                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g_score, neighbor, path + [neighbor]))
                    open_set_hash.add(neighbor)
    
    # If no path is found, stay in place
    return 'S', float('inf')

def calculate_score(map_data, robot_pos, package, deadline):
    """
    Calculate utility score for a robot to handle a package
    
    Args:
        map_data: Map layout
        robot_pos: Robot position (x, y)
        package: Package information (id, start_x, start_y, end_x, end_y, deadline)
        deadline: Package deadline
        
    Returns:
        score: Utility score (higher is better)
    """
    # Calculate distances
    start_pos = (package[1], package[2])
    end_pos = (package[3], package[4])
    
    # A* to find distances
    _, pickup_dist = run_astar(map_data, robot_pos, start_pos)
    _, delivery_dist = run_astar(map_data, start_pos, end_pos)
    
    total_dist = pickup_dist + delivery_dist
    time_remaining = deadline - package[5]
    urgency = 1.0
    
    # If package is likely to be delivered late, reduce its score
    if total_dist > time_remaining:
        urgency = 0.1
    
    # Calculate score (lower distance and higher urgency is better)
    score = (1000.0 / (total_dist + 1)) * urgency
    
    return score

class Agents:
    def __init__(self):
        self.agents = []
        self.packages = []
        self.packages_free = []
        self.n_robots = 0
        self.state = None
        self.current_time = 0
        self.robot_paths = {}
        self.robot_future_positions = {}
        self.reserved_positions = set()
        self.package_priorities = {}
        self.is_init = False

    def init_agents(self, state):
        """Initialize agent state from environment state"""
        self.state = state
        self.current_time = 0
        self.n_robots = len(state['robots'])
        self.map = state['map']
        self.robots = [(robot[0]-1, robot[1]-1, robot[2]) for robot in state['robots']]
        self.robots_target = ['free'] * self.n_robots
        self.robot_paths = {i: [] for i in range(self.n_robots)}
        self.robot_future_positions = {i: set() for i in range(self.n_robots)}
        self.reserved_positions = set()
        
        # Track all packages
        self.packages = [(p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5], self.current_time) for p in state['packages']]
        self.packages_free = [True] * len(self.packages)
        
        # Calculate package priorities based on urgency and distance
        self.update_package_priorities()

    def update_package_priorities(self):
        """Update package priorities based on urgency and distances"""
        self.package_priorities = {}
        
        for i, pkg in enumerate(self.packages):
            if not self.packages_free[i]:
                continue
                
            pkg_id, start_x, start_y, end_x, end_y, deadline, appeared_time = pkg
            time_to_deadline = deadline - self.current_time
            
            # Estimate pickup and delivery distances
            pickup_dist = float('inf')
            for robot_id in range(self.n_robots):
                if self.robots_target[robot_id] == 'free':
                    robot_pos = (self.robots[robot_id][0], self.robots[robot_id][1])
                    _, dist = run_astar(self.map, robot_pos, (start_x, start_y))
                    pickup_dist = min(pickup_dist, dist)
            
            # Estimate delivery distance
            _, delivery_dist = run_astar(self.map, (start_x, start_y), (end_x, end_y))
            
            total_est_time = pickup_dist + delivery_dist
            
            # Calculate urgency factor
            urgency = 10.0
            if time_to_deadline < total_est_time:
                # Package likely to be late, adjust priority
                urgency = 5.0
            elif time_to_deadline < 2 * total_est_time:
                # Package getting close to deadline, increase priority
                urgency = 20.0
            
            # Combine factors into priority score
            priority = urgency * (1.0 / (total_est_time + 1))
            
            self.package_priorities[pkg_id] = priority

    def update_move_to_target(self, robot_id, target_package_id, phase='start'):
        """Calculate move to target for a specific robot and package"""
        if phase == 'start':
            target_pos = (self.packages[target_package_id][1], self.packages[target_package_id][2])
        else:
            target_pos = (self.packages[target_package_id][3], self.packages[target_package_id][4])
        
        robot_pos = (self.robots[robot_id][0], self.robots[robot_id][1])
        
        # Collect occupied positions from other robots (excluding current robot)
        occupied_positions = set()
        for r_id, positions in self.robot_future_positions.items():
            if r_id != robot_id:
                occupied_positions.update(positions)
        
        # Use A* to find optimal path considering other robots
        move, distance = run_astar(self.map, robot_pos, target_pos, occupied_positions)
        
        # Determine package action
        pkg_act = 0
        if distance == 0:
            if phase == 'start':
                pkg_act = 1  # Pickup
            else:
                pkg_act = 2  # Drop
        
        # Update future positions
        if move != 'S':
            next_pos = robot_pos
            if move == 'U':
                next_pos = (robot_pos[0] - 1, robot_pos[1])
            elif move == 'D':
                next_pos = (robot_pos[0] + 1, robot_pos[1])
            elif move == 'L':
                next_pos = (robot_pos[0], robot_pos[1] - 1)
            elif move == 'R':
                next_pos = (robot_pos[0], robot_pos[1] + 1)
            
            self.robot_future_positions[robot_id].add(next_pos)
        
        return move, str(pkg_act)
    
    def update_inner_state(self, state):
        """Update agent's internal state based on environment state"""
        self.current_time += 1
        
        # Reset future positions
        self.robot_future_positions = {i: set() for i in range(self.n_robots)}
        
        # Update robot positions and states
        for i in range(len(state['robots'])):
            prev = (self.robots[i][0], self.robots[i][1], self.robots[i][2])
            robot = state['robots'][i]
            self.robots[i] = (robot[0]-1, robot[1]-1, robot[2])
            
            if prev[2] != 0 and self.robots[i][2] == 0:
                # Robot has dropped the package
                self.robots_target[i] = 'free'
            elif self.robots[i][2] != 0:
                # Robot is carrying a package
                self.robots_target[i] = self.robots[i][2]
        
        # Add new packages
        old_package_count = len(self.packages)
        self.packages += [(p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5], self.current_time) for p in state['packages']]
        self.packages_free += [True] * len(state['packages'])
        
        # Update package priorities
        self.update_package_priorities()

    def assign_packages_to_robots(self):
        """Assign packages to available robots using a priority-based approach"""
        available_robots = [i for i in range(self.n_robots) if self.robots_target[i] == 'free']
        unassigned_packages = [i for i in range(len(self.packages)) if self.packages_free[i]]
        
        # Sort packages by priority
        sorted_packages = sorted(unassigned_packages, 
                                key=lambda i: self.package_priorities.get(self.packages[i][0], 0), 
                                reverse=True)
        
        assignments = {}
        
        # For each package, find the best robot
        for pkg_idx in sorted_packages:
            if not available_robots:
                break
                
            pkg = self.packages[pkg_idx]
            pkg_id = pkg[0]
            start_pos = (pkg[1], pkg[2])
            
            best_robot = None
            best_score = -float('inf')
            
            for robot_id in available_robots:
                robot_pos = (self.robots[robot_id][0], self.robots[robot_id][1])
                score = calculate_score(self.map, robot_pos, pkg, self.current_time)
                
                if score > best_score:
                    best_score = score
                    best_robot = robot_id
            
            if best_robot is not None:
                assignments[best_robot] = pkg_idx
                available_robots.remove(best_robot)
                self.packages_free[pkg_idx] = False
                self.robots_target[best_robot] = pkg_id
        
        return assignments

    def get_actions(self, state):
        """Get actions for all robots based on current state"""
        if not self.is_init:
            # Initialize agent state on first call
            self.is_init = True
            self.init_agents(state)
        else:
            # Update internal state
            self.update_inner_state(state)
        
        actions = []
        
        # Assign free robots to packages
        assignments = self.assign_packages_to_robots()
        
        # Generate actions for each robot
        for i in range(self.n_robots):
            if self.robots_target[i] != 'free':
                target_pkg_id = self.robots_target[i]
                pkg_idx = next((j for j, p in enumerate(self.packages) if p[0] == target_pkg_id), None)
                
                if pkg_idx is None:
                    # Package doesn't exist or was already delivered
                    self.robots_target[i] = 'free'
                    actions.append(('S', '0'))
                    continue
                
                if self.robots[i][2] != 0:
                    # Robot has the package, move to delivery location
                    move, action = self.update_move_to_target(i, pkg_idx, 'target')
                else:
                    # Robot needs to pick up the package
                    move, action = self.update_move_to_target(i, pkg_idx, 'start')
                
                actions.append((move, action))
            else:
                # No assigned task, stay in place
                actions.append(('S', '0'))
        
        return actions