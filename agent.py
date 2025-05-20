
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque, namedtuple
import heapq
import os

# Define experience replay memory to store experiences
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, k=batch_size)
        
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# A* search algorithm for pathfinding
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
        path: Full path from start to goal
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
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    open_set_hash = {start}
    
    while open_set:
        _, _, current, path = heapq.heappop(open_set)
        open_set_hash.remove(current)
        
        if current == goal:
            if len(path) <= 1:
                return 'S', path, 0
            
            # Determine first step direction
            first_step = path[1]
            dx, dy = first_step[0] - start[0], first_step[1] - start[1]
            
            # Convert direction to action
            if dx == -1 and dy == 0:
                return 'U', path, len(path) - 1
            elif dx == 1 and dy == 0:
                return 'D', path, len(path) - 1
            elif dx == 0 and dy == -1:
                return 'L', path, len(path) - 1
            elif dx == 0 and dy == 1:
                return 'R', path, len(path) - 1
            return 'S', path, len(path) - 1
        
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
    return 'S', [start], float('inf')

# Function to create state representation for neural network
def create_state_representation(map_data, robot_pos, robot_carrying, packages, current_time, max_packages=10):
    """
    Create a state representation for the DQN
    
    Args:
        map_data: 2D grid map
        robot_pos: Current robot position (x, y)
        robot_carrying: Package ID the robot is carrying (0 if none)
        packages: List of packages with their details
        current_time: Current time step
        
    Returns:
        state_rep: Flattened representation of state for DQN input
    """
    # Map dimensions
    n_rows = len(map_data)
    n_cols = len(map_data[0])
    
    # Create a representation for the robot's position and carrying status
    robot_state = np.zeros((n_rows, n_cols, 2))
    robot_state[robot_pos[0], robot_pos[1], 0] = 1  # Robot position
    
    if robot_carrying != 0:
        robot_state[:, :, 1] = 1  # Robot is carrying a package
    
    # Create representation for packages (both pickup and delivery locations)
    package_state = np.zeros((n_rows, n_cols, 2))
    
    # Add active packages (waiting or in transit)
    package_count = 0
    package_features = []
    
    for pkg in packages:
        # Skip delivered packages
        if pkg[0] == robot_carrying or (pkg[5] <= current_time and package_count < max_packages):
            # Package pickup location
            package_state[pkg[1]-1, pkg[2]-1, 0] += 1
            
            # Package delivery location
            package_state[pkg[3]-1, pkg[4]-1, 1] += 1
            
            # Normalized time until deadline
            time_to_deadline = max(0, pkg[6] - current_time) / 100.0
            
            package_features.append([time_to_deadline])
            package_count += 1
            
            if package_count >= max_packages:
                break
    
    # Pad package features if needed
    while len(package_features) < max_packages:
        package_features.append([0.0])
    
    # Flatten and normalize features
    robot_features = robot_state.flatten()
    package_spatial_features = package_state.flatten()
    package_features = np.array(package_features).flatten()
    
    # Combine all features
    state_rep = np.concatenate([
        robot_features, 
        package_spatial_features,
        package_features,
        [current_time / 1000.0]  # Normalize time
    ])
    
    return state_rep

# DQN Agent class for controlling a single robot
class DQNAgent:
    def __init__(self, state_size, action_size, model_path=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(capacity=10000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_target_every = 100
        self.target_update_counter = 0
        
        # Main model and target model
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        
        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self.model.load_weights(model_path)
            self.target_model.load_weights(model_path)
            self.epsilon = self.epsilon_min
            print(f"Loaded model from {model_path}")
    
    def _build_model(self):
        # Neural Network for Deep-Q learning
        model = tf.keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def memorize(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
    
    def act(self, state, training=True):
        # Epsilon-greedy action selection
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Get Q-values from model
        q_values = self.model.predict(np.array([state]), verbose=0)[0]
        return np.argmax(q_values)

    @tf.function(reduce_retracing=True)
    def _train_step(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        # Sample minibatch with fixed batch_size and ensure shapes are consistent
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.amax(next_q_values[i])
        
        targets = tf.convert_to_tensor(targets, dtype=tf.float32)

        # Call the training step (wrapped to reduce retracing)
        self._train_step(states, targets)

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target model periodically
        self.target_update_counter += 1
        if self.target_update_counter >= self.update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    
    def save_model(self, filepath):
        self.model.save_weights(filepath)
        print(f"Model saved to {filepath}")

# Hybrid approach: RL for high-level decisions, A* for path planning
class Agents:
    def __init__(self, model_dir=None, use_rl=True):
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
        self.use_rl = use_rl
        self.model_dir = model_dir
        
        # State representation size
        self.state_size = None  # Will be set during initialization
        
        # Action space: 
        # 0: No assignment, 
        # 1-10: Assign to package with index 0-9
        self.action_size = 11
        
        # RL agents (one per robot)
        self.dqn_agents = []
        
        # Training parameters
        self.training = False
        self.batch_size = 32
        self.rewards = {}
        
        # Track assignments to prevent constant reassignment
        self.current_assignments = {}
        self.assignment_durations = {}
        self.min_assignment_steps = 5  # Minimum steps before reassignment
    
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
        self.rewards = {i: 0 for i in range(self.n_robots)}
        self.current_assignments = {i: None for i in range(self.n_robots)}
        self.assignment_durations = {i: 0 for i in range(self.n_robots)}
        
        # Calculate map dimensions for state representation
        n_rows = len(self.map)
        n_cols = len(self.map[0])
        
        # State size: robot_state + package_state + package_features + time
        self.state_size = n_rows * n_cols * 2 + n_rows * n_cols * 2 + 10 * 1 + 1
        
        # Initialize RL agents if not already done
        if self.use_rl and not self.dqn_agents:
            for i in range(self.n_robots):
                model_path = None
                if self.model_dir:
                    model_path = f"{self.model_dir}/agent_{i}.h5"
                self.dqn_agents.append(DQNAgent(self.state_size, self.action_size, model_path))
        
        # Track all packages
        self.packages = [(p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5], p[6]) for p in state['packages']]
        self.packages_free = [True] * len(self.packages)
        
        # Calculate package priorities based on urgency and distance
        self.update_package_priorities()
    
    def update_inner_state(self, state):
        """Update agent's internal state based on environment state"""
        self.current_time += 1
        
        # Store previous robot states for reward calculation
        prev_robots = self.robots.copy()
        prev_targets = self.robots_target.copy()
        
        # Reset future positions
        self.robot_future_positions = {i: set() for i in range(self.n_robots)}
        
        # Update robot positions and states
        for i in range(len(state['robots'])):
            prev = (self.robots[i][0], self.robots[i][1], self.robots[i][2])
            robot = state['robots'][i]
            self.robots[i] = (robot[0]-1, robot[1]-1, robot[2])
            
            # Update assignment durations
            if self.current_assignments[i] is not None:
                self.assignment_durations[i] += 1
            
            if prev[2] != 0 and self.robots[i][2] == 0:
                # Robot has dropped the package
                self.robots_target[i] = 'free'
                self.current_assignments[i] = None
                self.assignment_durations[i] = 0
            elif self.robots[i][2] != 0:
                # Robot is carrying a package
                self.robots_target[i] = self.robots[i][2]
                
        # Add new packages
        old_package_count = len(self.packages)
        self.packages += [(p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5], p[6]) for p in state['packages']]
        self.packages_free += [True] * len(state['packages'])
        
        # Update package priorities
        self.update_package_priorities()
    
    def update_package_priorities(self):
        """Update package priorities based on urgency and distances"""
        self.package_priorities = {}
        
        for i, pkg in enumerate(self.packages):
            if not self.packages_free[i]:
                continue
                
            pkg_id, start_x, start_y, end_x, end_y, appeared_time, deadline = pkg
            time_to_deadline = deadline - self.current_time
            
            # Estimate pickup and delivery distances
            pickup_dist = float('inf')
            for robot_id in range(self.n_robots):
                if self.robots_target[robot_id] == 'free':
                    robot_pos = (self.robots[robot_id][0], self.robots[robot_id][1])
                    _, _, dist = run_astar(self.map, robot_pos, (start_x, start_y))
                    pickup_dist = min(pickup_dist, dist)
            
            # Estimate delivery distance
            _, _, delivery_dist = run_astar(self.map, (start_x, start_y), (end_x, end_y))
            
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
    
    def get_state_representation(self, robot_id):
        """Get state representation for a specific robot for the DQN"""
        robot_pos = (self.robots[robot_id][0], self.robots[robot_id][1])
        robot_carrying = self.robots[robot_id][2]
        
        return create_state_representation(
            self.map,
            robot_pos,
            robot_carrying,
            self.packages,
            self.current_time
        )
    
    def assign_packages_with_rl(self):
        """Use RL agents to decide package assignments"""
        assignments = {}
        
        for robot_id in range(self.n_robots):
            # Skip robots already carrying packages
            if self.robots[robot_id][2] != 0:
                continue
            
            # Check if current assignment should be maintained
            if (self.current_assignments[robot_id] is not None and 
                self.assignment_durations[robot_id] < self.min_assignment_steps):
                pkg_idx = self.current_assignments[robot_id]
                if pkg_idx < len(self.packages) and self.packages_free[pkg_idx]:
                    assignments[robot_id] = pkg_idx
                    self.packages_free[pkg_idx] = False
                    self.robots_target[robot_id] = self.packages[pkg_idx][0]
                    continue
            
            # Get state representation
            state = self.get_state_representation(robot_id)
            
            # Get action from RL agent
            action = self.dqn_agents[robot_id].act(state, training=self.training)
            
            # Process action (0 means no assignment, 1-10 means assign to package index 0-9)
            if action > 0 and action <= 10:
                pkg_idx = action - 1
                
                # Check if package index is valid and package is free
                if pkg_idx < len(self.packages) and self.packages_free[pkg_idx]:
                    assignments[robot_id] = pkg_idx
                    self.packages_free[pkg_idx] = False
                    self.robots_target[robot_id] = self.packages[pkg_idx][0]
                    
                    # Track assignment
                    self.current_assignments[robot_id] = pkg_idx
                    self.assignment_durations[robot_id] = 0
        
        return assignments
    
    def assign_packages_greedy(self):
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
                
                # Calculate score based on distance and urgency
                _, _, pickup_dist = run_astar(self.map, robot_pos, start_pos)
                _, _, delivery_dist = run_astar(self.map, start_pos, (pkg[3], pkg[4]))
                
                total_dist = pickup_dist + delivery_dist
                time_to_deadline = pkg[6] - self.current_time
                
                # Higher score for closer packages with tighter deadlines
                score = (1000.0 / (total_dist + 1))
                
                # Adjust score based on deadline
                if time_to_deadline < total_dist:
                    # Package likely to be late
                    score *= 0.5
                elif time_to_deadline < 2 * total_dist:
                    # Urgency factor for packages with tight deadlines
                    score *= 2.0
                
                if score > best_score:
                    best_score = score
                    best_robot = robot_id
            
            if best_robot is not None:
                assignments[best_robot] = pkg_idx
                available_robots.remove(best_robot)
                self.packages_free[pkg_idx] = False
                self.robots_target[best_robot] = pkg_id
                
                # Track assignment
                self.current_assignments[best_robot] = pkg_idx
                self.assignment_durations[best_robot] = 0
        
        return assignments
    
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
        move, path, distance = run_astar(self.map, robot_pos, target_pos, occupied_positions)
        
        # Store path for future reference
        self.robot_paths[robot_id] = path
        
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
        if self.use_rl:
            assignments = self.assign_packages_with_rl()
        else:
            assignments = self.assign_packages_greedy()
        
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
    
    def train(self, batch_size=32):
        """Train the RL agents with experiences from memory"""
        if not self.use_rl or not self.training:
            return
        
        for i in range(self.n_robots):
            self.dqn_agents[i].replay(batch_size)
    
    def save_models(self, directory):
        """Save all agent models to disk"""
        if not self.use_rl:
            return
            
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        for i in range(self.n_robots):
            self.dqn_agents[i].save_model(f"{directory}/agent_{i}.h5")

# Function for training the agents
def train_agents(env, n_episodes=100, save_dir='models'):
    """
    Train the multi-agent RL system
    
    Args:
        env: Environment object
        n_episodes: Number of episodes to train
        save_dir: Directory to save models
    """
    agents = Agents(use_rl=True)
    agents.training = True
    
    best_reward = -float('inf')
    
    for episode in range(n_episodes):
        state = env.reset()
        agents.init_agents(state)
        done = False
        episode_reward = 0
        
        while not done:
            # Store current states for all agents
            current_states = [agents.get_state_representation(i) for i in range(agents.n_robots)]
            
            # Get actions from agents
            actions = agents.get_actions(state)
            
            # Execute actions in environment
            next_state, reward, done, infos = env.step(actions)
            episode_reward += reward
            
            # Store next states for all agents
            next_states = [agents.get_state_representation(i) for i in range(agents.n_robots)]
            
            # Store experiences in replay buffer
            for i in range(agents.n_robots):
                # Simplified action encoding for replay buffer
                # 0 if no package assigned, otherwise the package index + 1
                action_idx = 0
                if agents.current_assignments[i] is not None:
                    action_idx = agents.current_assignments[i] + 1
                
                # Store experience
                agents.dqn_agents[i].memorize(
                    current_states[i],
                    action_idx,
                    reward,  # Full reward (we could distribute it among agents)
                    next_states[i],
                    done
                )
            
            # Update state
            state = next_state
            
            # Train agents
            agents.train(batch_size=32)
        
        # After episode completion
        print(f"Episode {episode+1}/{n_episodes}, Total Reward: {episode_reward}, Steps: {env.t}")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agents.save_models(f"{save_dir}/best")
        
        # Save checkpoint models
        # if (episode + 1) % 10 == 0:
        #     agents.save_models(f"{save_dir}/episode_{episode+1}")
    
    # Save final models
    # agents.save_models(f"{save_dir}/final")
    
    return agents