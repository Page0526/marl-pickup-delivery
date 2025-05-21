from collections import deque
import numpy as np

# Cải tiến BFS: dừng sớm, rõ ràng
def run_bfs(map, start, goal):
    n_rows = len(map)
    n_cols = len(map[0])

    queue = deque()
    visited = set()
    parent = {}
    queue.append(goal)
    visited.add(goal)
    d = {goal: 0}

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    actions = ['U', 'D', 'L', 'R']

    while queue:
        current = queue.popleft()

        for dx, dy in dirs:
            next_pos = (current[0] + dx, current[1] + dy)
            if (0 <= next_pos[0] < n_rows and 0 <= next_pos[1] < n_cols and
                map[next_pos[0]][next_pos[1]] == 0 and next_pos not in visited):

                visited.add(next_pos)
                d[next_pos] = d[current] + 1
                parent[next_pos] = current
                queue.append(next_pos)

                if next_pos == start:
                    break

    if start not in d:
        return 'S', 100000

    for i, (dx, dy) in enumerate(dirs):
        next_pos = (start[0] + dx, start[1] + dy)
        if next_pos in d and d[next_pos] == d[start] - 1:
            return actions[i], d[start]

    return 'S', d[start]


class Agents:

    def __init__(self):
        self.agents = []
        self.packages = []
        self.packages_free = []
        self.n_robots = 0
        self.state = None
        self.is_init = False

    def init_agents(self, state):
        self.state = state
        self.n_robots = len(state['robots'])
        self.map = state['map']
        self.robots = [(robot[0]-1, robot[1]-1, 0) for robot in state['robots']]
        self.robots_target = ['free'] * self.n_robots
        self.packages = [(p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5]) for p in state['packages']]
        self.packages_free = [True] * len(self.packages)

    def update_move_to_target(self, robot_id, target_package_id, phase='start'):
        i = robot_id
        rx, ry = self.robots[i][0], self.robots[i][1]
        pkg = self.packages[target_package_id]

        # xác định mục tiêu
        if phase == 'start':
            target = (pkg[1], pkg[2])
        else:
            target = (pkg[3], pkg[4])

        # nếu đã tới nơi: nhặt hoặc thả hàng
        if (rx, ry) == target:
            pkg_act = 1 if phase == 'start' else 2
            return 'S', str(pkg_act)

        # nếu chưa đến → di chuyển bằng BFS
        move, _ = run_bfs(self.map, (rx, ry), target)
        return move, '0'

    def update_inner_state(self, state):
        for i in range(len(state['robots'])):
            prev = self.robots[i]
            robot = state['robots'][i]
            self.robots[i] = (robot[0]-1, robot[1]-1, robot[2])

            # Nếu robot vừa giao xong → đặt lại trạng thái
            if prev[2] != 0 and self.robots[i][2] == 0:
                self.robots_target[i] = 'free'

        new_pkgs = [(p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5]) for p in state['packages']]
        self.packages += new_pkgs
        self.packages_free += [True] * len(new_pkgs)

    def get_actions(self, state):
        if not self.is_init:
            self.is_init = True
            self.init_agents(state)
        else:
            self.update_inner_state(state)

        actions = []
        for i in range(self.n_robots):
            rx, ry, carrying = self.robots[i]

            if self.robots_target[i] != 'free':
                pkg_id = self.robots_target[i]
                if carrying != 0:
                    move, act = self.update_move_to_target(i, pkg_id-1, phase='target')
                else:
                    move, act = self.update_move_to_target(i, pkg_id-1, phase='start')
                actions.append((move, act))
            else:
                # tìm gói gần nhất chưa nhận
                best_id = None
                best_dist = 1e9
                for j, pkg in enumerate(self.packages):
                    if not self.packages_free[j]:
                        continue
                    dist = abs(pkg[1] - rx) + abs(pkg[2] - ry)
                    if dist < best_dist:
                        best_dist = dist
                        best_id = pkg[0]

                if best_id is not None:
                    self.packages_free[best_id - 1] = False
                    self.robots_target[i] = best_id
                    move, act = self.update_move_to_target(i, best_id - 1, phase='start')
                    actions.append((move, act))
                else:
                    actions.append(('S', '0'))

        return actions
