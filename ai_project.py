import random
import math
import heapq
from collections import deque
def init_maze(): #initialize maze
  maze = [[0 for _ in range(10)] for _ in range(10)]
  maze[0][0] = 2
  i = 0
  while i < 20:
    num1 = random.randint(2, 9)
    num2 = random.randint(2, 9)
    if maze[num1][num2] == 0:
      maze[num1][num2] = 1
      i += 1
  i = 0
  while i < 2:
    n1 = random.randrange(10)
    n2 = random.randrange(10)
    if i == 0:
      if maze[n1][n2] == 0:
        maze[n1][n2] = 3
        i += 1
    if i == 1:
      if maze[n1][n2] == 0:
        maze[n1][n2] = 4
        i += 1
  return maze

def print_maze(maze):
  cols = len(maze[0])
  horizontal_line = '+' + ('---+' * cols)

  for row in maze:
    print(horizontal_line)
    row_str = '| '
    for cell in row:
      if cell == 1:
        cell_display = f"\033[91m{cell}\033[0m"  # red
      elif cell == 2:
        cell_display = f"\033[33m{cell}\033[0m"  # yellow
      elif cell == 3:
        cell_display = f"\033[34m{cell}\033[0m"  # blue
      elif cell == 4:
        cell_display = f"\033[32m{cell}\033[0m"  # green
      elif cell == 5:
        cell_display = f"\033[35m{cell}\033[0m" #purple
      else:
        cell_display = f"\033[37m{cell}\033[0m"  # white for 0's
      row_str += f"{cell_display:^1} | "
    print(row_str)
  print(horizontal_line)
#--------------------------------end of init----------------------------#
#--------------------------------part one-------------------------------#
 # function to get the adjecent cells
def getadjecentcells(po):
    x,y=po
    direction=[(0,1),(0,-1),(1,0),(-1,0)]
    return [(x+dx,y+dy) for dx,dy in direction if 0<=x+dx<10 and 0<=y+dy<10]
# end of function
# use it to define the key , end and start
def findvalue(maze,value):
    for i in range(10):
        for j in range(10):
            if maze[i][j] == value:
                return (i,j)
# end

# dfs function
def DFS(maze, start, end):
    stack = [start]
    visited = set([start])
    parent = {} # to know the where i came from
    while stack:
        current_pos = stack.pop() # remove the cell to start exploring other cells
        if current_pos == end:
            path = []
            while current_pos != start: #starting from the last cell to reach the start
                path.append(current_pos) # adding the last cell to the path
                current_pos = parent[current_pos] # to go back in order to know the parent of current node
            path.append(start) # to add the start to the path
            return path[::-1]  # Reverse to get start->end

        for adjacent in getadjecentcells(current_pos):
            x, y = adjacent
            if maze[x][y] != 1 and adjacent not in visited:
                visited.add(adjacent)
                parent[adjacent] = current_pos  # Record parent
                stack.append(adjacent)

# end of dfs
# bfs
from collections import deque

def BFS(maze, start, end):
    queue = deque([start])
    visited = set([start])
    parent = {}

    while queue:
        current_pos = queue.popleft()
        if current_pos == end:
            path = []
            while current_pos != start:
                path.append(current_pos)
                current_pos = parent[current_pos]
            path.append(start)
            return path[::-1]  # Reverse to get start->end
        for adjacent in getadjecentcells(current_pos):
            x, y = adjacent
            if maze[x][y] != 1 and adjacent not in visited:
                visited.add(adjacent)
                parent[adjacent] = current_pos
                queue.append(adjacent)
    return None
#end bfs
#-------------------------------------------end part one---------------------------------------#
#-------------------------------------------part tow-------------------------------------------#
def euclidean_distance(x1, y1, x2, y2):
  return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def hill_climbing(maze, start, goal):
  rows, cols = len(maze), len(maze[0])
  directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
  path = [start]
  current = start

  while current != goal:
    best_move = None
    best_score = float('inf')

    for d in directions:
      neighbor = (current[0] + d[0], current[1] + d[1])

      if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
        if maze[neighbor[0]][neighbor[1]] == 1:  # Wall (Obstacle)
          continue

        score = euclidean_distance(*neighbor, *goal)

        if score < best_score:  # Choose best move
          best_score = score
          best_move = neighbor

    if best_move is None:  # No valid moves, stuck!
      return None

    current = best_move
    path.append(current)

  return path  # Returns the sequence of moves to the goal


def astar_search(maze, start, goal):
  rows, cols = len(maze), len(maze[0])
  directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
  open_list = []
  heapq.heappush(open_list, (0, start))  # Priority queue (cost, node)

  came_from = {}  # Tracks the path
  g_score = {start: 0}
  f_score = {start: euclidean_distance(*start, *goal)}

  while open_list:
    _, current = heapq.heappop(open_list)

    if current == goal:
      path = []
      while current in came_from:
        path.append(current)
        current = came_from[current]
      return path[::-1]  # Reverse to get correct order

    for d in directions:
      neighbor = (current[0] + d[0], current[1] + d[1])

      if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
        if maze[neighbor[0]][neighbor[1]] == 1:  # Wall (Obstacle)
          continue

        tentative_g_score = g_score[current] + 1  # Movement cost

        if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
          came_from[neighbor] = current
          g_score[neighbor] = tentative_g_score
          f_score[neighbor] = g_score[neighbor] + euclidean_distance(*neighbor, *goal)
          heapq.heappush(open_list, (f_score[neighbor], neighbor))

  return None  # No path found


# Locate Positions in the Maze
def find_position(maze, value):
  for i in range(len(maze)):
    for j in range(len(maze[0])):
      if maze[i][j] == value:
        return (i, j)
  return None
#-------------------------------------------end part tow---------------------------------------#
#--------------------------------------------part three----------------------------------------#
#minimax
def get_valid_moves(pos, maze,forbidden_pos):
  moves = []
  directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
  for dx, dy in directions:
    nx, ny = pos[0] + dx, pos[1] + dy
    if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]):
      if maze[nx][ny] !=1:
        if (nx, ny) == forbidden_pos:
          continue
        moves.append((nx, ny))
  return moves
def dest(pos1,goal):
  return abs(pos1[0] - goal[0]) + abs(pos1[1] - goal[1])
def win_check(ai_pos,op_pos,goal):
  if ai_pos==goal:
    return 1000
  elif op_pos==goal:
    return -1000
  else:
    ai_pos=dest(ai_pos,goal)
    op_pos=dest(op_pos,goal)
    return (op_pos - ai_pos)
def find_key(maze):
  for i in range(len(maze)):
    for j in range(len(maze[0])):
      if maze[i][j] == 3:
        return (i, j)
  return None
def find_tresure(maze):
  for i in range(len(maze)):
    for j in range(len(maze[0])):
      if maze[i][j] == 4:
        return (i, j)
  return None
def minimax(ai_pos,op_pos,maze,depth,action):
  if action==1:
    _, p = max_turn(ai_pos, op_pos, find_tresure(maze), maze, depth)
    return p
  _, p = max_turn(ai_pos, op_pos, find_key(maze), maze, depth)
  return p
def max_turn(ai_pos,op_pos,goal,maze,depth):
  if win_check(ai_pos,op_pos,goal)==1000 or win_check(ai_pos,op_pos,goal)==-1000 or depth==0:
    return win_check(ai_pos,op_pos,goal),[]
  v=-math.inf
  path=[]
  for move in get_valid_moves(ai_pos,maze,op_pos):
    n_ai_pos = move
    score,subpath=min_turn(n_ai_pos,op_pos,goal,maze,depth-1)
    if score>v:
      v=score
      path=[n_ai_pos]+subpath
  return v,path
def min_turn(ai_pos,op_pos,goal,maze,depth):
  if win_check(ai_pos,op_pos,goal)==1000 or win_check(ai_pos,op_pos,goal)==-1000 or depth==0:
    return win_check(ai_pos,op_pos,goal),[]
  v=math.inf
  path=[]
  for move in get_valid_moves(op_pos,maze,ai_pos):
    n_op_pos = move
    score,subpath=max_turn(ai_pos,n_op_pos,goal,maze,depth-1)
    if score<v:
      v=score
      path=[n_op_pos]+subpath
  return v,path
def ai_post(maze):
  i=0
  while i<1:
    num1 = random.randint(2, 9)
    num2 = random.randint(2, 9)
    if maze[num1][num2] == 0:
      maze[num1][num2] = 5
      i += 1
  return (num1,num2)
#alpha beta
def minimax_alpha_beta(ai_pos, op_pos, maze, depth, action):
  goal = find_tresure(maze) if action == 1 else find_key(maze)
  _, path = max_turn_ab(ai_pos, op_pos, goal, maze, depth, -math.inf, math.inf)
  return path

def max_turn_ab(ai_pos, op_pos, goal, maze, depth, alpha, beta):
  if win_check(ai_pos, op_pos, goal) in [1000, -1000] or depth == 0:
    return win_check(ai_pos, op_pos, goal), []

  max_val = -math.inf
  best_path = []

  for move in get_valid_moves(ai_pos, maze,op_pos):
    n_ai_pos = move
    score, subpath = min_turn_ab(n_ai_pos, op_pos, goal, maze, depth - 1, alpha, beta)

    if score > max_val:
      max_val = score
      best_path = [n_ai_pos] + subpath

    alpha = max(alpha, max_val)
    if beta <= alpha:
      break

  return max_val, best_path


def min_turn_ab(ai_pos, op_pos, goal, maze, depth, alpha, beta):
  if win_check(ai_pos, op_pos, goal) in [1000, -1000] or depth == 0:
    return win_check(ai_pos, op_pos, goal), []

  min_val = math.inf
  best_path = []

  for move in get_valid_moves(op_pos, maze,ai_pos):
    n_op_pos = move
    score, subpath = max_turn_ab(ai_pos, n_op_pos, goal, maze, depth - 1, alpha, beta)

    if score < min_val:
      min_val = score
      best_path = [n_op_pos] + subpath

    beta = min(beta, min_val)
    if beta <= alpha:
      break

  return min_val, best_path
#-----------------------------------end part three---------------------------------------#

#--------------------------------------main-----------------------------------------------#
i="yes"
while i=="yes":
  print("\nEnter the operation you want to perform:\n\n"
        "1.DFS and BFS algorithms\n"
        "2.HILL CLIMBING and AO* algorithms\n"
        "3.MINIMAX algorithm\n"
        "4.ALPHA-BETA algorithm\n")
  op = int(input())
  if op == 1:
    my_maze = init_maze()
    print_maze(my_maze)
    start = findvalue(my_maze, 2)
    end = findvalue(my_maze, 4)
    key = findvalue(my_maze, 3)
    path_dfs_part1 = DFS(my_maze, start, key)
    path_dfs_part2 = DFS(my_maze, key, end)
    path_dfs = path_dfs_part1 + path_dfs_part2[1:] if path_dfs_part1 and path_dfs_part2 else None
    # dfs
    if path_dfs:
      print(f"DFS: the number of steps is = {len(path_dfs)} ")
      print(f"DFS: the way to the end in DFS is  {path_dfs} ")
    else:
      print("DFS: No path found.")
      print(f"DFS: the way to the end in DFS is  {path_dfs}")

    # BFS result
    path_bfs_part1 = BFS(my_maze, start, key)
    path_bfs_part2 = BFS(my_maze, key, end)
    path_bfs = path_bfs_part1 + path_bfs_part2[1:] if path_bfs_part1 and path_bfs_part2 else None

    if path_bfs:
      print(f"BFS: the number of steps is = {len(path_bfs)} ")
      print(f"BFS: the way to the end in BFS is  {path_bfs} ")
    else:
      print("BFS: No path found.")
      print(f"BFS: the way to the end in BFS is  {path_bfs}")
  elif op== 2:
    # Find required positions
    my_maze = init_maze()
    print_maze(my_maze)
    start_pos = find_position(my_maze, 2)  # Start position
    intermediate_pos = find_position(my_maze, 3)  # First target position
    goal_pos = find_position(my_maze, 4)  # Final destination

    # Compute paths
    if start_pos and intermediate_pos and goal_pos:
      path_to_intermediate = hill_climbing(my_maze, start_pos, intermediate_pos)
      path_to_goal = hill_climbing(my_maze, intermediate_pos, goal_pos)

      if path_to_intermediate and path_to_goal:
        full_path = path_to_intermediate + path_to_goal
        print("Hill climbing path to treasure:", full_path)
      else:
        print("No valid path found!")
    else:
      print("Error: Could not locate all required positions in the maze.")

    # Compute paths
    if start_pos and intermediate_pos and goal_pos:
      path_to_intermediate = astar_search(my_maze, start_pos, intermediate_pos)
      path_to_goal = astar_search(my_maze, intermediate_pos, goal_pos)

      if path_to_intermediate and path_to_goal:
        full_path = path_to_intermediate + path_to_goal
        print("A* path to treasure:", full_path)
      else:
        print("No valid path found!")
    else:
      print("Error: Could not locate all required positions in the maze.")
  elif op == 3:
    my_maze = init_maze()
    print()
    op_pos = (0, 0)
    depth = 14
    AI = input("1. ai in random position \n"
               "2. ai in the same opposite position \n")
    if AI == "1":
      ai_pos = ai_post(my_maze)
    elif AI == "2":
      ai_pos = (0, 0)
    else:
      print("Invalid operation")
      ai_pos = (0, 0)

    full_path = minimax(ai_pos, op_pos, my_maze, depth, 2)
    print("Planned path:", full_path)
    goal = find_key(my_maze)
    print("key founded in " + str(goal))
    print()
    print_maze(my_maze)
    print()
    for idx, next_pos in enumerate(full_path):
      k = 0
      if idx % 2 == 0:
        print(f"AI moves to: {next_pos}")
        ai_pos = next_pos
      else:
        print(f"Opponent moves to: {next_pos}")
        op_pos = next_pos

      goal = find_key(my_maze)
      if win_check(ai_pos, op_pos, goal) == 1000:
        print("AI found the key!")
        k = 1
        break
      elif win_check(ai_pos, op_pos, goal) == -1000:
        print("Opponent found the key!")
        k = 2
        break
    else:
      print("No one found the key!.")
    for i in range(len(my_maze)):
      for j in range(len(my_maze[0])):
        if my_maze[i][j] == 2 or my_maze[i][j] == 5:
          my_maze[i][j] = 0
    my_maze[ai_pos[0]][ai_pos[1]] = 5
    my_maze[op_pos[0]][op_pos[1]] = 2
    print_maze(my_maze)
    if k == 1:
      goal1 = find_tresure(my_maze)
      print("tresure founded in " + str(goal1))
      depth = 14
      full_path = minimax(ai_pos, op_pos, my_maze, depth, 1)
      print("Planned path:", full_path)
      for idx in range(0, len(full_path), 2):
        next_pos = full_path[idx]
        print(f"AI moves to: {next_pos}")
        ai_pos = next_pos
        goal = find_tresure(my_maze)
        if win_check(ai_pos, op_pos, goal) == 1000:
          print("AI won!")
          break

      else:
        print("No one won.")
    elif k == 2:
      goal1 = find_tresure(my_maze)
      print("tresure founded in " + str(goal1))
      depth = 14
      full_path = minimax(ai_pos, op_pos, my_maze, depth, 1)
      print("Planned path:", full_path)
      for idx in range(1, len(full_path), 2):
        next_pos = full_path[idx]
        print(f"opponent moves to: {next_pos}")
        op_pos = next_pos
        goal = find_tresure(my_maze)
        if win_check(ai_pos, op_pos, goal) == -1000:
          print("opponent won!")
          break

      else:
        print("No one won.")
    else:
      print("No one won.")
  elif op == 4:
    my_maze = init_maze()
    print()
    op_pos = (0, 0)
    depth = 20
    AI = input("1. ai in random position \n"
               "2. ai in the same opposite position \n")
    if AI == "1":
      ai_pos = ai_post(my_maze)
    elif AI == "2":
      ai_pos = (0, 0)
    else:
      print("Invalid operation")
      ai_pos = (0, 0)

    full_path = minimax_alpha_beta(ai_pos, op_pos, my_maze, depth, 2)
    print("Planned path:", full_path)
    goal = find_key(my_maze)
    print("key founded in " + str(goal))
    print()
    print_maze(my_maze)
    print()
    for idx, next_pos in enumerate(full_path):
      k = 0
      if idx % 2 == 0:
        print(f"AI moves to: {next_pos}")
        ai_pos = next_pos
      else:
        print(f"Opponent moves to: {next_pos}")
        op_pos = next_pos

      goal = find_key(my_maze)
      if win_check(ai_pos, op_pos, goal) == 1000:
        print("AI found the key!")
        k = 1
        break
      elif win_check(ai_pos, op_pos, goal) == -1000:
        print("Opponent found the key!")
        k = 2
        break
    else:
      print("No one found the key.")
    for i in range(len(my_maze)):
      for j in range(len(my_maze[0])):
        if my_maze[i][j] == 2 or my_maze[i][j] == 5:
          my_maze[i][j] = 0
    my_maze[ai_pos[0]][ai_pos[1]] = 5
    my_maze[op_pos[0]][op_pos[1]] = 2
    print_maze(my_maze)
    if k == 1:
      goal1 = find_tresure(my_maze)
      print("tresure founded in " + str(goal1))
      depth = 20
      full_path = minimax_alpha_beta(ai_pos, op_pos, my_maze, depth, 1)
      print("Planned path:", full_path)
      for idx in range(0, len(full_path), 2):
        next_pos = full_path[idx]
        print(f"AI moves to: {next_pos}")
        ai_pos = next_pos
        goal = find_tresure(my_maze)
        if win_check(ai_pos, op_pos, goal) == 1000:
          print("AI won!")
          break

      else:
        print("No one won.")
    elif k == 2:
      goal1 = find_tresure(my_maze)
      print("tresure founded in " + str(goal1))
      depth = 20
      full_path = minimax_alpha_beta(ai_pos, op_pos, my_maze, depth, 1)
      print("Planned path:", full_path)
      for idx in range(1, len(full_path), 2):
        next_pos = full_path[idx]
        print(f"opponent moves to: {next_pos}")
        op_pos = next_pos
        goal = find_tresure(my_maze)
        if win_check(ai_pos, op_pos, goal) == -1000:
          print("opponent won!")
          break

      else:
        print("No one won.")
    else:
      print("No one won.")
  else:
    print("Invalid input.")
  i=input("Do you want to continue? (yes/no): ")


