from queue import PriorityQueue, Queue, LifoQueue
import pygame
import random
import pickle

pygame.init()
pygame.mixer.init()
pygame.mixer.music.load("instruction_music.mp3")
pygame.mixer.music.load("algorithm_music.mp3")
WIDTH = 900
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Path Finding Game")

algorithm_run = False
error_log = []


title_font = pygame.font.SysFont('Arial', 30)
section_font = pygame.font.SysFont('Arial', 26)
text_font = pygame.font.SysFont('Arial', 22)



RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
LIGHT_BLUE = (173, 216, 230)
LIGHT_YELLOW = (255, 255, 200)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False

def stop_music():
    pygame.mixer.music.stop()

def play_instruction_music():
    stop_music()
    pygame.mixer.music.load("instruction_music.mp3")
    pygame.mixer.music.play(-1)

def play_algorithm_music():
    stop_music()
    pygame.mixer.music.load("algorithm_music.mp3")
    pygame.mixer.music.play(-1)

def manhattan_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def euclidean_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def chebyshev_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return max(abs(x1 - x2), abs(y1 - y2))


def reconstruct_path(came_from, current, draw, start):
    path = []
    while current in came_from:
        path.append(current.get_pos())
        current = came_from[current]
        current.make_path()
        draw()
    path.reverse()
    print_path(path, start.get_pos(), len(path))
    start.make_start()
    draw()


def print_path(path, start, length):
    print("Path")
    print("Start:")
    print(f"  {start},")
    for pos in path:
        print(f"  {pos},")
    print("Goal")
    print(f"Length: {length} steps")

def print_no_path_found():
    print("No path found!")

def generate_maze(grid, start, end):
    global algorithm_run
    if algorithm_run:
        add_error("Please clear the grid before generating a maze.")
        return
    for row in grid:
        for spot in row:
            if spot != start and spot != end:
                spot.reset()

    recursive_division(grid, 0, len(grid) - 1, 0, len(grid) - 1, start, end)

def recursive_division(grid, row_start, row_end, col_start, col_end, start, end):
    if row_end - row_start < 2 or col_end - col_start < 2:
        return

    horizontal = random.choice([True, False])

    if horizontal:
        row = random.randint(row_start + 1, row_end - 1)
        gap = random.randint(col_start, col_end)
        for col in range(col_start, col_end + 1):
            if col != gap and grid[row][col] != start and grid[row][col] != end:
                grid[row][col].make_barrier()

        recursive_division(grid, row_start, row - 1, col_start, col_end, start, end)
        recursive_division(grid, row + 1, row_end, col_start, col_end, start, end)
    else:

        col = random.randint(col_start + 1, col_end - 1)
        gap = random.randint(row_start, row_end)
        for row in range(row_start, row_end + 1):
            if row != gap and grid[row][col] != start and grid[row][col] != end:
                grid[row][col].make_barrier()

        recursive_division(grid, row_start, row_end, col_start, col - 1, start, end)
        recursive_division(grid, row_start, row_end, col + 1, col_end, start, end)

def display_error_panel(win, width):
    if not error_log:
        return

    panel_height = 100
    panel_y = width - panel_height

    pygame.draw.rect(win, GREY, (0, panel_y, width, panel_height))

    font = pygame.font.SysFont('Arial', 20)
    for i, message in enumerate(error_log[-3:]):
        text = font.render(message, True, RED)
        win.blit(text, (20, panel_y + 10 + i * 25))

    pygame.display.update()

def add_error(message):
    error_log.append(message)
    print(message)

def a_star_algorithm(draw, grid, start, end, heuristic_choice):
    global algorithm_run
    if algorithm_run:
        add_error("Please clear the grid before running another algorithm.")
        return False
    play_algorithm_music()
    global error_log
    error_log = []
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid for spot in row}


    if heuristic_choice == 'manhattan':
        f_score[start] = manhattan_distance(start.get_pos(), end.get_pos())
    elif heuristic_choice == 'euclidean':
        f_score[start] = euclidean_distance(start.get_pos(), end.get_pos())
    elif heuristic_choice == 'chebyshev':
        f_score[start] = chebyshev_distance(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw, start)
            end.make_end()
            algorithm_run = True
            play_instruction_music()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score

                if heuristic_choice == 'manhattan':
                    f_score[neighbor] = temp_g_score + manhattan_distance(neighbor.get_pos(), end.get_pos())
                elif heuristic_choice == 'euclidean':
                    f_score[neighbor] = temp_g_score + euclidean_distance(neighbor.get_pos(), end.get_pos())
                elif heuristic_choice == 'chebyshev':
                    f_score[neighbor] = temp_g_score + chebyshev_distance(neighbor.get_pos(), end.get_pos())

                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    print_no_path_found()
    algorithm_run = True
    play_instruction_music()
    return False


def bfs_algorithm(draw, grid, start, end): #Works the same as Dijkstra because all edges have the same weight (1)
    global algorithm_run
    if algorithm_run:
        add_error("Please clear the grid before running another algorithm.")
        return False
    play_algorithm_music()
    global error_log
    error_log = []
    queue = Queue()
    queue.put(start)
    came_from = {}
    visited = {spot: False for row in grid for spot in row}
    visited[start] = True
    while not queue.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        current = queue.get()
        if current == end:
            reconstruct_path(came_from, end, draw, start)
            end.make_end()
            algorithm_run = True
            play_instruction_music()
            return True
        for neighbor in current.neighbors:
            if not visited[neighbor]:
                queue.put(neighbor)
                came_from[neighbor] = current
                visited[neighbor] = True
                neighbor.make_open()
        draw()
        if current != start:
            current.make_closed()

    print_no_path_found()
    algorithm_run = True
    play_instruction_music()
    return False

def dfs_algorithm(draw, grid, start, end):
    global algorithm_run
    if algorithm_run:
        add_error("Please clear the grid before running another algorithm.")
        return False
    play_algorithm_music()
    global error_log
    error_log = []
    stack = LifoQueue()
    stack.put(start)
    came_from = {}
    visited = {spot: False for row in grid for spot in row}
    visited[start] = True
    while not stack.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        current = stack.get()
        if current == end:
            reconstruct_path(came_from, end, draw, start)
            end.make_end()
            algorithm_run = True
            play_instruction_music()
            return True
        for neighbor in current.neighbors:
            if not visited[neighbor]:
                stack.put(neighbor)
                came_from[neighbor] = current
                visited[neighbor] = True
                neighbor.make_open()
        draw()
        if current != start:
            current.make_closed()

    print_no_path_found()
    algorithm_run = True
    play_instruction_music()
    return False

def ucs_algorithm(draw, grid, start, end):
    global algorithm_run
    if algorithm_run:
        add_error("Please clear the grid before running another algorithm.")
        return False
    play_algorithm_music()
    global error_log
    error_log = []
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    open_set_hash = {start}
    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        current = open_set.get()[2]
        open_set_hash.remove(current)
        if current == end:
            reconstruct_path(came_from, end, draw, start)
            end.make_end()
            algorithm_run = True
            play_instruction_music()
            return True
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((g_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()
        draw()
        if current != start:
            current.make_closed()

    print_no_path_found()
    algorithm_run = True
    play_instruction_music()
    return False

def dijkstra_algorithm(draw, grid, start, end): #Works and the code is the same as UCS in this game because we care about shortest path from start to end not to all nodes(which is what Dijkstra does, UCS is a special variation of Dijkstra which looks for the shortest path to a single node)
                                                #If we would look for the shortest path to all nodes we would need to change the algorithm so that it would not stop when it reaches the end node end it would start with a priority queue with all nodes not just the start node like in UCS
    global algorithm_run
    if algorithm_run:
        add_error("Please clear the grid before running another algorithm.")
        return False
    play_algorithm_music()
    global error_log
    error_log = []
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    open_set_hash = {start}
    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        current = open_set.get()[2]
        open_set_hash.remove(current)
        if current == end:
            reconstruct_path(came_from, end, draw, start)
            end.make_end()
            algorithm_run = True
            play_instruction_music()
            return True
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((g_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()
        draw()
        if current != start:
            current.make_closed()

    print_no_path_found()
    algorithm_run = True
    play_instruction_music()
    return False

def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)

    return grid


def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


def save_grid(grid, start, end):
    configuration = {
        'start': start.get_pos(),
        'end': end.get_pos(),
        'barriers': [(spot.row, spot.col) for row in grid for spot in row if spot.is_barrier()]
    }
    with open('grid_configuration.pkl', 'wb') as f:
        pickle.dump(configuration, f)
    print("Configuration saved.")


def load_grid(grid):
    try:
        with open('grid_configuration.pkl', 'rb') as f:
            configuration = pickle.load(f)
        start_pos = configuration['start']
        end_pos = configuration['end']
        start = grid[start_pos[0]][start_pos[1]]
        end = grid[end_pos[0]][end_pos[1]]

        for row in grid:
            for spot in row:
                spot.reset()
        for barrier in configuration['barriers']:
            grid[barrier[0]][barrier[1]].make_barrier()
        start.make_start()
        end.make_end()
        global algorithm_run
        algorithm_run = False
        print("Configuration loaded.")
        return start, end
    except FileNotFoundError:
        add_error("No saved configuration found.")
        return None, None

def draw(win, grid, rows, width):
    win.fill(WHITE)
    for row in grid:
        for spot in row:
            spot.draw(win)
    draw_grid(win, rows, width)

    display_error_panel(win, width)

    pygame.display.update()



def show_instructions(win, width):
    play_instruction_music()
    win.fill(WHITE)

    title_text = "Welcome to the Path Finding Game!"
    title_surface = title_font.render(title_text, True, BLACK)
    win.blit(title_surface, (width // 2 - title_surface.get_width() // 2, 7))

    # Instruction sections
    sections = [
        ("Setup", [
            "First 2 Left Clicks: Set start and end points (orange and turquoise).",
            "Other Left Clicks: Set barriers (black).",
            "Right Click: Remove start, end, or barriers."
        ]),
        ("Heuristics & Algorithms", [
            "Press '1': Manhattan Heuristic for A* Algorithm",
            "Press '2': Euclidean Heuristic for A* Algorithm",
            "Press '3': Chebyshev Heuristic for A* Algorithm",
            "Press 'A': Run A* Algorithm",
            "Press 'B': Run BFS Algorithm",
            "Press 'D': Run DFS Algorithm",
            "Press 'U': Run UCS Algorithm",
            "Press 'I': Run Dijkstra's Algorithm"
        ]),
        ("Additional Controls", [
            "Press 'C': Clear the grid",
            "Press 'M': Generate a maze",
            "Click 'Start' to begin the game",
            "Press 'S': Save the current grid configuration",
            "Press 'L': Load the last saved grid configuration"
        ]),
        ("Rules:", [
            "You can only run one algorithm at a time. You need to clear the grid or load a configuration to run again.",
            "You need to clear the grid or load a configuration to be able to set up / remove points or barriers.",
            "You need to clear the grid or load a configuration to be able to generate a new maze.",
            "You need to select start and end points before running an algorithm.",
            "If you want to run A* Algorithm, you need to select a heuristic first."
        ])
    ]

    y_offset = 50
    for section_title, lines in sections:
        pygame.draw.rect(win, LIGHT_BLUE, (20, y_offset, width - 40, 35))
        section_header = section_font.render(section_title, True, BLACK)
        win.blit(section_header, (30, y_offset))
        y_offset += 30

        pygame.draw.rect(win, LIGHT_YELLOW, (20, y_offset, width - 40, len(lines) * 30 + 10))
        for line in lines:
            instruction_text = text_font.render(line, True, BLACK)
            win.blit(instruction_text, (30, y_offset + 5))
            y_offset += 30
        y_offset += 10

    button_width = 120
    button_height = 40
    button_x = (width - button_width) // 2
    button_y = y_offset + 10
    pygame.draw.rect(win, GREY, (button_x, button_y, button_width, button_height))
    start_text = section_font.render("Start", True, BLACK)
    win.blit(start_text, (button_x + 35, button_y + 5))

    pygame.display.update()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if button_x <= x <= button_x + button_width and button_y <= y <= button_y + button_height:
                    waiting = False

def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    if row < 0 or row >= rows or col < 0 or col >= rows:
        return None

    return row, col

def main(win, width):
    rows = 50
    grid = make_grid(rows, width)
    start = None
    end = None
    run = True
    global algorithm_run
    heuristic_choice = ""
    while run:
        draw(win, grid, rows, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:
                if algorithm_run:
                    add_error("Please clear the grid before setting up a new configuration.")
                    continue
                pos = pygame.mouse.get_pos()
                clicked_pos = get_clicked_pos(pos, rows, width)
                if clicked_pos is not None:
                    row, col = clicked_pos
                    spot = grid[row][col]
                    if not start and spot != end:
                        start = spot
                        start.make_start()
                    elif not end and spot != start:
                        end = spot
                        end.make_end()
                    elif spot != end and spot != start:
                        spot.make_barrier()

            elif pygame.mouse.get_pressed()[2]:
                if algorithm_run:
                    add_error("Please clear the grid before setting up a new configuration.")
                    continue
                pos = pygame.mouse.get_pos()
                clicked_pos = get_clicked_pos(pos, rows, width)
                if clicked_pos is not None:
                    row, col = clicked_pos
                    spot = grid[row][col]
                    spot.reset()
                    if spot == start:
                        start = None
                    elif spot == end:
                        end = None

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_1:
                    heuristic_choice = 'manhattan'
                    print("Selected Manhattan Heuristic")
                elif event.key == pygame.K_2:
                    heuristic_choice = 'euclidean'
                    print("Selected Euclidean Heuristic")
                elif event.key == pygame.K_3:
                    heuristic_choice = 'chebyshev'
                    print("Selected Chebyshev Heuristic")

                if event.key == pygame.K_a and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    print("Running A* Algorithm")
                    if heuristic_choice == "":
                        add_error("Please select a heuristic first")
                    else:
                        a_star_algorithm(lambda: draw(win, grid, rows, width), grid, start, end, heuristic_choice)
                        heuristic_choice = ""
                elif event.key == pygame.K_a and (not start or not end):
                    add_error("Please select start and end points first")

                if event.key == pygame.K_b and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    print("Running BFS Algorithm")
                    bfs_algorithm(lambda: draw(win, grid, rows, width), grid, start, end)
                elif event.key == pygame.K_b and (not start or not end):
                    add_error("Please select start and end points first")

                if event.key == pygame.K_d and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    print("Running DFS Algorithm")
                    dfs_algorithm(lambda: draw(win, grid, rows, width), grid, start, end)
                elif event.key == pygame.K_d and (not start or not end):
                    print("Please select start and end points first")

                if event.key == pygame.K_u and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    print("Running UCS Algorithm")
                    ucs_algorithm(lambda: draw(win, grid, rows, width), grid, start, end)
                elif event.key == pygame.K_u and (not start or not end):
                    add_error("Please select start and end points first")

                if event.key == pygame.K_i and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    print("Running Dijkstra's Algorithm")
                    dijkstra_algorithm(lambda: draw(win, grid, rows, width), grid, start, end)
                elif event.key == pygame.K_i and (not start or not end):
                    add_error("Please select start and end points first")

                if event.key == pygame.K_c:
                    start = None
                    end = None
                    print("Clearing the grid")
                    algorithm_run = False
                    grid = make_grid(rows, width)

                if event.key == pygame.K_m:
                    print("Generating a maze")
                    generate_maze(grid, start, end)

                if event.key == pygame.K_s:
                    save_grid(grid, start, end)
                if event.key == pygame.K_l:
                    start, end = load_grid(grid)
    pygame.quit()
show_instructions(WIN, WIDTH)
main(WIN, WIDTH)