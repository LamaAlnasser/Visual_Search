from queue import PriorityQueue, Queue, LifoQueue
import pygame
import random
import pickle

pygame.init()
pygame.mixer.init()

WIDTH = 900
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Save the Princess | SWE485")

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

def manhattan_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

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
    print(f"The visualized grid shows the explored nodes and the final path.\n")
    path_str = " -> ".join([f"{pos}" for pos in [start] + path])
    print(f"The shortest path is {length} Such path is {path_str}")
    # print(f"Length: {length} steps")


def print_no_path_found():
    print("No solution is found! We need to eliminate more obstacles to find such a walk.")


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
            draw()
            algorithm_run = False  # Set to False once done
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score

                if heuristic_choice == 'manhattan':
                    f_score[neighbor] = temp_g_score + manhattan_distance(neighbor.get_pos(), end.get_pos())
              
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
    return False


def bfs_algorithm(draw, grid, start, end): #all edges have the same weight (1)
    global algorithm_run
    if algorithm_run:
        add_error("Please clear the grid before running another algorithm.")
        return False
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
            draw()
            algorithm_run = False  # Set to False once done
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
    height = WIDTH  # Assuming height is set to match the width for a square window

    win.fill(WHITE)
    title_text = "Welcome to Rescuing the Princess Game!"
    # Consider reducing the font size if necessary
    title_surface = title_font.render(title_text, True, BLACK)
    win.blit(title_surface, (width // 2 - title_surface.get_width() // 2, 10))

    # Instruction sections with potentially smaller fonts or tighter spacing
    sections = [
        ("Setup", [
            "First Left Click: Set the start point (orange).",
            "Second Left Click: Set the end point (turquoise).",
            "Subsequent Left Clicks: Place barriers (black).",
            "Right Click on Barrier: Remove a barrier."
        ]),
        ("Algorithms", [
            "Press 'A': Run A* Algorithm - finds a path using heuristics.",
            "Press 'B': Run BFS Algorithm - explores all possible paths."
        ]),
        ("Controls", [
            "Press 'C': Clear the entire grid to start over.",
            "Press 'S': Save the current grid setup.",
            "Press 'L': Load a previously saved grid setup."
        ]),
        ("Rules", [
            "Only one algorithm can run at a time.",
            "Set start and end points before running an algorithm.",
            "Clear the grid or load a setup to run algorithms repeatedly."
        ])
    ]

    y_offset = 40  # Smaller starting offset
    for section_title, lines in sections:
        section_header = section_font.render(section_title, True, BLACK)
        win.blit(section_header, (30, y_offset))
        y_offset += section_header.get_height() + 2  # Smaller vertical padding

        for line in lines:
            instruction_text = text_font.render(line, True, BLACK)
            win.blit(instruction_text, (50, y_offset))
            y_offset += instruction_text.get_height() + 2
        y_offset += 10

    # Position the button right below the last line of instructions
    button_width = 120
    button_height = 40
    button_x = (width - button_width) // 2
    button_y = min(height - button_height - 20, y_offset + 10)  # Ensure it's within the window

    # Draw the Start button
    pygame.draw.rect(win, GREY, (button_x, button_y, button_width, button_height))
    start_text = section_font.render("Start", True, BLACK)
    win.blit(start_text, (button_x + (button_width - start_text.get_width()) // 2, button_y + (button_height - start_text.get_height()) // 2))

    pygame.display.update()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False  # Game should quit
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if button_x <= x <= button_x + button_width and button_y <= y <= button_y + button_height:
                    waiting = False  # User clicked Start

    return True  # Game should continue


def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    if row < 0 or row >= rows or col < 0 or col >= rows:
        return None

    return row, col

def main(win, width):
    rows = 10
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
                    if spot.is_barrier():
                     spot.reset()  # Allow barrier removal
                    if spot == start:
                        start = None # Unset start if it's the start node
                    elif spot == end:
                        end = None  # Unset end if it's the end node

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_a and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    print("--- Running A* Algorithm ---")
                    a_star_algorithm(lambda: draw(win, grid, rows, width), grid, start, end, "manhattan")
                elif event.key == pygame.K_a and (not start or not end):
                    add_error("Please select start and end nodes first!")

                if event.key == pygame.K_b and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    print("--- Running BFS Algorithm ---")
                    bfs_algorithm(lambda: draw(win, grid, rows, width), grid, start, end)
                elif event.key == pygame.K_b and (not start or not end):
                    add_error("Please select start and end nodes first!")


                if event.key == pygame.K_c:
                    start = None
                    end = None
                    print("Clearing the grid")
                    algorithm_run = False
                    grid = make_grid(rows, width)
               

                if event.key == pygame.K_s:
                    save_grid(grid, start, end)
                if event.key == pygame.K_l:
                    start, end = load_grid(grid)
    pygame.quit()
show_instructions(WIN, WIDTH)
main(WIN, WIDTH)