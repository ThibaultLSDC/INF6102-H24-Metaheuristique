from copy import deepcopy
import random as rd
import multiprocessing as mp
from functools import partial

from utils import *


SELECT_PARALLEL = False
AVAILABLE_PARALLEL = False
CPU_COUNT = mp.cpu_count()


class CustomWall(Wall):

    """ You are completely free to extend classes defined in utils,
        this might prove useful or enhance code readability
    """

    def __init__(self, w, h):
        super().__init__(w, h)
        self.pieces = []
        self.board = [[0] * h for _ in range(w)]
        self.right_space = [[i for i in range(h, 0, -1)] for _ in range(w)]
        self.upper_space = [[i]*h for i in range(w, 0, -1)]
    
    def add_piece(self, p_id, x, y, w, h):
        if self._is_available(x, y, w, h):
            piece = {
                'id': p_id,
                'x': x,
                'y': y,
                'width': w,
                'height': h
            }
            self.pieces.append(piece)
            for i in range(x, x + w):
                for j in range(y, y + h):
                    self.board[i][j] = p_id
                    self.right_space[i][j] = -1
                    self.upper_space[i][j] = -1
            
            for xx in range(x, x+w):
                for j in range(y-1, -1, -1):
                    if self.right_space[xx][j] == -1:
                        break
                    self.right_space[xx][j] -= h+1

            for yy in range(y, y+h):
                for i in range(x-1, -1, -1):
                    if self.upper_space[i][yy] == -1:
                        break
                    self.upper_space[i][yy] -= w+1
            

            return True
        else:
            return False
    
    def _is_in(self, x, y, w, h):
        # Check if the piece is inside the wall
        if x < 0 or y < 0 or x + w > self.width() or y + h > self.height():
            return False
        return True
    
    def _is_available(self, x, y, w, h):
        # Check if the piece has room
        if any([self.right_space[i][y] < h for i in range(x, x + w)]) \
        or any([self.upper_space[x][j] < w for j in range(y, y + h)]):
            return False
        return True
    
    def _corners(self, x, y, w, h):
        """
        Returns True if one of the corners of the piece is in a corner of the wall or an other piece
        """
        X = x+w-1
        Y = y+h-1

        if not (self.board[x][y] == 0 \
                and self.board[X][y] == 0 \
                and self.board[x][Y] == 0 \
                and self.board[X][Y] == 0):
            return False

        # Wall corners
        if (x == 0 or X == self.width()) and (y == 0 or Y == self.height()):
            return True

        # Wall borders
        if x == 0 or X == self.width() - 1:
            if y > 0 and (self.board[x][y - 1] != 0 or self.board[X][y - 1] != 0):
                return True
            if Y < self.height() - 1 and (self.board[x][Y + 1] != 0 or self.board[X][Y + 1] != 0):
                return True
            return False
        if y == 0 or Y == self.height() - 1:
            if x > 0 and (self.board[x - 1][y] != 0 or self.board[x - 1][Y] != 0):
                return True
            if X < self.width() - 1 and (self.board[X + 1][y] != 0 or self.board[X + 1][Y] != 0):
                return True
            return False

        # Piece corners
        if self.board[x - 1][y] != 0:
            if self.board[x][y - 1] != 0 or self.board[x][y + 1] != 0:
                return True
        if self.board[X + 1][y] != 0:
            if self.board[X][y-1] != 0 or self.board[X][y+1] != 0:
                return True
        if self.board[x][Y + 1] != 0:
            if self.board[x - 1][Y] != 0 or self.board[x + 1][Y] != 0:
                return True
        if self.board[X][Y + 1] != 0:
            if self.board[X + 1][Y] != 0 or self.board[X - 1][Y] != 0:
                return True
        
        return False
    
    def get_available(self, w, h):
        available = []
        for x in range(self.width()):
            for y in range(self.height()):
                if self._is_in(x, y, w, h):
                    if self._corners(x, y, w, h):
                        if self._is_available(x, y, w, h):
                            available.append((x, y))
        return available
    
    def getX_available(self, available, w, h, x):
        available.extend([(x, y) for y in range(self.height()) if self._is_available(x, y, w, h) and self._corners(x, y, w, h)])
    
    def parallel_get_available(self, w, h):
        available = []
        pool = mp.Pool(mp.cpu_count())
        func = partial(self.getX_available, available, w, h)
        pool.map(func, range(self.width()))
        pool.close()
        pool.join()
        return available


def solve(instance: Instance) -> Solution:
    """Write your code here

    Args:
        instance (Instance): An Instance object containing all you need to solve the problem

    Returns:
        Solution: A solution object initialized with 
                  a list of tuples of the form (<artipiece_id>, <wall_id>, <x_pos>, <y_pos>)
    """

    if instance.wall.width() > 50 or instance.wall.height() > 50:
        global SELECT_PARALLEL
        global AVAILABLE_PARALLEL
        SELECT_PARALLEL = True
        AVAILABLE_PARALLEL = True
    if instance.wall.width() > 1000 or instance.wall.height() > 1000:
        global CPU_COUNT
        CPU_COUNT = 1

    current = [(i,i,0,0) for i in instance.artpieces_dict.keys()]
    total_walls = len(set(w[1] for w in current))
    for restart in range(1):
        print(f"Restart {restart}")

        remaining_pieces = [{'id': i, 'width': p.width(), 'height': p.height()}
                            for i, p in instance.artpieces_dict.items()]

        rd.shuffle(remaining_pieces)
        # sort by largest piece
        # init_walls.sort(key=lambda x: x.pieces[-1]['width'] * x.pieces[-1]['height'], reverse=True)

        walls = []
        walls.append(make_wall(instance.wall.width(), instance.wall.height(), remaining_pieces.pop(0)))
        while remaining_pieces:
            print(f"Remaining: {len(remaining_pieces)}, walls: {len(walls)}")
            if not reduce_last(walls, remaining_pieces):
                rd.shuffle(remaining_pieces)
                # sort by largest piece
                # init_walls.sort(key=lambda x: x.pieces[-1]['width'] * x.pieces[-1]['height'], reverse=True)                
                walls.append(make_wall(instance.wall.width(), instance.wall.height(), remaining_pieces.pop(0)))

        solution = [(p['id'], i, p['x'], p['y']) for i, w in enumerate(walls) for p in w.pieces]
        wall_count = len(set(w[1] for w in solution))

        if solution == current:
            print(f"Solution did not change, restarting")
            continue

        if wall_count < total_walls:
            print(f"New solution found: {wall_count} < {total_walls}")
            current = solution
            total_walls = wall_count

    return Solution(current)


def make_wall(w, h, piece):
    wall = CustomWall(w, h)
    wall.add_piece(piece['id'], 0, 0, piece['width'], piece['height'])
    return wall


def reduce_last(walls, remaining_pieces):
    # chosen_walls = rd.choices(list(enumerate(init_walls)), k=min(len(init_walls), len(init_walls)))
    chosen_pieces = [(i, p) for i, p in enumerate(remaining_pieces)]
    # sort by largest piece
    chosen_pieces.sort(key=lambda x: x[1]['width'] * x[1]['height'], reverse=True)
    chosen_pieces = chosen_pieces[:min(len(remaining_pieces), len(remaining_pieces))]

    artpieces = [(i, p) for i, p in chosen_pieces]

    if AVAILABLE_PARALLEL:
        pool = mp.Pool(mp.cpu_count())
        input_data = [(walls, idx, init_idx, artpiece) for init_idx, artpiece in artpieces for idx in range(len(walls))]
        available = pool.starmap(parallel_get_available, input_data)
        pool.close()
        pool.join()
        available = [a for sublist in available for a in sublist]
    else:
        available = []    
        for init_index, artpiece in artpieces:
            # dict w/ idx: (x, y)
            tmp = {j: w.get_available(artpiece['width'], artpiece['height']) for j, w in enumerate(walls)}
            # list of tuples (idx, init_idx, artpiece_id, x, y, width, height)
            available += [(j, init_index, artpiece['id'], x, y, artpiece['width'], artpiece['height']) for j in tmp for x, y in tmp[j]]


    if available:
        if SELECT_PARALLEL:
            selected = parallel_select(walls, available)
        else:
            selected = select(walls, available)
        walls[selected[0]].add_piece(*selected[2:])
        # pop the selected piece from init_walls
        remaining_pieces.pop(selected[1])
        return True
    return False


def parallel_get_available(final_walls, idx, init_idx, artpiece):
    available = final_walls[idx].get_available(artpiece['width'], artpiece['height'])
    res = [(idx, init_idx, artpiece['id'], *a, artpiece['width'], artpiece['height'])
           for a in available]
    return res


def select(final_walls, available):
    # sort by largest rectangle
    largest = []
    for a in available:
        wall = deepcopy(final_walls[a[0]])
        wall.add_piece(*a[2:])
        # list of tuples (largest rectangle, (idx, init_idx, artpiece_id, x, y, width, height))
        largest.append((largest_rectangle(wall), a))
    # sort wrt largest rectangle
    largest.sort(key=lambda x: (x[0], -x[1][3], -x[1][4]), reverse=True)
    # list of tuples (idx, init_idx, artpiece_id, x, y, width, height)
    sorted_positions = [a for _, a in largest]
    # tuple (idx, init_idx, artpiece_id, x, y, width, height)
    return sorted_positions[0]


def parallel_select(final_walls, available):
    # sort by largest rectangle
    pool = mp.Pool(mp.cpu_count())
    input_data = [(final_walls, a) for a in available]
    largest = pool.starmap(get_largest, input_data)
    pool.close()
    pool.join()
    largest.sort(key=lambda x: (x[0], -x[1][3], -x[1][4]), reverse=True)
    # list of tuples (idx, init_idx, artpiece_id, x, y, width, height)
    sorted_positions = [a for _, a in largest]
    # tuple (idx, init_idx, artpiece_id, x, y, width, height)
    return sorted_positions[0]


def largest_rectangle(wall: CustomWall):
    max_area = 0
    hist = [0] * wall.width()
    for i in range(wall.height()):
        for j in range(wall.width()):
            hist[j] = hist[j] + 1 if wall.board[j][i] == 0 else 0
        stack = []
        for j, h in enumerate(hist):
            while stack and hist[stack[-1]] > h:
                height = hist[stack.pop()]
                width = j if not stack else j - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(j)
        while stack:
            height = hist[stack.pop()]
            width = wall.width() if not stack else wall.width() - stack[-1] - 1
            max_area = max(max_area, height * width)
    return max_area


def get_largest(final_walls, available):
    wall = deepcopy(final_walls[available[0]])
    wall.add_piece(*available[2:])
    return largest_rectangle(wall), available