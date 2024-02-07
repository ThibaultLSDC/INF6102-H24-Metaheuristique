from copy import deepcopy
import random as rd
import multiprocessing as mp
from functools import partial
import time

from utils import *


AVAILABLE_PARALLEL = False
SELECT_PARALLEL = False
CPU_COUNT = mp.cpu_count()

AVAILABLE_DURATIONS = []
ORDER_DURATIONS = []

RESTART = 1

SELECT_MODE = {
    1: 'largest',
    2: 'random',
    3: 'topk',
    4: 'all'}
NEW_PIECE_SELECT = 2 # 1: largest, 2: random (all), 3: topk (random in top k)
NEW_PIECE_TOPK = 10

REDUCE_SELECT = 2 # 2: random(random k), 3: topk (not random), 4: all
REDUCE_K = 5

ADD_SELECT = 1 # 1: largest, 3: topk (random in top k)
ADD_K = 10

METRIC_MODE = {
    1: 'largest_rectangle',
    2: 'largest_fill',
    3: 'largest_both',}
METRIC = 3 # 1: largest_rectangle, 2: largest_fill, 3: largest_both

P_NORM = 'last' # 'inf', 'last', or p value


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

        self.largest_rectangle = self.width() * self.height()
        self.largest_fill = 0
        
    @property
    def metric(self):
        if METRIC_MODE[METRIC] == 'largest_rectangle':
            return self.largest_rectangle
        elif METRIC_MODE[METRIC] == 'largest_fill':
            return self.largest_fill
        elif METRIC_MODE[METRIC] == 'largest_both':
            return mean(self.largest_rectangle, self.largest_fill)
        
    @property
    def empty_area(self):
        return sum([sum([1 for j in i if j == 0]) for i in self.board])
    
    @property
    def fill_area(self):
        return sum([sum([1 for j in i if j != 0]) for i in self.board])
    
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
                for dist, j in enumerate(range(y-1, -1, -1)):
                    if self.right_space[xx][j] == -1:
                        break
                    self.right_space[xx][j] = dist+1

            for yy in range(y, y+h):
                for dist, i in enumerate(range(x-1, -1, -1)):
                    if self.upper_space[i][yy] == -1:
                        break
                    self.upper_space[i][yy] = dist+1
            self.largest_rectangle, self.largest_fill = largest_both(self)
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

    # if instance.wall.width() > 50 or instance.wall.height() > 50:
    #     global SELECT_PARALLEL
    #     global AVAILABLE_PARALLEL
    #     SELECT_PARALLEL = True
    #     AVAILABLE_PARALLEL = True
    # if instance.wall.width() > 1000 or instance.wall.height() > 1000:
    #     global CPU_COUNT
    #     CPU_COUNT = 1

    current = [(i,i,0,0) for i in instance.artpieces_dict.keys()]
    total_walls = len(set(w[1] for w in current))
    for restart in range(RESTART):
        print(f"Restart {restart}")

        remaining_pieces = [{'id': i, 'width': p.width(), 'height': p.height()}
                            for i, p in instance.artpieces_dict.items()]

        walls = []
        add_wall(instance, walls, remaining_pieces)
        broken = 0
        while remaining_pieces:
            print(f"Remaining: {len(remaining_pieces)}, walls: {len(walls)}")
            if not reduce_last(walls, remaining_pieces):
                add_wall(instance, walls, remaining_pieces)
            if not remaining_pieces and not broken > 100:
                broken += 1
                smallest_area = float('inf')
                idx = -1
                for i, w in enumerate(walls):
                    if w.fill_area < smallest_area:
                        smallest_area = w.fill_area
                        idx = i
                
                if sum([w.largest_rectangle for i, w in enumerate(walls) if i != idx]) > .8*smallest_area:

                # if any([w.empty_area > w.fill_area for w in walls]):
                # if len(walls) > 8:
                    print(f"Saving {broken} walls")
                    # idx = max(range(len(walls)), key=lambda x: walls[x].empty_area)
                    # kept = walls.pop(idx)
                    # while walls:
                    #     break_wall(walls, remaining_pieces, 0)
                    # walls.append(kept)

                    walls.sort(key=lambda x: x.empty_area)
                    for i in range(len(walls)):
                        if i <= broken:
                            continue
                        if rd.random() < .5:
                            idx = rd.randint(0, len(walls)-1)
                        else:
                            idx = -1
                        break_wall(walls, remaining_pieces, idx)
                else:
                    print(f"Restarting")
                    
        
        print(f"Broke {broken} walls")
        print([(w.empty_area, w.fill_area) for w in walls])

        solution = [(p['id'], i, p['x'], p['y']) for i, w in enumerate(walls) for p in w.pieces]
        wall_count = len(set(w[1] for w in solution))

        if solution == current:
            print(f"Solution did not change, restarting")
            continue

        if wall_count < total_walls:
            print(f"New solution found: {wall_count} < {total_walls}")
            current = solution
            total_walls = wall_count
        else:
            print(f"Solution had more walls: {wall_count} >= {total_walls}")
    
    print(f"Average available duration: {sum(AVAILABLE_DURATIONS) / len(AVAILABLE_DURATIONS)}")
    print(f"Average order duration: {sum(ORDER_DURATIONS) / len(ORDER_DURATIONS)}")
    return Solution(current)


def make_wall(w, h, piece):
    wall = CustomWall(w, h)
    wall.add_piece(piece['id'], 0, 0, piece['width'], piece['height'])
    return wall


def add_wall(instance, walls: list, remaining_pieces: list):
    if SELECT_MODE[NEW_PIECE_SELECT] == 'largest':
        remaining_pieces.sort(key=lambda x: x['width'] * x['height'], reverse=True)
        res = remaining_pieces.pop(0)
    elif SELECT_MODE[NEW_PIECE_SELECT] == 'random':
        rd.shuffle(remaining_pieces)
        res = remaining_pieces.pop(0)
    elif SELECT_MODE[NEW_PIECE_SELECT] == 'topk':
        remaining_pieces.sort(key=lambda x: x['width'] * x['height'], reverse=True)
        res = remaining_pieces.pop(rd.randint(0, min(NEW_PIECE_TOPK, len(remaining_pieces)) - 1))
    walls.append(make_wall(instance.wall.width(), instance.wall.height(), res))


def break_wall(walls, remaining_pieces, idx):
    wall = walls.pop(idx)
    for p in wall.pieces:
        remaining_pieces.append({'id': p['id'], 'width': p['width'], 'height': p['height']})
    return wall


def reduce_last(walls, remaining_pieces):
    if SELECT_MODE[REDUCE_SELECT] == 'all':
        chosen_pieces = [(i, p) for i, p in enumerate(remaining_pieces)]
    elif SELECT_MODE[REDUCE_SELECT] == 'topk':
        chosen_pieces = [(i, p) for i, p in enumerate(remaining_pieces)]
        chosen_pieces.sort(key=lambda x: x[1]['width'] * x[1]['height'], reverse=True)
        chosen_pieces = chosen_pieces[:min(REDUCE_K, len(remaining_pieces))]
    elif SELECT_MODE[REDUCE_SELECT] == 'random':
        chosen_pieces = rd.choices(list(enumerate(remaining_pieces)), k=min(REDUCE_K, len(remaining_pieces)))

    artpieces = [(i, p) for i, p in chosen_pieces]

    if AVAILABLE_PARALLEL:
        top = time.time()
        pool = mp.Pool(mp.cpu_count())
        input_data = [(walls, idx, init_idx, artpiece) for init_idx, artpiece in artpieces for idx in range(len(walls))]
        available = pool.starmap(parallel_get_available, input_data)
        pool.close()
        pool.join()
        available = [a for sublist in available for a in sublist]
        AVAILABLE_DURATIONS.append(time.time() - top)
    else:
        top = time.time()
        available = []    
        for init_index, artpiece in artpieces:
            # dict w/ idx: (x, y)
            tmp = {j: w.get_available(artpiece['width'], artpiece['height']) for j, w in enumerate(walls)}
            # list of tuples (idx, init_idx, artpiece_id, x, y, width, height)
            available += [(j, init_index, artpiece['id'], x, y, artpiece['width'], artpiece['height']) for j in tmp for x, y in tmp[j]]
        AVAILABLE_DURATIONS.append(time.time() - top)

    if available:
        if SELECT_PARALLEL:
            top = time.time()
            ordered = parallel_order(walls, available)
            ORDER_DURATIONS.append(time.time() - top)
        else:
            top = time.time()
            ordered = order(walls, available)
            ORDER_DURATIONS.append(time.time() - top)

        if SELECT_MODE[ADD_SELECT] == 'largest':
            selected = ordered[0]
        elif SELECT_MODE[ADD_SELECT] == 'topk':
            selected = rd.choice(ordered[:min(ADD_K, len(ordered))])

        # print(f"Selected: {selected[2]}")
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


def order(final_walls, available):
    # sort by largest rectangle
    metrics = []
    for a in available:
        wall = deepcopy(final_walls[a[0]])
        wall.add_piece(*a[2:])
        metric = []
        for i, w in enumerate(final_walls):
            if i != a[0]:
                metric.append(w.metric)
        # list of tuples (metric, (idx, init_idx, artpiece_id, x, y, width, height))
        metric.append(wall.metric)
        metrics.append((p_norm(metric, p=P_NORM), a))
    # sort wrt largest rectangle
    metrics.sort(key=lambda x: (x[0], -x[1][3], -x[1][4]), reverse=True)
    # list of tuples (idx, init_idx, artpiece_id, x, y, width, height)
    sorted_positions = [a for _, a in metrics]
    # tuple (idx, init_idx, artpiece_id, x, y, width, height)
    # rd from top 50% of largest rectangles
    return sorted_positions


def parallel_order(final_walls, available):
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
    # rd from top 10% of largest rectangles
    return sorted_positions


def get_largest(final_walls, available):
    wall = deepcopy(final_walls[available[0]])
    largest = []
    for w in final_walls:
        if w != final_walls[available[0]]:
            largest.append(w.metric)
    wall.add_piece(*available[2:])
    largest.append(wall.metric)
    return p_norm(largest, p=P_NORM), available


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


def largest_fill(wall: CustomWall):
    """
    Same as largest rectangle but with filled areas
    """
    max_area = 0
    hist = [0] * wall.width()
    for i in range(wall.height()):
        for j in range(wall.width()):
            hist[j] = hist[j] + 1 if wall.board[j][i] != 0 else 0
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


def largest_both(wall: CustomWall):
    max_area_empty = 0
    hist_empty = [0] * wall.width()
    max_area_fill = 0
    hist_fill = [0] * wall.width()
    for i in range(wall.height()):
        for j in range(wall.width()):
            hist_empty[j] = hist_empty[j] + 1 if wall.board[j][i] == 0 else 0
            hist_fill[j] = hist_fill[j] + 1 if wall.board[j][i] != 0 else 0
        stack_empty = []
        stack_fill = []
        for j, h in enumerate(hist_empty):
            while stack_empty and hist_empty[stack_empty[-1]] > h:
                height = hist_empty[stack_empty.pop()]
                width = j if not stack_empty else j - stack_empty[-1] - 1
                max_area_empty = max(max_area_empty, height * width)
            stack_empty.append(j)
        for j, h in enumerate(hist_fill):
            while stack_fill and hist_fill[stack_fill[-1]] > h:
                height = hist_fill[stack_fill.pop()]
                width = j if not stack_fill else j - stack_fill[-1] - 1
                max_area_fill = max(max_area_fill, height * width)
            stack_fill.append(j)
        while stack_empty:
            height = hist_empty[stack_empty.pop()]
            width = wall.width() if not stack_empty else wall.width() - stack_empty[-1] - 1
            max_area_empty = max(max_area_empty, height * width)
        while stack_fill:
            height = hist_fill[stack_fill.pop()]
            width = wall.width() if not stack_fill else wall.width() - stack_fill[-1] - 1
            max_area_fill = max(max_area_fill, height * width)
    return max_area_empty, max_area_fill


def p_norm(l, p='inf'):
    if p == 'inf':
        return max(l)
    if p == 'last':
        return l[-1]
    return sum([x**p for x in l])**(1/p)


def mean(x, y):
    return (x + y) / 2