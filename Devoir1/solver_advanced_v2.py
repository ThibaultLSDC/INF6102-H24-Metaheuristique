from copy import deepcopy
import random as rd

from utils import *


class CustomWall(Wall):

    """ You are completely free to extend classes defined in utils,
        this might prove useful or enhance code readability
    """

    def __init__(self, w, h):
        super().__init__(w, h)
        self.pieces = []
        self.board = [[0] * h for _ in range(w)]
    
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
            return True
        else:
            return False
    
    def _is_available(self, x, y, w, h):
        # Check if the piece is inside the wall
        if x < 0 or y < 0 or x + w > self.width() or y + h > self.height():
            return False

        # Check if the piece overlaps with another piece
        for i in range(x, x + w):
            for j in range(y, y + h):
                if self.board[i][j] != 0:
                    return False
        return True
    
    def _corners(self, x, y, w, h):
        """
        Returns True if one of the corners of the piece is in a corner of the wall or an other piece
        """
        X = x+w-1
        Y = y+h-1

        assert self.board[x][y] == 0
        assert self.board[X][y] == 0
        assert self.board[x][Y] == 0
        assert self.board[X][Y] == 0

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
                if self._is_available(x, y, w, h):
                    if self._corners(x, y, w, h):
                        available.append((x, y))
        return available


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


def solve(instance: Instance) -> Solution:
    """Write your code here

    Args:
        instance (Instance): An Instance object containing all you need to solve the problem

    Returns:
        Solution: A solution object initialized with 
                  a list of tuples of the form (<artipiece_id>, <wall_id>, <x_pos>, <y_pos>)
    """

    current = [(i,i,0,0) for i in instance.artpieces_dict.keys()]
    total_walls = len(set(w[1] for w in current))
    for restart in range(10):
        print(f"Restart {restart}")

        init_walls = []
        for i in range(1, instance.n+1):
            init_walls.append(CustomWall(instance.wall.width(), instance.wall.height()))
            init_walls[-1].add_piece(
                i, 0, 0, instance.artpieces_dict[i].width(), instance.artpieces_dict[i].height())

        rd.shuffle(init_walls)

        final_walls = []

        while init_walls:
            print(f"Remaining walls: {len(init_walls)}, final walls: {len(final_walls)}")
            if not reduce_last(final_walls, init_walls):
                final_walls.append(init_walls.pop(0))

        solution = [(p['id'], i, p['x'], p['y']) for i, w in enumerate(final_walls) for p in w.pieces]
        wall_count = len(set(w[1] for w in solution))

        if solution == current:
            print(f"Solution did not change, restarting")
            continue

        if wall_count < total_walls:
            print(f"New solution found: {wall_count} < {total_walls}")
            current = solution
            total_walls = wall_count

    return Solution(current)


def reduce_last(final_walls, init_walls):
    if len(final_walls) == 0:
        final_walls.append(init_walls.pop())

    artpieces = [(i, w.pieces[-1]) for i, w in rd.choices(list(enumerate(init_walls)), k=min(1, len(init_walls)))]
    available = []
    for init_index, artpiece in artpieces:
        # dict w/ idx: (x, y)
        tmp = {j: w.get_available(artpiece['width'], artpiece['height']) for j, w in enumerate(final_walls)}
        # list of tuples (idx, init_idx, artpiece_id, x, y, width, height)
        available += [(j, init_index, artpiece['id'], x, y, artpiece['width'], artpiece['height']) for j in tmp for x, y in tmp[j]]

    if available:
        # sort by largest rectangle
        largest = []
        for a in available:
            wall = deepcopy(final_walls[a[0]])
            wall.add_piece(*a[2:])
            # list of tuples (largest rectangle, (idx, init_idx, artpiece_id, x, y, width, height))
            largest.append((largest_rectangle(wall), a))
            wall.pieces.pop()
        # sort wrt largest rectangle
        largest.sort(key=lambda x: x[0], reverse=True)
        # list of tuples (idx, init_idx, artpiece_id, x, y, width, height)
        available = [a for _, a in largest]
        # tuple (idx, init_idx, artpiece_id, x, y, width, height)
        selected = select(available)
        final_walls[selected[0]].add_piece(*selected[2:])
        # pop the selected piece from init_walls
        init_walls.pop(selected[1])
        return True
    return False


def select(elements):
    return elements[0]