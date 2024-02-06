import random as rd

from utils import *


class CustomWall(Wall):

    """ You are completely free to extend classes defined in utils,
        this might prove useful or enhance code readability
    """

    def __init__(self, w, h):
        super().__init__(w, h)
        self.pieces = []

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
            return True
        else:
            return False

    def _is_available(self, x, y, w, h):
        # Check if the piece is inside the wall
        if x < 0 or y < 0 or x + w > self.width() or y + h > self.height():
            return False

        # Check if the piece overlaps with another piece
        tmp = {'id':666, 'x':x, 'y':y, 'width':w, 'height':h}
        pieces = sorted(self.pieces + [tmp], key=lambda x: (x['y'], x['x']))
        for i, p in enumerate(pieces):
            for j, q in enumerate(pieces[i+1:]):
                if (p['x'] < q['x'] + q['width'] and
                        p['x'] + p['width'] > q['x'] and
                        p['y'] < q['y'] + q['height'] and
                        p['y'] + p['height'] > q['y']):
                    return False
        return True
    
    def get_available(self, w, h):
        available = []
        for x in range(self.width()):
            for y in range(self.height()):
                if self._is_available(x, y, w, h):
                    available.append((x, y))
        return available


def solve(instance: Instance) -> Solution:
    """Write your code here

    Args:
        instance (Instance): An Instance object containing all you need to solve the problem

    Returns:
        Solution: A solution object initialized with 
                  a list of tuples of the form (<artipiece_id>, <wall_id>, <x_pos>, <y_pos>)
    """

    current = [(i,i,0,0) for i in instance.artpieces_dict.keys()]
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
        
        if solution == current:
            print(f"Solution did not change, restarting")
            continue
        
        if len(solution) < len(current):
            print(f"New solution found: {len(solution)} < {len(current)}")
            current = solution

    return Solution(solution)


def reduce_last(final_walls, init_walls):
    if len(final_walls) == 0:
        final_walls.append(init_walls.pop(1))
    
    artpieces = [(i, w.pieces[-1]) for i, w in enumerate(init_walls)]
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
            final_walls[a[0]].add_piece(*a[2:])
            # list of tuples (largest rectangle, (idx, init_idx, artpiece_id, x, y, width, height))
            largest.append((largest_rectangle(final_walls[a[0]]), a))
            final_walls[a[0]].pieces.pop()
        # sort available wrt largest rectangle
        largest.sort(key=lambda x: x[0], reverse=True)
        # list of tuples (idx, init_idx, artpiece_id, x, y, width, height)
        available = [a for _, a in largest]
        # tuple (idx, init_idx, artpiece_id, x, y, width, height)
        selected = select(available)
        final_walls[selected[0]].add_piece(*selected[2:])
        # pop the selected wall
        init_walls.pop(selected[1])
        return True
    return False
    
# def reduce_last(walls: List[CustomWall]):
#     if len(walls[-1].pieces) > 1:
#         return False
#     artpiece = walls.pop().pieces.pop()
#     for wall in walls:
#         available = wall.get_available(artpiece[3], artpiece[4])
#         if available:
#             # sort by largest rectangle
#             largest = []
#             for a in available:
#                 wall.add_piece(artpiece[0], *a, artpiece[3], artpiece[4])
#                 largest.append((largest_rectangle(wall), a))
#                 wall.pieces.pop()
#             largest.sort(key=lambda x: x[0], reverse=True)
#             wall.add_piece(artpiece[0], *available[0], artpiece[3], artpiece[4])
#             return True
#     return False


def select(elements):
    return elements[0]


def largest_rectangle(wall: CustomWall):
    """
    Largest empty rectangle
    """
    grid = [[0] * wall.width() for _ in range(wall.height())]
    for p in wall.pieces:
        for i in range(p['x'], p['x'] + p['width']):
            for j in range(p['y'], p['y'] + p['height']):
                grid[j][i] = 1

    max_area = 0
    hist = [0] * wall.width()

    for i in range(wall.height()):
        for j in range(wall.width()):
            if grid[i][j] == 0:
                hist[j] += 1
            else:
                hist[j] = 0

        stack = []
        for j in range(wall.width()):
            while stack and hist[j] < hist[stack[-1]]:
                h = hist[stack.pop()]
                w = j if not stack else j - stack[-1] - 1
                max_area = max(max_area, h * w)
            stack.append(j)

        while stack:
            h = hist[stack.pop()]
            w = wall.width() if not stack else wall.width() - stack[-1] - 1
            max_area = max(max_area, h * w)

    return max_area

