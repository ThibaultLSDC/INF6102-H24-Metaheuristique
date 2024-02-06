import matplotlib.pyplot as plt
import random
import numpy as np
import os 
from typing import List, Tuple

make_universal = lambda x: os.sep.join(x.split('/'))

class ArtPiece:
    def __init__(self,idx,w,h):
        self.__idx = idx
        self._w = w 
        self._h = h 
    
    def width(self):
        return self._w 
    
    def height(self):
        return self._h

    def _hash_(self):
        return self.__idx

    def _repr_(self):
        return 'ArtPiece'+str([b+':'+str(a) for a,b in zip(self.to_tuple(),['idx','w','h'])])
    
    def to_tuple(self):
        return (self.__idx,self._w,self._h)
    
    def get_idx(self):
        return self.__idx
    

class Wall:
    def __init__(self,w,h):
        self._w = w 
        self._h = h
        self._artpieces = {}

    def width(self):
        return self._w 
    
    def height(self):
        return self._h 
    
    def gen_for_solution(self,id:int) -> List[Tuple[int]]:
        """
            Generates a sequence of tuples (<artpiece_id>,id,<x_pos>,<y_pos>)
            where id is the id you wish to give to this wall
        """
        return [(k,id,*pos) for k,pos in self._artpieces.items()]
        

class Solution:
    def __init__(self,items: List[Tuple[int,int,int,int]]):
        self.items = items 
        self.items_by_wall = {}  # Dictionary to group items by wall
        for item in items:
            wall_id = item[1]
            if wall_id not in self.items_by_wall:
                self.items_by_wall[wall_id] = []
            self.items_by_wall[wall_id].append(item)
        self.nwalls = len(self.items_by_wall)


class Instance:
    def __init__(self,type: str,number:int=1):
        self.filepath = make_universal("./instances/{:}/in_{:02d}.txt".format(type,number))
        with open(self.filepath) as f:
            lines = list([[int(x.strip()) for x in x.split('	')] for x in f.readlines()])
            self.wall = Wall(*lines[0])
            self.n = lines[1][0]
            self.artpieces = [ArtPiece(*x) for x in lines[2:]]
            self.artpieces_dict = {x[0]:ArtPiece(*x) for x in lines[2:]}

    def is_valid_solution(self,sol:Solution) -> bool:
        """ 
            Returns True when the solution is valid
        """
        return self.is_solution_inside_wall(sol) and self.is_solution_non_overlapping(sol)
    
    def is_solution_non_overlapping(self,sol:Solution) -> bool:
        """
            Returns True when the solution has no overlapping pieces
        """
        return len(self.overlapping_pieces(sol))==0


    def overlapping_pieces(self,sol:Solution)->List[Tuple[int,int,int,int]]:
        """
            Returns all the overlapping pairs or artpieces
        """
        items_by_wall = sol.items_by_wall
        pairs = []
        for wall_id, wall_items in items_by_wall.items():
            # O((k^2)/2) with k the number of pieces on the wall
            sorted_items = sorted(wall_items, key=lambda x: (x[-1], x[-2]))  # Sort items by y-coordinate and then by x-coordinate
            for i, it1 in enumerate(sorted_items):
                item1 = {'x':it1[-2],'y':it1[-1],'width':self.artpieces_dict[it1[0]].width(),'height':self.artpieces_dict[it1[0]].height(),'id':it1[0]}

                for j, it2 in enumerate(sorted_items[i + 1:]):
                    item2 = {'x':it2[-2],'y':it2[-1],'width':self.artpieces_dict[it2[0]].width(),'height':self.artpieces_dict[it2[0]].height(),'id':it2[0]}
                    
                    # Check for overlap between item1 and item2
                    if (item1['x'] < item2['x'] + item2['width'] and
                            item1['x'] + item1['width'] > item2['x'] and
                            item1['y'] < item2['y'] + item2['height'] and
                            item1['y'] + item1['height'] > item2['y']):
                        pairs.append((item1['id'],item2['id']))
        return pairs
    

    def is_solution_inside_wall(self,sol:Solution)->bool:
        """
            Returns True when the solution has no pieces that are outside the walls
        """
        return len(self.pieces_outside_wall(sol))==0
    

    def pieces_outside_wall(self,sol:Solution)->List[Tuple[int,int,int,int]]:
        """
            Returns all the pieces that are not fully on the solution walls
        """
        p=[]
        items_by_wall = sol.items_by_wall
        for wall_id, wall_items in items_by_wall.items():
            for i, it1 in enumerate(wall_items):
                item1 = {'x':it1[-2],'y':it1[-1],'width':self.artpieces_dict[it1[0]].width(),'height':self.artpieces_dict[it1[0]].height()}
                if item1['x']+item1['width']>self.wall.width() or item1['y']+item1['height']>self.wall.height():
                    p.append(it1[0])
        return p


        
    def visualize_solution(self, sol:Solution, max_subplots_per_row = 5, visualisation_file='visualization.png', show=True):
        """ 
            Shows a solution
        """
        items = sol.items
        wall_id = list(sorted(list(set([x[1] for x in items]))))
        num_walls = len(wall_id)
        num_rows = (num_walls + max_subplots_per_row - 1) // max_subplots_per_row
        fig, axs = plt.subplots(num_rows, min(max_subplots_per_row, num_walls), figsize=(5*min(max_subplots_per_row, num_walls), 5*num_rows), sharex=True, sharey=True)
        axs=axs if num_rows>1 else [axs]

        for i, ax_row in enumerate(axs):
            ax_row=ax_row if num_walls>1 else [ax_row]
            for j, ax in enumerate(ax_row):
                wall_index = i * max_subplots_per_row + j
                if wall_index < num_walls:
                    wall_dim = self.wall.width(), self.wall.height()

                    # Plot wall
                    ax.add_patch(plt.Rectangle((0, 0), wall_dim[0], wall_dim[1], fill=None, edgecolor='blue'))

                    # Plot items in the current wall
                    items_in_wall = [item for item in items if item[1] == wall_id[wall_index]]
                    for item in items_in_wall:
                        id,id_wall,x,y = item
                        color = (random.random(), random.random(), random.random())
                        ax.add_patch(plt.Rectangle((x,y), self.artpieces_dict[id].width(), self.artpieces_dict[id].height(), fill=True, edgecolor='black', facecolor=color, alpha=.8,label=f'Cadre #{id}'))

                    ax.set_xlim(0, wall_dim[0])
                    ax.set_ylim(0, wall_dim[1])
                    ax.set_xticks(range(0, wall_dim[0]+1) if wall_dim[0]<100 else np.arange(0, wall_dim[0]+1,50))
                    ax.set_xticklabels(ax.get_xticks(),rotation=90)
                    ax.set_yticks(range(0, wall_dim[1]+1) if wall_dim[0]<100 else np.arange(0, wall_dim[1]+1,50))
                    ax.set_aspect('equal', adjustable='box')
                    ax.set_title(f'Mur #{wall_id[wall_index]}')
                    ax.legend()
                    ax.grid()
                    
        # Hide the last subplot if it's not needed
        if num_walls>1 and num_rows>1 and num_walls % max_subplots_per_row != 0:
            for i in range(num_walls % max_subplots_per_row, max_subplots_per_row):
                fig.delaxes(axs[-1][i])

        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.suptitle(f'Visualisation de la solution {self.filepath}')
        plt.tight_layout()
        plt.savefig(visualisation_file)
        if show:
            plt.show()
    
    def save_solution(self, solution: Solution, filedir: str) -> None:
        """
            Saves the solution
        """
        filename = make_universal(filedir+f"/{'/'.join(self.filepath.split(os.sep)[-2:])}")
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename,'w+') as f:
            for x in solution.items:
                f.write(f'{" ".join([str(w) for w in x])}\n')
