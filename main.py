# Byan Alkas & Amal Mohamed


# Annotate the variable, argument, and return value types
from typing import Tuple, Iterable, Any
import copy
import numpy as np

# 1 for #, 0 for empty spaces
windy_maze = [  [0, 0, 1, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1],
                [1, 1, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 1],
                [0, 1, 1, 1, 0, 0, 0]  ]

windy_maze = tuple(map(tuple, windy_maze))

# Moving probablity, straight, drift to left, drift to right with .75, .15, .10 respectively. 
moving_probabilties = [0.10, 0.75, 0.15]

# Nested classes
class HMM:
    class Robot:
        
        SENSING = 0
        MOVING = 1

    # Constructor, return None. 
    def __init__(self, maze: Iterable[Iterable[int]], moving_probabilties: Iterable[float]) -> None:        
        self.maze, self.prediction_move = self.__init_game(maze)
        self.moving_probabilties = moving_probabilties

    def __init_game(self, maze):
        game = np.array(maze)
        game = np.ndarray = game
        padding = ((1, 1), (1, 1))
        constant_values = ((1, 1), (1, 1))
        game = np.pad(game, pad_width = padding, mode = 'constant', constant_values = constant_values)
        game = np.transpose(game) 
        # initial precedence for open cells
        probability = 1 / np.sum(game == 0) 
        prediction_move = np.where(game == 0, probability, float('-inf')).astype(float)

        return game, prediction_move
    
    def __cell_neighbors(self, cell: Tuple[int, int], matrix: Iterable[Iterable[Any]]) -> Iterable[int]:

        # Return the values of a specific cell's from the specified matrix.
        x, y = tuple(cell) 
        
        # indicates the direction (West, North, East, South) respectively.
        neighbor_offsets = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        neighbor_cells = [(x + dx, y + dy) for dx, dy in neighbor_offsets]

        return [matrix[n[0]][n[1]] for n in neighbor_cells]
    
    def __sensor_probability(self, cell: Tuple[int, int], sensing: Iterable[int]) -> float: 
        # check this *** - Calculate filtering. 
        cell_neighbors = self.__cell_neighbors(cell, self.maze)
        n_val = cell_neighbors

        probabilities = { 
            # 90% when robot successfully recognizes the obstacle.
            (1, 1): 0.90,
            # 5% when robot misidentifies an open obstacle.
            (1, 0): 0.05,
            # 10% when robot fails to detect an obstacle.
            (0, 1): 0.10,
            # 95% when robot successfully finds an open obstacle.
            (0, 0): 0.95
        }

        prob = 1

        for neighbor, sensor in zip(cell_neighbors, sensing):
            key = (sensor, neighbor)
            prob *= probabilities[key]
        
        return prob
    
    def __prediction_matrix(self, cell: Tuple[int, int], direction: str):
        # Get cell neighbors
        dir = self.__cell_neighbors(cell, self.maze)
        west, north, east, south = dir

        # Initialize transition probabilities(west, north, east, south, self).
        west_dir, north_dir, east_dir, south_dir, self_dir = 0.0, 0.0, 0.0, 0.0, 0.0
        moving_probabilties = self.moving_probabilties
        left, target, right = moving_probabilties

        # Moving north
        if direction == 'N':

            # if there's an obstacle to the west ..
            if west:
                self_dir += left
            else:
                west_dir += left

            # if there's an obstacle to the east ..
            if east:
                self_dir += right
            else:
                east_dir += right

            # if there's an obstacle to the north ..
            if north:
                self_dir += target

            # if there's no open obstacle to the south ..
            if not south:
                south_dir += target

        # Moving west
        elif direction == 'W':

            ## if there's an obstacle to the north ..
            if north:
                self_dir += left
            else:
                north_dir += left

            # if there's an obstacle to the south ..
            if south:
                self_dir += right
            else:
                south_dir += right

            # if there's an obstacle to the west ..
            if west:
                self_dir += target

            # if there's no open obstacle to the east ..
            if not east:
                east_dir += target

        return [self_dir, west_dir, north_dir, east_dir, south_dir]
    
    def __open_space(self):
        for x in range(self.maze.shape[0]):
            for y in range(self.maze.shape[1]):
                maze_cell = self.maze[x, y]
                prediction_cell = self.prediction_move[x, y] 

                if maze_cell == 1:
                    continue           
                yield x, y, prediction_cell

    def sensing_mov(self, sensing: Iterable[int]) -> None:

        # P(Zt|St) = P(ZW,t|St) P(ZN,t|St) P(ZE,t|St) P(ZS,t|St)
        mapper = (lambda cell, sensing = sensing: 
          self.__sensor_probability((cell[0], cell[1]), sensing) * cell[2])
        
        # add up all the conditional probabilities ..
        open_space = self.__open_space()
        total = 0
        for cell in open_space:
            value = mapper(cell)
            total += value
        sum_up = np.array([total]).astype(float)

        # get the answer (cells multiplication / summation)
        open_space = self.__open_space()
        for cell in open_space:
            x, y, past_val = cell
            results = mapper(cell) / sum_up
            self.prediction_move[x, y] = results

    def moving_processes(self, direction: str) -> None:
        
        # Make a copy of the prediction matrix for any changes during runtime.
        prediction_move = copy.deepcopy(self.prediction_move)

        for x, y, past_val in self.__open_space():
            transition_matrix = self.__prediction_matrix((x, y), direction)

            # Create a list with the current value of past_val
            past_val_list = [past_val]

            # Append the values of neighboring cells to past_val_list
            cell_neighbors = self.__cell_neighbors((x, y), prediction_move)
            for neighbor in cell_neighbors:
                past_val_list.append(neighbor)

            # Remove infinite to avoid overflow and runtime problems
            past_val_array = np.array(past_val_list)
            past_val_array[past_val_array == float('-inf')] = 0

            # Compute the updated probability for the current cell
            p = np.dot(transition_matrix, past_val_array).sum()
            self.prediction_move[(x, y)] = p


    def simulate(self, events: Iterable[Tuple[Robot, Any]]) -> None:
        print(' ')
        print("Initial Location Probabilities")
        print(' ')
        self.show_res()

        # Create a separator string of 75 dashes
        separator = '-' * 75

        # Print the separator and the start message
        print(separator)
        

        # Loop through each event and process it accordingly
        for robot, input in events:
            if robot == self.Robot.SENSING:
                self.sensing_mov(input)
            elif robot == self.Robot.MOVING:
                self.moving_processes(input)
                
            # Print the action and input values, where Sensing = 0 and Moving = 1
            print(f"{robot}: {input}: \n")
            
            # Display the prediction and print the separator
            self.show_res()
            print(separator)


    def show_res(self):

        # Loop over the transposed 'prediction_move' array, ignoring the beginning and end rows.
        for i in range(1, self.prediction_move.shape[1] - 1):
            row = self.prediction_move[:, i]
            
            # Format the row as a list of strings with two decimal places, or as '####' if the value is <= 0
            formatted_row = ['####' if value <= 0 else f"{value * 100:.2f}" for value in row[1:-1]]

            print('\t'.join(formatted_row))

        print()

sequence_actions = [ 
    (HMM.Robot.SENSING,[0, 0, 0, 0]), 
    (HMM.Robot.MOVING,'N'), 
    (HMM.Robot.SENSING, [0, 0, 1, 0]), 
    (HMM.Robot.MOVING, 'N'), 
    (HMM.Robot.SENSING, [0, 1, 1, 0]), 
    (HMM.Robot.MOVING, 'W'), 
    (HMM.Robot.SENSING, [0, 1, 0, 0]), 
    (HMM.Robot.MOVING, 'N'), 
    (HMM.Robot.SENSING, [0, 0, 0, 0])  
]

def main():
    hmm = HMM(maze = windy_maze, moving_probabilties = moving_probabilties)
    hmm.simulate(sequence_actions)

if __name__ == "__main__":
    main()
