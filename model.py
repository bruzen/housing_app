import random
from mesa import Agent, Model
from mesa.space import SingleGrid
from mesa.time import SimultaneousActivation


class Cell(Agent):
    """
    Represents a single cell in the grid.
    """

    ALIVE = 1
    DEAD = 0

    def __init__(self, pos, model):
        super().__init__(pos, model)
        self.state = self.DEAD

    def step(self):
        """
        Compute the next state of the cell.
        """
        # Count the number of alive neighbors
        alive_neighbors = sum(
            1 for neighbor in self.model.grid.get_neighbors(self.pos, moore=True) if neighbor.state == self.ALIVE
        )

        # Apply the rules of Conway's Game of Life
        if self.state == self.ALIVE:
            if alive_neighbors < 2 or alive_neighbors > 3:
                self.state = self.DEAD
        else:
            if alive_neighbors == 3:
                self.state = self.ALIVE


class ConwaysGameOfLife(Model):
    """
    Represents the 2-dimensional array of cells in Conway's Game of Life.
    """

    def __init__(self, width=50, height=50):
        """
        Create a new playing area of (width, height) cells.
        """

        # Set up the grid and schedule
        self.grid = SingleGrid(width, height, torus=True)
        self.schedule = SimultaneousActivation(self)

        # Place a cell at each location, with some initialized to ALIVE and some to DEAD
        for x in range(width):
            for y in range(height):
                cell = Cell((x, y), self)
                if random.random() < 0.1:
                    cell.state = cell.ALIVE
                self.grid.place_agent(cell, (x, y))
                self.schedule.add(cell)

        self.running = True

    def step(self):
        """
        Have the scheduler advance each cell by one step.
        """
        self.schedule.step()
