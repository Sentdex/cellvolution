import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import time



SAVE_FRAMES = False # Will save frames at the expense of considerable simulation speed.

class Cell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Cell, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.energy = 100
    
    def forward(self, x):
        return self.network(x)

class Simulation:
    def __init__(self, width, height, n_races, initial_density=0.3):
        self.width = width
        self.height = height
        self.n_races = n_races
        self.grid = np.full((height, width), -1, dtype=int)  # Initialize all cells as empty (-1)
        self.cells = [[None for _ in range(width)] for _ in range(height)]
        self.resources = self.initialize_resources()
        self.step_count = 0
        self.replication_threshold = 150
        self.mutation_rate = 0.1
        self.collision_stats = {i: {'wins': 0, 'losses': 0} for i in range(n_races)}
        self.collision_stats[-1] = {'wins': 0, 'losses': 0}  # Add stats for empty/dead cells
        self.all_cells_dead = False

        # Initialize cells based on initial_density
        total_cells = int(width * height * initial_density)
        cells_per_race = total_cells // n_races
        remaining_cells = total_cells % n_races

        for race in range(n_races):
            for _ in range(cells_per_race + (1 if race < remaining_cells else 0)):
                while True:
                    x, y = random.randint(0, width-1), random.randint(0, height-1)
                    if self.grid[y][x] == -1:  # If the spot is empty
                        self.grid[y][x] = race
                        self.cells[y][x] = Cell(75, 30, 5)
                        break

        initial_counts = [np.sum(self.grid == i) for i in range(self.n_races)]
        print("Initial cell counts for each race:", initial_counts)

    
    
    def initialize_resources(self):
        resources = np.zeros((self.height, self.width))
        num_patches = (self.height * self.width) // 100  # Create resource patches in about 1% of the grid
        for _ in range(num_patches):
            x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            resources[y, x] = random.randint(50, 100)  # Resource-rich patches
        return resources

    def step(self):
        new_grid = self.grid.copy()
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == -1:  # Skip dead cells
                    continue
                
                cell = self.cells[y][x]
                if cell is None or cell.energy <= 0:
                    new_grid[y, x] = -1  # Dead cell
                    self.cells[y][x] = None
                    continue
                
                neighborhood = self.get_neighborhood(x, y)
                action_probs = torch.softmax(cell(torch.FloatTensor(neighborhood).unsqueeze(0)).squeeze(0), dim=0)
                action = torch.multinomial(action_probs, 1).item()
                
                cell.energy -= 1  # Energy cost for existing
                
                new_x, new_y = x, y
                if action == 0:  # Move up
                    new_y = (y - 1) % self.height
                elif action == 1:  # Move right
                    new_x = (x + 1) % self.width
                elif action == 2:  # Move down
                    new_y = (y + 1) % self.height
                elif action == 3:  # Move left
                    new_x = (x - 1) % self.width
                elif action == 4:  # Stay and consume
                    consumed = min(self.resources[y, x], 10)
                    cell.energy += consumed
                    self.resources[y, x] -= consumed
                
                if cell.energy >= self.replication_threshold:
                    self.replicate(cell, new_grid, x, y)
                
                if new_grid[new_y, new_x] == -1:  # Empty space
                    new_grid[new_y, new_x] = self.grid[y, x]
                    self.cells[new_y][new_x] = cell
                    new_grid[y, x] = -1
                    self.cells[y][x] = None
                    self.collision_stats[self.grid[y, x]]['wins'] += 1
                    self.collision_stats[-1]['losses'] += 1
                elif new_grid[new_y, new_x] != self.grid[y, x]:  # Different race collision
                    other_cell = self.cells[new_y][new_x]
                    if other_cell is None or cell.energy > other_cell.energy:
                        # Current cell wins
                        cell.energy -= other_cell.energy if other_cell else 0
                        new_grid[new_y, new_x] = self.grid[y, x]
                        self.cells[new_y][new_x] = cell
                        new_grid[y, x] = -1
                        self.cells[y][x] = None
                        self.collision_stats[self.grid[y, x]]['wins'] += 1
                        self.collision_stats[new_grid[new_y, new_x]]['losses'] += 1
                    else:
                        other_cell.energy -= cell.energy
                        new_grid[y, x] = -1
                        self.cells[y][x] = None
                        self.collision_stats[self.grid[y, x]]['losses'] += 1   
                        self.collision_stats[new_grid[new_y, new_x]]['wins'] += 1 
        self.grid = new_grid
        
        # Slow resource regeneration
        self.resources += np.random.random(self.resources.shape) < 0.05  # 0.1% chance of adding 1 resource
        
        self.step_count += 1       
        if self.step_count % 100 == 0:
            print(f"Step {self.step_count}: Active cells: {np.sum(self.grid != -1)}")
            print("Collision stats:", self.collision_stats)

    def get_neighborhood(self, x, y):
        neighborhood = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                nx, ny = (x + i) % self.width, (y + j) % self.height
                cell_type = self.grid[ny, nx]
                cell_energy = self.cells[ny][nx].energy if self.cells[ny][nx] is not None else 0
                resource = self.resources[ny, nx]
                neighborhood.extend([cell_type, cell_energy, resource])
        return neighborhood

    def replicate(self, parent_cell, new_grid, x, y):
        neighbors = [(x, (y-1) % self.height), ((x+1) % self.width, y),
                     (x, (y+1) % self.height), ((x-1) % self.width, y)]
        empty_neighbors = [pos for pos in neighbors if new_grid[pos[1]][pos[0]] == -1]
        
        if empty_neighbors:
            new_x, new_y = random.choice(empty_neighbors)
            
            new_cell = Cell(75, 30, 5)  # Match the new input size
            new_cell.load_state_dict(parent_cell.state_dict())
            
            if random.random() < self.mutation_rate:
                with torch.no_grad():
                    for param in new_cell.parameters():
                        param.add_(torch.randn(param.size()) * 0.1)
            
            new_cell.energy = parent_cell.energy // 2
            parent_cell.energy = parent_cell.energy // 2
            
            new_grid[new_y][new_x] = new_grid[y][x]  # Same race as parent
            self.cells[new_y][new_x] = new_cell

    def evolve(self):
        performances = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] != -1 and self.cells[y][x] is not None:
                    performances[y, x] = self.cells[y][x].energy
        
        worst_indices = np.argsort(performances.flatten())[:100]
        best_indices = np.argsort(performances.flatten())[-100:]
        
        for worst, best in zip(worst_indices, best_indices):
            wy, wx = np.unravel_index(worst, performances.shape)
            by, bx = np.unravel_index(best, performances.shape)
            
            if self.grid[wy][wx] != -1 and self.grid[by][bx] != -1 and self.cells[wy][wx] is not None and self.cells[by][bx] is not None:
                self.cells[wy][wx].load_state_dict(self.cells[by][bx].state_dict())
                with torch.no_grad():
                    for param in self.cells[wy][wx].parameters():
                        param.add_(torch.randn(param.size()) * 0.1)
                self.cells[wy][wx].energy = 100  # Reset energy for new cell
        
        print(f"Evolution performed at step {self.step_count}")


class Visualization:
    def __init__(self, simulation):
        self.simulation = simulation
        plt.ion()  # Turn on interactive mode
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(14, 14))  # Square figure
        
        # Modify the gridspec to allocate 80% height to the top graph
        self.gs = self.fig.add_gridspec(5, 1)  # 5 rows, 1 column
        self.ax1 = self.fig.add_subplot(self.gs[:4, 0])  # Top 4 rows (80%)
        self.ax2 = self.fig.add_subplot(self.gs[4, 0])   # Bottom row (20%)
        
        self.population_history = [[] for _ in range(simulation.n_races)]
        self.steps = []
        
        if SAVE_FRAMES:
            # Create directory for saving frames
            self.sim_dir = os.path.join("simulations", str(int(time.time())))
            os.makedirs(self.sim_dir, exist_ok=True)

        self.frame_count = 0
        
        self.setup_plot()

    def setup_plot(self):
        self.ax1.set_title("Cells and Resources")
        self.ax2.set_title("Population by Race Over Time")

        # Define less bright colors for races
        self.race_colors = ['#8B0000', '#006400', '#00008B', '#8B8B00', '#8B008B']  # Dark Red, Dark Green, Dark Blue, Dark Yellow, Dark Magenta
        if self.simulation.n_races > len(self.race_colors):
            self.race_colors.extend(['#' + ''.join([random.choice('456789ABCD') for _ in range(6)]) 
                                     for _ in range(self.simulation.n_races - len(self.race_colors))])

        # Create custom colormap for races
        race_cmap = ListedColormap(['#1A1A1A'] + self.race_colors)  # Dark gray for empty/dead cells

        # Create custom colormap for resources (black to white)
        resource_cmap = LinearSegmentedColormap.from_list("resource_cmap", ['#000000', '#FFFFFF'])

        # Create the main plot
        self.im1 = self.ax1.imshow(self.simulation.grid, cmap=race_cmap, vmin=-1, vmax=self.simulation.n_races-1, interpolation='nearest')
        
        # Create a semi-transparent overlay for resources
        self.resource_overlay = self.ax1.imshow(self.simulation.resources, cmap=resource_cmap, alpha=0.5, vmin=0, vmax=100, interpolation='nearest')

        # Setup line graph for population over time
        self.lines = [self.ax2.plot([], [], color=color, label=f'Race {i}')[0] 
                      for i, color in enumerate(self.race_colors)]
        self.ax2.set_xlabel('Steps')
        self.ax2.set_ylabel('Population')
        self.legend = self.ax2.legend(loc='upper left', fontsize='small')

        # Set background color for both subplots
        self.ax1.set_facecolor('#1A1A1A')
        self.ax2.set_facecolor('#1A1A1A')

        # Remove x-axis labels from the top plot to save space
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])

        self.fig.tight_layout()

    def update(self):
        self.im1.set_data(self.simulation.grid)
        self.resource_overlay.set_data(self.simulation.resources)

        # Update population line graph
        populations = [np.sum(self.simulation.grid == i) for i in range(self.simulation.n_races)]
        self.steps.append(self.simulation.step_count)
        for i, pop in enumerate(populations):
            self.population_history[i].append(pop)
        
        for line, pop_history, pop in zip(self.lines, self.population_history, populations):
            line.set_data(self.steps, pop_history)
            line.set_label(f'Race {line.get_label().split()[1]} ({pop} alive)')
        
        self.ax2.relim()
        self.ax2.autoscale_view()

        # Update the legend with current population counts
        self.legend.remove()
        self.legend = self.ax2.legend(loc='upper left')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()  # Refresh the plot

        if SAVE_FRAMES:
            self.save_frame()

    def save_frame(self):
        self.frame_count += 1
        filename = os.path.join(self.sim_dir, f"{self.frame_count:06d}.png")
        self.fig.savefig(filename)

    def run(self):
        while True:
            try:
                self.simulation.step()
                if random.random() < 0.3:  # Evolve occasionally
                    self.simulation.evolve()
                self.update()
                
                # Check if all cells are dead
                if np.all(self.simulation.grid == -1):
                    print("All cells have died. Simulation ending.")
                    break
                
            except KeyboardInterrupt:
                print("Simulation stopped by user.")
                break
        
        # Keep the plot window open after simulation ends
        plt.ioff()
        plt.show()


#sim = Simulation(800, 600, 10, initial_density=0.05)  # for the very patient among us.
sim = Simulation(40, 30, 5, initial_density=0.1)  # for the rest of you
vis = Visualization(sim)
vis.run()
