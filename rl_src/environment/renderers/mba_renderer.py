import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np

from vec_storm.storm_vec_env import StormVecEnv

class MBARenderer:

    def __init__(self, simulator : StormVecEnv = None):
        self.simulator = simulator
        self.initialize_grid_matrix()

    def initialize_grid_matrix(self):
        # Implements initialization of the grid matrix for the environment
        self.grid_size = (5, 4) # (x_max, y_max)
        self.walls = {(1, 0), (1, 1), (1, 2), (3, 0), (3, 1), (3, 2)}  # Pevné překážky
        self.goal = (2, 0)  # Cílová pozice
        # self.bad = (0, 0)

    def get_trajectory_data(self):
        state_labels = self.simulator.get_state_labels()
        state_valuations = self.simulator.get_state_values().tolist()
        current_batch_states = self.simulator.simulator_states.vertices.tolist()
        # get x and y coordinates from the first state given state_labels
        x_dim = state_labels.index("x")
        y_dim = state_labels.index("y")
        state = current_batch_states[0]
        state_values = state_valuations[state]
        x = state_values[x_dim]
        y = state_values[y_dim]
        transition = (x, y)
        return transition

    def plot_slideshow_grid(self, trajectory):
        # Zde budou souřadnice trajektorie (doplníš je)
        # trajectory = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 2), (2, 1), (2, 0)]  # Ukázková trajektorie
        # Funkce pro vykreslení jednoho framu animace
        def draw_frame(i):
            ax.clear()
            ax.set_xlim(-0.5, self.grid_size[0] - 0.5)
            ax.set_ylim(-0.5, self.grid_size[1] - 0.5)
            ax.set_xticks(np.arange(-0.5, self.grid_size[0], 1))
            ax.set_yticks(np.arange(-0.5, self.grid_size[1], 1))
            ax.grid(True)
            ax.invert_yaxis()

            # Vybarvení celého prostředí na bílo
            for x in range(self.grid_size[0]):
                for y in range(self.grid_size[1]):
                    ax.add_patch(patches.Rectangle((x - 0.5, y - 0.5), 1, 1, color="white", edgecolor="black"))

            # Vykreslení překážek (černé)
            for (x, y) in self.walls:
                ax.add_patch(patches.Rectangle((x - 0.5, y - 0.5), 1, 1, color="black"))

            # Cíl (zelená)
            ax.add_patch(patches.Rectangle((self.goal[0] - 0.5, self.goal[1] - 0.5), 1, 1, color="green", alpha=0.5, label="Goal"))

            # Nebezpečná oblast (červená)
            # ax.add_patch(patches.Rectangle((self.bad[0] - 0.5, self.bad[1] - 0.5), 1, 1, color="red", alpha=0.5, label="Bad"))

            # Historie pohybu agenta (šedá stopa)
            for j in range(i):
                ax.add_patch(patches.Circle((trajectory[j][0], trajectory[j][1]), 0.2, color="gray", alpha=0.5))

            # Aktuální pozice agenta (modrá)
            ax.add_patch(patches.Circle((trajectory[i][0], trajectory[i][1]), 0.3, color="blue", label="Agent"))

            ax.legend()
            plt.gca().invert_yaxis()  # Otočení osy Y pro správné zobrazení
        fig, ax = plt.subplots(figsize=(5, 5))
        
        ani = animation.FuncAnimation(fig, draw_frame, frames=len(trajectory), interval=500)  # 500 ms mezi kroky
        
        ani.save("animation.gif", writer="pillow", fps=1)  # Uložení animace do souboru

if __name__ == "__main__":
    print("Testing MBA renderer")
    renderer = MBARenderer()
    renderer.plot_slideshow_grid(trajectory = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 2), (2, 1), (2, 0)])
