import matplotlib.pyplot as plt


class Plotter:
    fac = 1 / 255
    exp = 1.15
    colors = {
        "fig": ((34 * fac) ** exp, (34 * fac) ** exp, (35 * fac) ** exp),
        "plot": ((30 * fac) ** exp,) * 3,
        "legend": ((37 * fac) ** exp, (37 * fac) ** exp, (38 * fac) ** exp, 0.5),
        "text": ((212 * fac) ** exp,) * 3,
        "text_dark": ((133 * fac) ** exp,) * 3,
        "border": ((65 * fac) ** exp,) * 3,
        "red": ((206 * fac) ** exp, (145 * fac) ** exp, (120 * fac) ** exp),
        "red_transp": ((206 * fac) ** exp, (145 * fac) ** exp, (120 * fac) ** exp, 0.3),
        "green": ((181 * fac) ** exp, (206 * fac) ** exp, (168 * fac) ** exp),
        "green_transp": ((181 * fac) ** exp, (206 * fac) ** exp, (168 * fac) ** exp, 0.3),
        "blue": ((156 * fac) ** exp, (220 * fac) ** exp, (254 * fac) ** exp),
        "blue_transp": ((156 * fac) ** exp, (220 * fac) ** exp, (254 * fac) ** exp, 0.3),
    }

    def __init__(self, title, data, line_width=1.5):
        plt.ion()  # Enable interactive mode

        # Perfect window size for Retina Macbook: 2880x1536 (1440x768)
        self.figure = plt.figure(figsize=(15, 8), dpi=96)
        self.figure.patch.set_facecolor(self.colors["fig"])
        self.figure.suptitle(title, color=self.colors["text_dark"])

        # Initialize empty plots
        self.plots = {}
        for plot_name in data:

            plot_data = data[plot_name]

            plot = self.figure.add_subplot(plot_data["placement"])

            graphs = []
            for graph in plot_data["graphs"]:
                graphs.append(
                    plt.plot(
                        [],
                        [],
                        linewidth=line_width,
                        color=self.colors[graph["color"]] if "color" in graph else None,
                        label=graph["label"] if "label" in graph else None,
                    )[0]
                )

            if "ylabel" in plot_data:
                plt.ylabel(plot_data["ylabel"], color=self.colors["text_dark"])
            if "xlabel" in plot_data:
                plt.xlabel(plot_data["xlabel"], color=self.colors["text_dark"])

            if "legend" in plot_data and plot_data["legend"]:
                legend = plt.legend()
                legend_frame = legend.get_frame()
                legend_frame.set_facecolor(self.colors["legend"])
                legend_frame.set_edgecolor(self.colors["border"])
                for text in legend.get_texts():
                    text.set_color(self.colors["text"])

            plt.grid(color=self.colors["border"])

            plot.set_facecolor(self.colors["plot"])
            plot.tick_params(color=self.colors["border"], labelcolor=self.colors["text"])
            for spine in plot.spines.values():
                spine.set_edgecolor(self.colors["border"])

            self.plots[plot_name] = {"plot": plot, "graphs": graphs}

        self.figure.align_labels()

    def update_data(self, data):

        for plot_name in data:
            if plot_name not in self.plots:
                continue

            plot_data = data[plot_name]
            plot_graphs = self.plots[plot_name]["graphs"]

            for i, graph_data in enumerate(plot_data[:-1]):
                graph = plot_graphs[i]
                graph.set_data(plot_data[-1], graph_data)

            plot = self.plots[plot_name]["plot"]
            plot.relim()  # Recalculate limits
            plot.autoscale()  # Autoscale
            plot.set_xlim(left=0)

        plt.pause(1e-6)

    def save_image(self, file_name):
        plt.ioff()
        plt.savefig(file_name, facecolor=self.figure.get_facecolor(), edgecolor="none")
