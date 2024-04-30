from pandas import DataFrame
from matplotlib.pyplot import figure, rcParams, savefig, title, xlabel, xlim, xticks, xscale, ylabel, ylim, yticks, \
    yscale
from pandas import read_csv
from pathlib import Path
from seaborn import color_palette, lineplot, set_palette, set_theme
from shutil import rmtree
from warnings import filterwarnings

from goffls.util.config_parser_util import parse_config_section
from goffls.util.logger_util import load_logger, log_message


class ResultAnalyzer:
    def __init__(self,
                 config_file: Path) -> None:
        # Initialize the attributes.
        self._config_file = config_file
        self._logging_settings = None
        self._input_settings = None
        self._analysis_settings = None
        self._selected_fit_clients_history_settings = None
        self._selected_evaluate_clients_history_settings = None
        self._logger = None
        self._analysis_name = None
        self._analysis_result_folder = None
        self._results_df = None
        self._plotting_df = None
        # Parse the settings.
        self._parse_settings()
        # Set the logger.
        self._set_logger()
        # Filter 'ignore' warnings.
        filterwarnings("ignore")

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    def _parse_settings(self) -> None:
        # Get the necessary attributes.
        config_file = self.get_attribute("_config_file")
        # Parse and set the logging settings.
        logging_section = "Logging Settings"
        logging_settings = parse_config_section(config_file, logging_section)
        self._set_attribute("_logging_settings", logging_settings)
        # Parse and set the input settings.
        input_section = "Input Settings"
        input_settings = parse_config_section(config_file, input_section)
        self._set_attribute("_input_settings", input_settings)
        # Parse and set the analysis settings.
        analysis_section = "Analysis Settings"
        analysis_settings = parse_config_section(config_file, analysis_section)
        self._set_attribute("_analysis_settings", analysis_settings)
        # Parse and set the selected fit clients history settings.
        selected_fit_clients_history_section = "Selected Fit Clients History Settings"
        selected_fit_clients_history_settings = parse_config_section(config_file, selected_fit_clients_history_section)
        self._set_attribute("_selected_fit_clients_history_settings", selected_fit_clients_history_settings)
        # Parse and set the selected evaluate clients history settings.
        selected_evaluate_clients_history_section = "Selected Evaluate Clients History Settings"
        selected_evaluate_clients_history_settings = parse_config_section(config_file,
                                                                          selected_evaluate_clients_history_section)
        self._set_attribute("_selected_evaluate_clients_history_settings", selected_evaluate_clients_history_settings)

    def _set_logger(self) -> None:
        logging_settings = self._logging_settings
        logger_name = type(self).__name__ + "_Logger"
        logger = load_logger(logging_settings, logger_name)
        self._logger = logger

    def _load_theme_and_palette(self) -> None:
        # Get the necessary attributes.
        selected_fit_clients_history_settings = self.get_attribute("_selected_fit_clients_history_settings")
        plotting_settings = selected_fit_clients_history_settings["plotting_settings"]
        theme_style = plotting_settings["theme_style"]
        palette = plotting_settings["palette"]
        # Set aspects of the visual theme for all matplotlib and seaborn plots.
        set_theme(style=theme_style)
        # Set the matplotlib color cycle using a seaborn palette.
        if palette == "color_palette":
            line_colors = plotting_settings["line_colors"]
            n_colors = plotting_settings["n_colors"]
            set_palette(palette=color_palette(line_colors),
                        n_colors=n_colors)

    def _load_figure_settings(self) -> None:
        # Get the necessary attributes.
        selected_fit_clients_history_settings = self.get_attribute("_selected_fit_clients_history_settings")
        plotting_settings = selected_fit_clients_history_settings["plotting_settings"]
        figure_size = plotting_settings["figure_size"]
        figure_dpi = plotting_settings["figure_dpi"]
        figure_face_color = plotting_settings["figure_face_color"]
        figure_edge_color = plotting_settings["figure_edge_color"]
        figure_frame_on = plotting_settings["figure_frame_on"]
        figure_layout = plotting_settings["figure_layout"]
        title_label = plotting_settings["title_label"]
        title_font_size = plotting_settings["title_font_size"]
        legend_font_size = plotting_settings["legend_font_size"]
        x_label = plotting_settings["x_label"]
        x_font_size = plotting_settings["x_font_size"]
        x_ticks = plotting_settings["x_ticks"]
        x_rotation = plotting_settings["x_rotation"]
        x_lim = plotting_settings["x_lim"]
        x_scale = plotting_settings["x_scale"]
        y_label = plotting_settings["y_label"]
        y_font_size = plotting_settings["y_font_size"]
        y_ticks = plotting_settings["y_ticks"]
        y_rotation = plotting_settings["y_rotation"]
        y_lim = plotting_settings["y_lim"]
        y_scale = plotting_settings["y_scale"]
        # Set the figure settings.
        figure(figsize=figure_size,
               dpi=figure_dpi,
               facecolor=figure_face_color,
               edgecolor=figure_edge_color,
               frameon=figure_frame_on,
               layout=figure_layout)
        # Set the title settings.
        title(label=title_label,
              fontsize=title_font_size)
        # Set the legend settings.
        rcParams["legend.fontsize"] = legend_font_size
        # Set the x-axis settings.
        xlabel(xlabel=x_label,
               fontsize=x_font_size)
        xticks(ticks=x_ticks,
               rotation=x_rotation)
        if x_lim is not None:
            xlim(x_lim)
        if x_scale is not None:
            xscale(x_scale)
        # Set the y-axis settings.
        ylabel(ylabel=y_label,
               fontsize=y_font_size)
        yticks(ticks=y_ticks,
               rotation=y_rotation)
        if y_lim is not None:
            ylim(y_lim)
        if y_scale is not None:
            yscale(y_scale)

    def _plot_data(self) -> None:
        # Get the necessary attributes.
        selected_fit_clients_history_settings = self.get_attribute("_selected_fit_clients_history_settings")
        plotting_settings = selected_fit_clients_history_settings["plotting_settings"]
        plotting_df = self.get_attribute("_plotting_df")
        x_data = plotting_settings["x_data"]
        y_data = plotting_settings["y_data"]
        hue = plotting_settings["hue"]
        hue_order = plotting_settings["hue_order"]
        style = plotting_settings["style"]
        dashes = plotting_settings["dashes"]
        markers = plotting_settings["markers"]
        markers_size = plotting_settings["markers_size"]
        alpha = plotting_settings["alpha"]
        size = plotting_settings["size"]
        line_sizes = plotting_settings["line_sizes"]
        # Plot the 'plotting_df' dataframe into the figure.
        ax = lineplot(data=plotting_df,
                      x=x_data,
                      y=y_data,
                      hue=hue,
                      hue_order=hue_order,
                      style=style,
                      dashes=dashes,
                      markers=markers,
                      markersize=markers_size,
                      alpha=alpha,
                      size=size,
                      sizes=line_sizes)
        # Fix the legend handles and labels.
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels)

    def _save_figure(self) -> None:
        # Get the necessary attributes.
        selected_fit_clients_history_settings = self.get_attribute("_selected_fit_clients_history_settings")
        plotting_settings = selected_fit_clients_history_settings["plotting_settings"]
        figure_bbox_inches = plotting_settings["figure_bbox_inches"]
        n_tasks = plotting_settings["n_tasks"]
        metric_name = plotting_settings["metric_name"]
        analysis_name = self.get_attribute("_analysis_name")
        analysis_result_folder = self.get_attribute("_analysis_result_folder")
        logger = self.get_attribute("_logger")
        # Set the figure's output file.
        figure_output_file = Path("fig_{0}_{1}_tasks_{2}.pdf".format(analysis_name, n_tasks, metric_name.lower()))
        figure_output_file = analysis_result_folder.joinpath(figure_output_file)
        # Save the figure.
        savefig(fname=figure_output_file,
                bbox_inches=figure_bbox_inches)
        # Log a 'figure successfully generated' message.
        message = "Figure '{0}' was successfully generated.".format(figure_output_file)
        log_message(logger, message, "INFO")

    def _generate_figures_for_selected_fit_clients_history_file(self) -> None:
        # Get the necessary attributes.
        selected_fit_clients_history_settings = self.get_attribute("_selected_fit_clients_history_settings")
        num_tasks = selected_fit_clients_history_settings["num_tasks"]
        num_available_clients = selected_fit_clients_history_settings["num_available_clients"]
        client_selectors = selected_fit_clients_history_settings["client_selectors"]
        metrics_names = selected_fit_clients_history_settings["metrics_names"]
        time_unit = selected_fit_clients_history_settings["time_unit"]
        plotting_settings = selected_fit_clients_history_settings["plotting_settings"]
        results_df = self.get_attribute("_results_df")
        # Initialize the time unit symbol ('s' stands for seconds).
        time_unit_symbol = "s"
        # Generate a figure for each (num_tasks, metric_name) tuple.
        # The figures contain a data point for each (client_selector, num_available_clients) tuple.
        for n_tasks in num_tasks:
            # Update the plotting settings with the current number of tasks.
            plotting_settings.update({"n_tasks": n_tasks})
            for metric_name in metrics_names:
                # Update the plotting settings with the current metric name.
                plotting_settings.update({"metric_name": metric_name})
                # Initialize the plotting data.
                plotting_data = []
                # Iterate through the list of number of available clients.
                for n_available_clients in num_available_clients:
                    # Iterate through the list of client selectors.
                    for client_selector in client_selectors:
                        # Filter the 'results_df' dataframe considering the current
                        # (client_selector, num_tasks, num_available_clients) tuple.
                        filtered_df = results_df[(results_df["client_selector"] == client_selector) &
                                                 (results_df["num_tasks"] == n_tasks) &
                                                 (results_df["num_available_clients"] == n_available_clients)]
                        # If the current metric name equals to 'selection_duration'...
                        if metric_name == "selection_duration":
                            # Get the number of communication rounds ran by the current client selector.
                            num_comm_rounds = len(list(filtered_df["comm_round"].sort_values().unique()))
                            # Calculate the mean value of its selection duration.
                            selection_duration_mean = filtered_df["selection_duration"].sum() / num_comm_rounds
                            # Convert the mean selection duration value to...
                            if time_unit == "milliseconds":
                                # Milliseconds (1 ms = 1e+3 s).
                                time_unit_symbol = "ms"
                                selection_duration_mean = selection_duration_mean * pow(10, 3)
                            elif time_unit == "microseconds":
                                # Microseconds (1 µs = 1e+6 s).
                                time_unit_symbol = "µs"
                                selection_duration_mean = selection_duration_mean * pow(10, 6)
                            # Append the selection duration mean of the current client selector to the plotting data.
                            plotting_data.append({"client_selector": client_selector,
                                                  "num_available_clients": n_available_clients,
                                                  "selection_duration_mean": selection_duration_mean})
                # If the current metric name equals to 'selection_duration'...
                if metric_name == "selection_duration":
                    # Set the 'x_data' value, if equals to 'Auto'.
                    if plotting_settings["x_data"] == "Auto":
                        x_data = "num_available_clients"
                        plotting_settings["x_data"] = x_data
                    # Set the 'y_data' value, if equals to 'Auto'.
                    if plotting_settings["y_data"] == "Auto":
                        y_data = "selection_duration_mean"
                        plotting_settings["y_data"] = y_data
                    # Set the 'x_label' value, if equals to 'Auto'.
                    if plotting_settings["x_label"] == "Auto":
                        x_label = "Number of available clients"
                        plotting_settings["x_label"] = x_label
                    # Set the 'y_label' value, if equals to 'Auto'.
                    if plotting_settings["y_label"] == "Auto":
                        y_scale = plotting_settings["y_scale"]
                        scale_str = ", log scale" if y_scale == "log" else ""
                        y_label = "{0} ({1}{2})".format(metric_name.replace("_", " ").capitalize(),
                                                        time_unit_symbol,
                                                        scale_str)
                        plotting_settings["y_label"] = y_label
                    # Set the 'hue' value, if equals to 'Auto'.
                    if plotting_settings["hue"] == "Auto":
                        hue = "client_selector"
                        plotting_settings["hue"] = hue
                    # Set the 'hue_order' value, if equals to 'Auto'.
                    if plotting_settings["hue_order"] == "Auto":
                        hue_order = client_selectors
                        plotting_settings["hue_order"] = hue_order
                        # Set the 'style' value, if equals to 'Auto'.
                    if plotting_settings["style"] == "Auto":
                        style = "client_selector"
                        plotting_settings["style"] = style
                        # Set the 'size' value, if equals to 'Auto'.
                    if plotting_settings["size"] == "Auto":
                        size = "client_selector"
                        plotting_settings["size"] = size
                        # Update the plotting settings.
                selected_fit_clients_history_settings["plotting_settings"] = plotting_settings
                self._set_attribute("_selected_fit_clients_history_settings", selected_fit_clients_history_settings)
                # Set the 'plotting_df' dataframe (data that will be plotted into the figure).
                plotting_df = DataFrame(data=plotting_data)
                self._set_attribute("_plotting_df", plotting_df)
                # Load the figure settings for plots.
                self._load_figure_settings()
                # Plot data into the figure.
                self._plot_data()
                # Save the figure.
                self._save_figure()

    def _analyze_selected_fit_clients_history_file(self) -> None:
        # Get the necessary attributes.
        input_settings = self.get_attribute("_input_settings")
        analysis_settings = self.get_attribute("_analysis_settings")
        selected_fit_clients_history_file = input_settings["selected_fit_clients_history_file"]
        analysis_root_folder = Path(analysis_settings["analysis_root_folder"])
        selected_fit_clients_history_settings = self.get_attribute("_selected_fit_clients_history_settings")
        results_df_settings = selected_fit_clients_history_settings["results_df_settings"]
        num_tasks = selected_fit_clients_history_settings["num_tasks"]
        num_available_clients = selected_fit_clients_history_settings["num_available_clients"]
        client_selectors = selected_fit_clients_history_settings["client_selectors"]
        plotting_settings = selected_fit_clients_history_settings["plotting_settings"]
        logger = self.get_attribute("_logger")
        # Log an 'analyzing the results' message.
        message = "Analyzing the results in the '{0}' file...".format(selected_fit_clients_history_file)
        log_message(logger, message, "INFO")
        # Set the analysis name.
        analysis_name = "selected_fit_clients_history"
        self._set_attribute("_analysis_name", analysis_name)
        # Set the analysis result folder.
        analysis_result_folder = analysis_root_folder.joinpath(analysis_name)
        self._set_attribute("_analysis_result_folder", analysis_result_folder)
        # Remove the analysis result folder and its contents (if exists).
        if analysis_result_folder.is_dir():
            rmtree(analysis_result_folder)
        # Create the parents directories of the analysis result folder (if not exist yet).
        analysis_result_folder.parent.mkdir(exist_ok=True, parents=True)
        # Create the analysis result folder.
        analysis_result_folder.mkdir(exist_ok=True, parents=True)
        # Load a dataframe from the results' file.
        results_df = read_csv(filepath_or_buffer=selected_fit_clients_history_file,
                              comment=results_df_settings["comments_prefix"])
        # Order the dataframe by client selector's name in ascending order.
        results_df = results_df.sort_values(by=results_df_settings["sort_by"],
                                            ascending=results_df_settings["sort_ascending_order"])
        # Set the 'results_df' dataframe.
        self._set_attribute("_results_df", results_df)

        # TODO: Fix Parser for dict and tuple values for dictionary keys.
        plotting_settings["figure_size"] = (6, 5)
        plotting_settings["line_colors"] = ["#00FFFF", "#FFA500", "#E0115F", "#0000FF", "#7FFFD4", "#228B22"]
        plotting_settings["line_sizes"] = [2, 2, 2, 2, 2, 2]

        # Set the 'num_tasks' value, if equals to 'Auto'.
        if num_tasks == "Auto":
            num_tasks = list(results_df["num_tasks"].sort_values().unique())
            selected_fit_clients_history_settings["num_tasks"] = num_tasks
        # Set the 'num_available_clients' value, if equals to 'Auto'.
        if num_available_clients == "Auto":
            num_available_clients = list(results_df["num_available_clients"].sort_values().unique())
            selected_fit_clients_history_settings["num_available_clients"] = num_available_clients
        # Set the 'client_selectors' value, if equals to 'Auto'.
        if client_selectors == "Auto":
            client_selectors = list(results_df["client_selector"].sort_values().unique())
            selected_fit_clients_history_settings["client_selectors"] = client_selectors
        # Set the 'n_colors' value, if equals to 'Auto'.
        if plotting_settings["n_colors"] == "Auto":
            plotting_settings["n_colors"] = len(client_selectors)
        # Set the 'x_ticks' value, if equals to 'Auto'.
        if plotting_settings["x_ticks"] == "Auto":
            plotting_settings["x_ticks"] = num_available_clients
        # Set the 'y_ticks' value, if equals to 'Auto'.
        if plotting_settings["y_ticks"] == "Auto":
            plotting_settings["y_ticks"] = [pow(10, 0), pow(10, 2), pow(10, 4), pow(10, 6), pow(10, 8), pow(10, 10)]
        # Set the 'y_lim' value, if equals to 'Auto'.
        if plotting_settings["y_lim"] == "Auto":
            plotting_settings["y_lim"] = min(plotting_settings["y_ticks"]), max(plotting_settings["y_ticks"])
        # Update the plotting settings.
        selected_fit_clients_history_settings["plotting_settings"] = plotting_settings
        self._set_attribute("_selected_fit_clients_history_settings", selected_fit_clients_history_settings)
        # Load the theme and palette for plots.
        self._load_theme_and_palette()
        # Generate the figures.
        self._generate_figures_for_selected_fit_clients_history_file()

    def _generate_figures_for_selected_evaluate_clients_history_file(self) -> None:
        # Get the necessary attributes.
        selected_evaluate_clients_history_settings = self.get_attribute("_selected_evaluate_clients_history_settings")
        num_tasks = selected_evaluate_clients_history_settings["num_tasks"]
        num_available_clients = selected_evaluate_clients_history_settings["num_available_clients"]
        client_selectors = selected_evaluate_clients_history_settings["client_selectors"]
        metrics_names = selected_evaluate_clients_history_settings["metrics_names"]
        time_unit = selected_evaluate_clients_history_settings["time_unit"]
        plotting_settings = selected_evaluate_clients_history_settings["plotting_settings"]
        results_df = self.get_attribute("_results_df")
        # Initialize the time unit symbol ('s' stands for seconds).
        time_unit_symbol = "s"
        # Generate a figure for each (num_tasks, metric_name) tuple.
        # The figures contain a data point for each (client_selector, num_available_clients) tuple.
        for n_tasks in num_tasks:
            # Update the plotting settings with the current number of tasks.
            plotting_settings.update({"n_tasks": n_tasks})
            for metric_name in metrics_names:
                # Update the plotting settings with the current metric name.
                plotting_settings.update({"metric_name": metric_name})
                # Initialize the plotting data.
                plotting_data = []
                # Iterate through the list of number of available clients.
                for n_available_clients in num_available_clients:
                    # Iterate through the list of client selectors.
                    for client_selector in client_selectors:
                        # Filter the 'results_df' dataframe considering the current
                        # (client_selector, num_tasks, num_available_clients) tuple.
                        filtered_df = results_df[(results_df["client_selector"] == client_selector) &
                                                 (results_df["num_tasks"] == n_tasks) &
                                                 (results_df["num_available_clients"] == n_available_clients)]
                        # If the current metric name equals to 'selection_duration'...
                        if metric_name == "selection_duration":
                            # Get the number of communication rounds ran by the current client selector.
                            num_comm_rounds = len(list(filtered_df["comm_round"].sort_values().unique()))
                            # Calculate the mean value of its selection duration.
                            selection_duration_mean = filtered_df["selection_duration"].sum() / num_comm_rounds
                            # Convert the mean selection duration value to...
                            if time_unit == "milliseconds":
                                # Milliseconds (1 ms = 1e+3 s).
                                time_unit_symbol = "ms"
                                selection_duration_mean = selection_duration_mean * pow(10, 3)
                            elif time_unit == "microseconds":
                                # Microseconds (1 µs = 1e+6 s).
                                time_unit_symbol = "µs"
                                selection_duration_mean = selection_duration_mean * pow(10, 6)
                            # Append the selection duration mean of the current client selector to the plotting data.
                            plotting_data.append({"client_selector": client_selector,
                                                  "num_available_clients": n_available_clients,
                                                  "selection_duration_mean": selection_duration_mean})
                # If the current metric name equals to 'selection_duration'...
                if metric_name == "selection_duration":
                    # Set the 'x_data' value, if equals to 'Auto'.
                    if plotting_settings["x_data"] == "Auto":
                        x_data = "num_available_clients"
                        plotting_settings["x_data"] = x_data
                    # Set the 'y_data' value, if equals to 'Auto'.
                    if plotting_settings["y_data"] == "Auto":
                        y_data = "selection_duration_mean"
                        plotting_settings["y_data"] = y_data
                    # Set the 'x_label' value, if equals to 'Auto'.
                    if plotting_settings["x_label"] == "Auto":
                        x_label = "Number of available clients"
                        plotting_settings["x_label"] = x_label
                    # Set the 'y_label' value, if equals to 'Auto'.
                    if plotting_settings["y_label"] == "Auto":
                        y_scale = plotting_settings["y_scale"]
                        scale_str = ", log scale" if y_scale == "log" else ""
                        y_label = "{0} ({1}{2})".format(metric_name.replace("_", " ").capitalize(),
                                                        time_unit_symbol,
                                                        scale_str)
                        plotting_settings["y_label"] = y_label
                    # Set the 'hue' value, if equals to 'Auto'.
                    if plotting_settings["hue"] == "Auto":
                        hue = "client_selector"
                        plotting_settings["hue"] = hue
                    # Set the 'hue_order' value, if equals to 'Auto'.
                    if plotting_settings["hue_order"] == "Auto":
                        hue_order = client_selectors
                        plotting_settings["hue_order"] = hue_order
                        # Set the 'style' value, if equals to 'Auto'.
                    if plotting_settings["style"] == "Auto":
                        style = "client_selector"
                        plotting_settings["style"] = style
                        # Set the 'size' value, if equals to 'Auto'.
                    if plotting_settings["size"] == "Auto":
                        size = "client_selector"
                        plotting_settings["size"] = size
                        # Update the plotting settings.
                selected_evaluate_clients_history_settings["plotting_settings"] = plotting_settings
                self._set_attribute("_selected_evaluate_clients_history_settings",
                                    selected_evaluate_clients_history_settings)
                # Set the 'plotting_df' dataframe (data that will be plotted into the figure).
                plotting_df = DataFrame(data=plotting_data)
                self._set_attribute("_plotting_df", plotting_df)
                # Load the figure settings for plots.
                self._load_figure_settings()
                # Plot data into the figure.
                self._plot_data()
                # Save the figure.
                self._save_figure()

    def _analyze_selected_evaluate_clients_history_file(self) -> None:
        # Get the necessary attributes.
        input_settings = self.get_attribute("_input_settings")
        analysis_settings = self.get_attribute("_analysis_settings")
        selected_evaluate_clients_history_file = input_settings["selected_evaluate_clients_history_file"]
        analysis_root_folder = Path(analysis_settings["analysis_root_folder"])
        selected_evaluate_clients_history_settings = self.get_attribute("_selected_evaluate_clients_history_settings")
        results_df_settings = selected_evaluate_clients_history_settings["results_df_settings"]
        num_tasks = selected_evaluate_clients_history_settings["num_tasks"]
        num_available_clients = selected_evaluate_clients_history_settings["num_available_clients"]
        client_selectors = selected_evaluate_clients_history_settings["client_selectors"]
        plotting_settings = selected_evaluate_clients_history_settings["plotting_settings"]
        logger = self.get_attribute("_logger")
        # Log an 'analyzing the results' message.
        message = "Analyzing the results in the '{0}' file...".format(selected_evaluate_clients_history_file)
        log_message(logger, message, "INFO")
        # Set the analysis name.
        analysis_name = "selected_evaluate_clients_history"
        self._set_attribute("_analysis_name", analysis_name)
        # Set the analysis result folder.
        analysis_result_folder = analysis_root_folder.joinpath(analysis_name)
        self._set_attribute("_analysis_result_folder", analysis_result_folder)
        # Remove the analysis result folder and its contents (if exists).
        if analysis_result_folder.is_dir():
            rmtree(analysis_result_folder)
        # Create the parents directories of the analysis result folder (if not exist yet).
        analysis_result_folder.parent.mkdir(exist_ok=True, parents=True)
        # Create the analysis result folder.
        analysis_result_folder.mkdir(exist_ok=True, parents=True)
        # Load a dataframe from the results' file.
        results_df = read_csv(filepath_or_buffer=selected_evaluate_clients_history_file,
                              comment=results_df_settings["comments_prefix"])
        # Order the dataframe by client selector's name in ascending order.
        results_df = results_df.sort_values(by=results_df_settings["sort_by"],
                                            ascending=results_df_settings["sort_ascending_order"])
        # Set the 'results_df' dataframe.
        self._set_attribute("_results_df", results_df)

        # TODO: Fix Parser for dict and tuple values for dictionary keys.
        plotting_settings["figure_size"] = (6, 5)
        plotting_settings["line_colors"] = ["#00FFFF", "#FFA500", "#E0115F", "#0000FF", "#7FFFD4", "#228B22"]
        plotting_settings["line_sizes"] = [2, 2, 2, 2, 2, 2]

        # Set the 'num_tasks' value, if equals to 'Auto'.
        if num_tasks == "Auto":
            num_tasks = list(results_df["num_tasks"].sort_values().unique())
            selected_evaluate_clients_history_settings["num_tasks"] = num_tasks
        # Set the 'num_available_clients' value, if equals to 'Auto'.
        if num_available_clients == "Auto":
            num_available_clients = list(results_df["num_available_clients"].sort_values().unique())
            selected_evaluate_clients_history_settings["num_available_clients"] = num_available_clients
        # Set the 'client_selectors' value, if equals to 'Auto'.
        if client_selectors == "Auto":
            client_selectors = list(results_df["client_selector"].sort_values().unique())
            selected_evaluate_clients_history_settings["client_selectors"] = client_selectors
        # Set the 'n_colors' value, if equals to 'Auto'.
        if plotting_settings["n_colors"] == "Auto":
            plotting_settings["n_colors"] = len(client_selectors)
        # Set the 'x_ticks' value, if equals to 'Auto'.
        if plotting_settings["x_ticks"] == "Auto":
            plotting_settings["x_ticks"] = num_available_clients
        # Set the 'y_ticks' value, if equals to 'Auto'.
        if plotting_settings["y_ticks"] == "Auto":
            plotting_settings["y_ticks"] = [pow(10, 0), pow(10, 2), pow(10, 4), pow(10, 6), pow(10, 8), pow(10, 10)]
        # Set the 'y_lim' value, if equals to 'Auto'.
        if plotting_settings["y_lim"] == "Auto":
            plotting_settings["y_lim"] = min(plotting_settings["y_ticks"]), max(plotting_settings["y_ticks"])
        # Update the plotting settings.
        selected_evaluate_clients_history_settings["plotting_settings"] = plotting_settings
        self._set_attribute("_selected_evaluate_clients_history_settings", selected_evaluate_clients_history_settings)
        # Load the theme and palette for plots.
        self._load_theme_and_palette()
        # Generate the figures.
        self._generate_figures_for_selected_evaluate_clients_history_file()

    def _analyze_individual_fit_metrics_history_file(self) -> None:
        pass

    def _analyze_individual_evaluate_metrics_history_file(self) -> None:
        pass

    def _analyze_aggregated_fit_metrics_history_file(self) -> None:
        pass

    def _analyze_aggregated_evaluate_metrics_history_file(self) -> None:
        pass

    def analyze_results(self) -> None:
        # Get the necessary attributes.
        analysis_settings = self.get_attribute("_analysis_settings")
        results_to_analyze = analysis_settings["results_to_analyze"]
        # Iterate through the list of results to analyze.
        for result_to_analyze in results_to_analyze:
            if result_to_analyze == "selected_fit_clients_history_file":
                self._analyze_selected_fit_clients_history_file()
            elif result_to_analyze == "selected_evaluate_clients_history_file":
                self._analyze_selected_evaluate_clients_history_file()
            elif result_to_analyze == "individual_fit_metrics_history_file":
                self._analyze_individual_fit_metrics_history_file()
            elif result_to_analyze == "individual_evaluate_metrics_history_file":
                self._analyze_individual_evaluate_metrics_history_file()
            elif result_to_analyze == "aggregated_fit_metrics_history_file":
                self._analyze_aggregated_fit_metrics_history_file()
            elif result_to_analyze == "aggregated_evaluate_metrics_history_file":
                self._analyze_aggregated_evaluate_metrics_history_file()
