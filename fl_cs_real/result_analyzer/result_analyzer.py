from matplotlib.pyplot import figure, rcParams, savefig, title, xlabel, xlim, xticks, xscale, ylabel, ylim, yticks, \
    yscale
from pandas import DataFrame, read_csv
from pathlib import Path
from seaborn import color_palette, lineplot, set_palette, set_theme
from shutil import rmtree
from warnings import filterwarnings

from fl_cs_real.utils.config_parser_util import parse_config_section
from fl_cs_real.utils.logger_util import load_logger, log_message


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
        self._individual_fit_metrics_history_settings = None
        self._individual_evaluate_metrics_history_settings = None
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
        # Parse and set the individual fit metrics history settings.
        individual_fit_metrics_history_section = "Individual Fit Metrics History Settings"
        individual_fit_metrics_history_settings = parse_config_section(config_file,
                                                                       individual_fit_metrics_history_section)
        self._set_attribute("_individual_fit_metrics_history_settings", individual_fit_metrics_history_settings)
        # Parse and set the individual evaluate metrics history settings.
        individual_evaluate_metrics_history_section = "Individual Evaluate Metrics History Settings"
        individual_evaluate_metrics_history_settings = parse_config_section(config_file,
                                                                            individual_evaluate_metrics_history_section)
        self._set_attribute("_individual_evaluate_metrics_history_settings",
                            individual_evaluate_metrics_history_settings)

    def _set_logger(self) -> None:
        logging_settings = self._logging_settings
        logger_name = type(self).__name__ + "_Logger"
        logger = load_logger(logging_settings, logger_name)
        self._logger = logger

    @staticmethod
    def _load_theme_and_palette(plotting_settings: dict) -> None:
        # Get the necessary attributes.
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

    @staticmethod
    def _convert_time(time_in_seconds: float,
                      time_unit_to_output: str) -> tuple:
        converted_time = None
        converted_time_unit_symbol = None
        if time_unit_to_output == "seconds":
            # Seconds (s).
            converted_time = time_in_seconds
            converted_time_unit_symbol = "s"
        if time_unit_to_output == "milliseconds":
            # Milliseconds (1 ms = 1e+3 s).
            converted_time = time_in_seconds * pow(10, 3)
            converted_time_unit_symbol = "ms"
        elif time_unit_to_output == "microseconds":
            # Microseconds (1 µs = 1e+6 s).
            converted_time = time_in_seconds * pow(10, 6)
            converted_time_unit_symbol = "µs"
        return converted_time, converted_time_unit_symbol

    @staticmethod
    def _load_figure_settings(plotting_settings: dict) -> None:
        # Get the necessary attributes.
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
        x_ticks_label_size = plotting_settings["x_ticks_label_size"]
        x_rotation = plotting_settings["x_rotation"]
        x_lim = plotting_settings["x_lim"]
        x_scale = plotting_settings["x_scale"]
        y_label = plotting_settings["y_label"]
        y_font_size = plotting_settings["y_font_size"]
        y_ticks = plotting_settings["y_ticks"]
        y_ticks_label_size = plotting_settings["y_ticks_label_size"]
        y_rotation = plotting_settings["y_rotation"]
        y_lim = plotting_settings["y_lim"]
        y_scale = plotting_settings["y_scale"]
        # Set the figure settings.
        figure(figsize=figure_size,
               dpi=figure_dpi,
               facecolor=figure_face_color,
               edgecolor=figure_edge_color,
               frameon=figure_frame_on,
               layout=figure_layout,
               clear=True)
        # Set the title settings.
        title(label=title_label,
              fontsize=title_font_size)
        # Set the legend settings.
        rcParams["legend.fontsize"] = legend_font_size
        # Set the X-axis settings.
        xlabel(xlabel=x_label,
               fontsize=x_font_size)
        if x_ticks is not None:
            xticks(ticks=x_ticks,
                   rotation=x_rotation)
        rcParams["xtick.labelsize"] = x_ticks_label_size
        if x_lim is not None:
            xlim(x_lim)
        if x_scale is not None:
            xscale(x_scale)
        # Set the Y-axis settings.
        ylabel(ylabel=y_label,
               fontsize=y_font_size)
        if y_ticks is not None:
            yticks(ticks=y_ticks,
                   rotation=y_rotation)
        rcParams["ytick.labelsize"] = y_ticks_label_size
        if y_lim is not None:
            ylim(y_lim)
        if y_scale is not None:
            yscale(y_scale)

    @staticmethod
    def _plot_data(plotting_settings: dict,
                   plotting_df: DataFrame) -> None:
        # Get the necessary attributes.
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

    def _save_figure(self,
                     plotting_settings: dict,
                     figure_name: str) -> None:
        # Get the necessary attributes.
        figure_bbox_inches = plotting_settings["figure_bbox_inches"]
        analysis_result_folder = Path(self.get_attribute("_analysis_result_folder")).absolute()
        logger = self.get_attribute("_logger")
        # Set the figure's output file.
        figure_output_file = analysis_result_folder.joinpath(figure_name)
        # Save the figure.
        savefig(fname=figure_output_file,
                bbox_inches=figure_bbox_inches)
        # Log a 'figure successfully generated' message.
        message = "Figure '{0}' was successfully generated.".format(figure_output_file)
        log_message(logger, message, "INFO")

    def _generate_figures_for_selected_fit_clients_history_file(self) -> None:
        # Get the necessary attributes.
        analysis_settings = self.get_attribute("_analysis_settings")
        figure_files_extension = analysis_settings["figure_files_extension"]
        selected_fit_clients_history_settings = self.get_attribute("_selected_fit_clients_history_settings")
        num_tasks = selected_fit_clients_history_settings["num_tasks"]
        num_available_clients = selected_fit_clients_history_settings["num_available_clients"]
        comm_rounds = selected_fit_clients_history_settings["comm_rounds"]
        client_selectors = selected_fit_clients_history_settings["client_selectors"]
        metrics_names = selected_fit_clients_history_settings["metrics_names"]
        time_unit_to_output = selected_fit_clients_history_settings["time_unit_to_output"]
        plotting_settings = selected_fit_clients_history_settings["plotting_settings"]
        x_data = plotting_settings["x_data"]
        x_label = plotting_settings["x_label"]
        x_ticks = plotting_settings["x_ticks"]
        y_data = plotting_settings["y_data"]
        y_label = plotting_settings["y_label"]
        y_ticks = plotting_settings["y_ticks"]
        y_lim = plotting_settings["y_lim"]
        y_scale = plotting_settings["y_scale"]
        hue = plotting_settings["hue"]
        hue_order = plotting_settings["hue_order"]
        style = plotting_settings["style"]
        size = plotting_settings["size"]
        analysis_name = self.get_attribute("_analysis_name")
        results_df = self.get_attribute("_results_df")
        # Initialize the time unit symbol.
        time_unit_symbol = None
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
                        # (num_tasks, num_available_clients, client_selector) tuple.
                        filtered_df = results_df[(results_df["num_tasks"] == n_tasks) &
                                                 (results_df["num_available_clients"] == n_available_clients) &
                                                 (results_df["client_selector"] == client_selector)]
                        # If the current metric name equals to 'total_selection_duration'...
                        if metric_name == "total_selection_duration":
                            if "selection_duration" in filtered_df.columns:
                                # Calculate the total selection duration.
                                total_selection_duration = filtered_df["selection_duration"].sum()
                                # Convert the total selection duration, if needed to.
                                total_selection_duration, time_unit_symbol = \
                                    self._convert_time(total_selection_duration,
                                                       time_unit_to_output)
                                # Append the total selection duration to the plotting data.
                                plotting_data.append({"num_available_clients": n_available_clients,
                                                      "client_selector": client_selector,
                                                      "total_selection_duration": total_selection_duration})
                        # If the current metric name equals to 'mean_selection_duration'...
                        elif metric_name == "mean_selection_duration":
                            if all(columns in filtered_df.columns for columns in ["comm_round", "selection_duration"]):
                                # Get the number of communication rounds ran by the current client selector.
                                num_comm_rounds = len(list(filtered_df["comm_round"].sort_values().unique()))
                                # Calculate the mean selection duration.
                                mean_selection_duration = filtered_df["selection_duration"].sum() / num_comm_rounds
                                # Convert the mean selection duration, if needed to.
                                mean_selection_duration, time_unit_symbol = self._convert_time(mean_selection_duration,
                                                                                               time_unit_to_output)
                                # Append the mean selection duration to the plotting data.
                                plotting_data.append({"num_available_clients": n_available_clients,
                                                      "client_selector": client_selector,
                                                      "mean_selection_duration": mean_selection_duration})
                        # If the current metric name equals to 'mean_num_selected_clients'...
                        elif metric_name == "mean_num_selected_clients":
                            if "num_selected_clients" in filtered_df.columns:
                                # Calculate the mean number of selected clients.
                                mean_num_selected_clients = filtered_df["num_selected_clients"].mean()
                                # Append the mean number of selected clients to the plotting data.
                                plotting_data.append({"num_available_clients": n_available_clients,
                                                      "client_selector": client_selector,
                                                      "mean_num_selected_clients": mean_num_selected_clients})
                        # If the current metric name equals to 'num_selected_clients'...
                        elif metric_name == "num_selected_clients":
                            if "num_selected_clients" in filtered_df.columns:
                                # Iterate through the list of communication rounds.
                                for index, _ in enumerate(comm_rounds):
                                    # Get the current communication round.
                                    comm_round = comm_rounds[index]
                                    comm_round_str = "comm_round_{0}".format(comm_round)
                                    # Get the number of selected clients for the current communication round.
                                    comm_round_df = filtered_df.loc[filtered_df['comm_round'] == comm_round_str]
                                    num_selected_clients_comm_round = comm_round_df["num_selected_clients"].values[0]
                                    # Append the number of selected clients per round to the plotting data.
                                    plotting_data.append({"comm_round": comm_round,
                                                          "client_selector": client_selector,
                                                          "num_selected_clients": num_selected_clients_comm_round})
                # If there is data to plot...
                if plotting_data:
                    # If the current metric name equals to 'total_selection_duration'...
                    if metric_name == "total_selection_duration":
                        # Set the 'y_data' value, if equals to 'Auto'.
                        if y_data == "Auto":
                            y_data_new = "total_selection_duration"
                            plotting_settings["y_data"] = y_data_new
                        # Set the 'y_scale' value, if equals to 'Auto'.
                        if y_scale == "Auto":
                            y_scale_new = "log"
                            plotting_settings["y_scale"] = y_scale_new
                        # Set the 'y_label' value, if equals to 'Auto'.
                        if y_label == "Auto":
                            scale_str = ", log scale" if plotting_settings["y_scale"] == "log" else ""
                            y_label_new = "{0} ({1}{2})".format("Total selection duration",
                                                                time_unit_symbol,
                                                                scale_str)
                            plotting_settings["y_label"] = y_label_new
                        # Set the 'y_ticks' value, if equals to 'Auto'.
                        if y_ticks == "Auto":
                            y_ticks_new = None
                            # If the Y-axis scale to be used is 'log'...
                            if plotting_settings["y_scale"] == "log":
                                y_ticks_new = [pow(10, 0), pow(10, 2), pow(10, 4), pow(10, 6), pow(10, 8), pow(10, 10)]
                            plotting_settings["y_ticks"] = y_ticks_new
                        # Set the 'y_lim' value, if equals to 'Auto'.
                        if y_lim == "Auto":
                            y_lim_new = None
                            # If the Y-axis scale to be used is 'log'...
                            if plotting_settings["y_scale"] == "log":
                                y_lim_new = min(plotting_settings["y_ticks"]), max(plotting_settings["y_ticks"])
                            plotting_settings["y_lim"] = y_lim_new
                    # If the current metric name equals to 'mean_selection_duration'...
                    elif metric_name == "mean_selection_duration":
                        # Set the 'y_data' value, if equals to 'Auto'.
                        if y_data == "Auto":
                            y_data_new = "mean_selection_duration"
                            plotting_settings["y_data"] = y_data_new
                        # Set the 'y_scale' value, if equals to 'Auto'.
                        if y_scale == "Auto":
                            y_scale_new = "log"
                            plotting_settings["y_scale"] = y_scale_new
                        # Set the 'y_label' value, if equals to 'Auto'.
                        if y_label == "Auto":
                            scale_str = ", log scale" if plotting_settings["y_scale"] == "log" else ""
                            y_label_new = "{0} ({1}{2})".format("Mean selection duration",
                                                                time_unit_symbol,
                                                                scale_str)
                            plotting_settings["y_label"] = y_label_new
                        # Set the 'y_ticks' value, if equals to 'Auto'.
                        if y_ticks == "Auto":
                            y_ticks_new = None
                            # If the Y-axis scale to be used is 'log'...
                            if plotting_settings["y_scale"] == "log":
                                y_ticks_new = [pow(10, 0), pow(10, 2), pow(10, 4), pow(10, 6), pow(10, 8), pow(10, 10)]
                            plotting_settings["y_ticks"] = y_ticks_new
                        # Set the 'y_lim' value, if equals to 'Auto'.
                        if y_lim == "Auto":
                            y_lim_new = None
                            # If the Y-axis scale to be used is 'log'...
                            if plotting_settings["y_scale"] == "log":
                                y_lim_new = min(plotting_settings["y_ticks"]), max(plotting_settings["y_ticks"])
                            plotting_settings["y_lim"] = y_lim_new
                    # If the current metric name equals to 'mean_num_selected_clients'...
                    elif metric_name == "mean_num_selected_clients":
                        # Set the 'y_data' value, if equals to 'Auto'.
                        if y_data == "Auto":
                            y_data_new = "mean_num_selected_clients"
                            plotting_settings["y_data"] = y_data_new
                        # Set the 'y_scale' value, if equals to 'Auto'.
                        if y_scale == "Auto":
                            y_scale_new = None
                            plotting_settings["y_scale"] = y_scale_new
                        # Set the 'y_label' value, if equals to 'Auto'.
                        if y_label == "Auto":
                            y_label_new = "{0}".format("Mean number of selected clients")
                            plotting_settings["y_label"] = y_label_new
                        # Set the 'y_ticks' value, if equals to 'Auto'.
                        if y_ticks == "Auto":
                            y_ticks_new = None
                            plotting_settings["y_ticks"] = y_ticks_new
                        # Set the 'y_lim' value, if equals to 'Auto'.
                        if y_lim == "Auto":
                            y_lim_new = None
                            plotting_settings["y_lim"] = y_lim_new
                    # If the current metric name equals to 'num_selected_clients'...
                    elif metric_name == "num_selected_clients":
                        # Set the 'y_data' value, if equals to 'Auto'.
                        if y_data == "Auto":
                            y_data_new = "num_selected_clients"
                            plotting_settings["y_data"] = y_data_new
                        # Set the 'y_scale' value, if equals to 'Auto'.
                        if y_scale == "Auto":
                            y_scale_new = None
                            plotting_settings["y_scale"] = y_scale_new
                        # Set the 'y_label' value, if equals to 'Auto'.
                        if y_label == "Auto":
                            y_label_new = "{0}".format("Number of selected clients")
                            plotting_settings["y_label"] = y_label_new
                        # Set the 'y_ticks' value, if equals to 'Auto'.
                        if y_ticks == "Auto":
                            y_ticks_new = None
                            plotting_settings["y_ticks"] = y_ticks_new
                        # Set the 'y_lim' value, if equals to 'Auto'.
                        if y_lim == "Auto":
                            y_lim_new = None
                            plotting_settings["y_lim"] = y_lim_new
                    # Set the 'x_data' value, if equals to 'Auto'.
                    if x_data == "Auto":
                        x_data_new = "num_available_clients"
                        if metric_name == "num_selected_clients":
                            x_data_new = "comm_round"
                        plotting_settings["x_data"] = x_data_new
                    # Set the 'x_label' value, if equals to 'Auto'.
                    if x_label == "Auto":
                        x_label_new = "Number of available clients"
                        if metric_name == "num_selected_clients":
                            x_label_new = "Communication round"
                        plotting_settings["x_label"] = x_label_new
                    # Set the 'x_ticks' value, if equals to 'Auto'.
                    if x_ticks == "Auto":
                        plotting_settings["x_ticks"] = num_available_clients
                        if metric_name == "num_selected_clients":
                            plotting_settings["x_ticks"] = list(range(0, max(comm_rounds) + 1, 10))
                    # Set the 'hue' value, if equals to 'Auto'.
                    if hue == "Auto":
                        hue_new = "client_selector"
                        plotting_settings["hue"] = hue_new
                    # Set the 'hue_order' value, if equals to 'Auto'.
                    if hue_order == "Auto":
                        hue_order_new = client_selectors
                        plotting_settings["hue_order"] = hue_order_new
                        # Set the 'style' value, if equals to 'Auto'.
                    if style == "Auto":
                        style_new = "client_selector"
                        plotting_settings["style"] = style_new
                        # Set the 'size' value, if equals to 'Auto'.
                    if size == "Auto":
                        size_new = "client_selector"
                        plotting_settings["size"] = size_new
                    # Update the plotting settings.
                    selected_fit_clients_history_settings["plotting_settings"] = plotting_settings
                    self._set_attribute("_selected_fit_clients_history_settings", selected_fit_clients_history_settings)
                    # Set the 'plotting_df' dataframe (data that will be plotted into the figure).
                    plotting_df = DataFrame(data=plotting_data)
                    self._set_attribute("_plotting_df", plotting_df)
                    # Load the figure settings for plots.
                    self._load_figure_settings(plotting_settings)
                    # Plot data into the figure.
                    self._plot_data(plotting_settings, plotting_df)
                    # Set the figure name.
                    figure_name = "fig_{0}_{1}_tasks_{2}.{3}" \
                                  .format(analysis_name,
                                          n_tasks,
                                          metric_name.lower(),
                                          figure_files_extension)
                    # Save the figure.
                    self._save_figure(plotting_settings, figure_name)

    def _analyze_selected_fit_clients_history_file(self) -> None:
        # Get the necessary attributes.
        input_settings = self.get_attribute("_input_settings")
        analysis_settings = self.get_attribute("_analysis_settings")
        selected_fit_clients_history_file = Path(input_settings["selected_fit_clients_history_file"]).absolute()
        analysis_root_folder = Path(analysis_settings["analysis_root_folder"]).absolute()
        selected_fit_clients_history_settings = self.get_attribute("_selected_fit_clients_history_settings")
        results_df_settings = selected_fit_clients_history_settings["results_df_settings"]
        num_tasks = selected_fit_clients_history_settings["num_tasks"]
        num_available_clients = selected_fit_clients_history_settings["num_available_clients"]
        comm_rounds = selected_fit_clients_history_settings["comm_rounds"]
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
        plotting_settings["line_colors"] = ["#00FFFF", "#FFA500", "#FF4F4F", "#0000FF",
                                            "#7FFFD4", "#228B22", "#635F4C", "#A82378"]
        plotting_settings["line_sizes"] = [2, 2, 2, 2, 2, 2]

        # Set the 'num_tasks' value, if equals to 'Auto'.
        if num_tasks == "Auto":
            num_tasks = list(results_df["num_tasks"].sort_values().unique())
            selected_fit_clients_history_settings["num_tasks"] = num_tasks
        # Set the 'num_available_clients' value, if equals to 'Auto'.
        if num_available_clients == "Auto":
            num_available_clients = list(results_df["num_available_clients"].sort_values().unique())
            selected_fit_clients_history_settings["num_available_clients"] = num_available_clients
        # Set the 'comm_rounds' value, if equals to 'Auto'.
        if comm_rounds == "Auto":
            comm_rounds = sorted([int(comm_round.replace("comm_round_", ""))
                                  for comm_round in list(results_df["comm_round"].sort_values().unique())])
            selected_fit_clients_history_settings["comm_rounds"] = comm_rounds
        # Set the 'client_selectors' value, if equals to 'Auto'.
        if client_selectors == "Auto":
            client_selectors = list(results_df["client_selector"].sort_values().unique())
            selected_fit_clients_history_settings["client_selectors"] = client_selectors
        # Set the 'n_colors' value, if equals to 'Auto'.
        if plotting_settings["n_colors"] == "Auto":
            plotting_settings["n_colors"] = len(client_selectors)
        # Update the plotting settings.
        selected_fit_clients_history_settings["plotting_settings"] = plotting_settings
        self._set_attribute("_selected_fit_clients_history_settings", selected_fit_clients_history_settings)
        # Load the theme and palette for plots.
        self._load_theme_and_palette(plotting_settings)
        # Generate the figures.
        self._generate_figures_for_selected_fit_clients_history_file()

    def _generate_figures_for_selected_evaluate_clients_history_file(self) -> None:
        # Get the necessary attributes.
        analysis_settings = self.get_attribute("_analysis_settings")
        figure_files_extension = analysis_settings["figure_files_extension"]
        selected_evaluate_clients_history_settings = self.get_attribute("_selected_evaluate_clients_history_settings")
        num_tasks = selected_evaluate_clients_history_settings["num_tasks"]
        num_available_clients = selected_evaluate_clients_history_settings["num_available_clients"]
        comm_rounds = selected_evaluate_clients_history_settings["comm_rounds"]
        client_selectors = selected_evaluate_clients_history_settings["client_selectors"]
        metrics_names = selected_evaluate_clients_history_settings["metrics_names"]
        time_unit_to_output = selected_evaluate_clients_history_settings["time_unit_to_output"]
        plotting_settings = selected_evaluate_clients_history_settings["plotting_settings"]
        x_data = plotting_settings["x_data"]
        x_label = plotting_settings["x_label"]
        x_ticks = plotting_settings["x_ticks"]
        y_data = plotting_settings["y_data"]
        y_label = plotting_settings["y_label"]
        y_ticks = plotting_settings["y_ticks"]
        y_lim = plotting_settings["y_lim"]
        y_scale = plotting_settings["y_scale"]
        hue = plotting_settings["hue"]
        hue_order = plotting_settings["hue_order"]
        style = plotting_settings["style"]
        size = plotting_settings["size"]
        analysis_name = self.get_attribute("_analysis_name")
        results_df = self.get_attribute("_results_df")
        # Initialize the time unit symbol.
        time_unit_symbol = None
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
                        # (num_tasks, num_available_clients, client_selector) tuple.
                        filtered_df = results_df[(results_df["num_tasks"] == n_tasks) &
                                                 (results_df["num_available_clients"] == n_available_clients) &
                                                 (results_df["client_selector"] == client_selector)]
                        # If the current metric name equals to 'total_selection_duration'...
                        if metric_name == "total_selection_duration":
                            if "selection_duration" in filtered_df.columns:
                                # Calculate the total selection duration.
                                total_selection_duration = filtered_df["selection_duration"].sum()
                                # Convert the total selection duration, if needed to.
                                total_selection_duration, time_unit_symbol = \
                                    self._convert_time(total_selection_duration,
                                                       time_unit_to_output)
                                # Append the total selection duration to the plotting data.
                                plotting_data.append({"num_available_clients": n_available_clients,
                                                      "client_selector": client_selector,
                                                      "total_selection_duration": total_selection_duration})
                        # If the current metric name equals to 'mean_selection_duration'...
                        elif metric_name == "mean_selection_duration":
                            if all(columns in filtered_df.columns for columns in ["comm_round", "selection_duration"]):
                                # Get the number of communication rounds ran by the current client selector.
                                num_comm_rounds = len(list(filtered_df["comm_round"].sort_values().unique()))
                                # Calculate the mean selection duration.
                                mean_selection_duration = filtered_df["selection_duration"].sum() / num_comm_rounds
                                # Convert the mean selection duration, if needed to.
                                mean_selection_duration, time_unit_symbol = self._convert_time(mean_selection_duration,
                                                                                               time_unit_to_output)
                                # Append the selection duration mean to the plotting data.
                                plotting_data.append({"num_available_clients": n_available_clients,
                                                      "client_selector": client_selector,
                                                      "mean_selection_duration": mean_selection_duration})
                        # If the current metric name equals to 'mean_num_selected_clients'...
                        elif metric_name == "mean_num_selected_clients":
                            if "num_selected_clients" in filtered_df.columns:
                                # Calculate the mean number of selected clients.
                                mean_num_selected_clients = filtered_df["num_selected_clients"].mean()
                                # Append the mean number of selected clients to the plotting data.
                                plotting_data.append({"num_available_clients": n_available_clients,
                                                      "client_selector": client_selector,
                                                      "mean_num_selected_clients": mean_num_selected_clients})
                        # If the current metric name equals to 'num_selected_clients'...
                        elif metric_name == "num_selected_clients":
                            if "num_selected_clients" in filtered_df.columns:
                                # Iterate through the list of communication rounds.
                                for index, _ in enumerate(comm_rounds):
                                    # Get the current communication round.
                                    comm_round = comm_rounds[index]
                                    comm_round_str = "comm_round_{0}".format(comm_round)
                                    # Get the number of selected clients for the current communication round.
                                    comm_round_df = filtered_df.loc[filtered_df['comm_round'] == comm_round_str]
                                    num_selected_clients_comm_round = comm_round_df["num_selected_clients"].values[0]
                                    # Append the number of selected clients per round to the plotting data.
                                    plotting_data.append({"comm_round": comm_round,
                                                          "client_selector": client_selector,
                                                          "num_selected_clients": num_selected_clients_comm_round})
                # If there is data to plot...
                if plotting_data:
                    # If the current metric name equals to 'total_selection_duration'...
                    if metric_name == "total_selection_duration":
                        # Set the 'y_data' value, if equals to 'Auto'.
                        if y_data == "Auto":
                            y_data_new = "total_selection_duration"
                            plotting_settings["y_data"] = y_data_new
                        # Set the 'y_scale' value, if equals to 'Auto'.
                        if y_scale == "Auto":
                            y_scale_new = "log"
                            plotting_settings["y_scale"] = y_scale_new
                        # Set the 'y_label' value, if equals to 'Auto'.
                        if y_label == "Auto":
                            scale_str = ", log scale" if plotting_settings["y_scale"] == "log" else ""
                            y_label_new = "{0} ({1}{2})".format("Total selection duration",
                                                                time_unit_symbol,
                                                                scale_str)
                            plotting_settings["y_label"] = y_label_new
                        # Set the 'y_ticks' value, if equals to 'Auto'.
                        if y_ticks == "Auto":
                            y_ticks_new = None
                            # If the Y-axis scale to be used is 'log'...
                            if plotting_settings["y_scale"] == "log":
                                y_ticks_new = [pow(10, 0), pow(10, 2), pow(10, 4), pow(10, 6), pow(10, 8), pow(10, 10)]
                            plotting_settings["y_ticks"] = y_ticks_new
                        # Set the 'y_lim' value, if equals to 'Auto'.
                        if y_lim == "Auto":
                            y_lim_new = None
                            # If the Y-axis scale to be used is 'log'...
                            if plotting_settings["y_scale"] == "log":
                                y_lim_new = min(plotting_settings["y_ticks"]), max(plotting_settings["y_ticks"])
                            plotting_settings["y_lim"] = y_lim_new
                    # If the current metric name equals to 'mean_selection_duration'...
                    elif metric_name == "mean_selection_duration":
                        # Set the 'y_data' value, if equals to 'Auto'.
                        if y_data == "Auto":
                            y_data_new = "mean_selection_duration"
                            plotting_settings["y_data"] = y_data_new
                        # Set the 'y_scale' value, if equals to 'Auto'.
                        if y_scale == "Auto":
                            y_scale_new = "log"
                            plotting_settings["y_scale"] = y_scale_new
                        # Set the 'y_label' value, if equals to 'Auto'.
                        if y_label == "Auto":
                            scale_str = ", log scale" if plotting_settings["y_scale"] == "log" else ""
                            y_label_new = "{0} ({1}{2})".format("Mean selection duration",
                                                                time_unit_symbol,
                                                                scale_str)
                            plotting_settings["y_label"] = y_label_new
                        # Set the 'y_ticks' value, if equals to 'Auto'.
                        if y_ticks == "Auto":
                            y_ticks_new = None
                            # If the Y-axis scale to be used is 'log'...
                            if plotting_settings["y_scale"] == "log":
                                y_ticks_new = [pow(10, 0), pow(10, 2), pow(10, 4), pow(10, 6), pow(10, 8), pow(10, 10)]
                            plotting_settings["y_ticks"] = y_ticks_new
                        # Set the 'y_lim' value, if equals to 'Auto'.
                        if y_lim == "Auto":
                            y_lim_new = None
                            # If the Y-axis scale to be used is 'log'...
                            if plotting_settings["y_scale"] == "log":
                                y_lim_new = min(plotting_settings["y_ticks"]), max(plotting_settings["y_ticks"])
                            plotting_settings["y_lim"] = y_lim_new
                    # If the current metric name equals to 'mean_num_selected_clients'...
                    elif metric_name == "mean_num_selected_clients":
                        # Set the 'y_data' value, if equals to 'Auto'.
                        if y_data == "Auto":
                            y_data_new = "mean_num_selected_clients"
                            plotting_settings["y_data"] = y_data_new
                        # Set the 'y_scale' value, if equals to 'Auto'.
                        if y_scale == "Auto":
                            y_scale_new = None
                            plotting_settings["y_scale"] = y_scale_new
                        # Set the 'y_label' value, if equals to 'Auto'.
                        if y_label == "Auto":
                            y_label_new = "{0}".format("Mean number of selected clients")
                            plotting_settings["y_label"] = y_label_new
                        # Set the 'y_ticks' value, if equals to 'Auto'.
                        if y_ticks == "Auto":
                            y_ticks_new = None
                            plotting_settings["y_ticks"] = y_ticks_new
                        # Set the 'y_lim' value, if equals to 'Auto'.
                        if y_lim == "Auto":
                            y_lim_new = None
                            plotting_settings["y_lim"] = y_lim_new
                    # If the current metric name equals to 'num_selected_clients'...
                    elif metric_name == "num_selected_clients":
                        # Set the 'y_data' value, if equals to 'Auto'.
                        if y_data == "Auto":
                            y_data_new = "num_selected_clients"
                            plotting_settings["y_data"] = y_data_new
                        # Set the 'y_scale' value, if equals to 'Auto'.
                        if y_scale == "Auto":
                            y_scale_new = None
                            plotting_settings["y_scale"] = y_scale_new
                        # Set the 'y_label' value, if equals to 'Auto'.
                        if y_label == "Auto":
                            y_label_new = "{0}".format("Number of selected clients")
                            plotting_settings["y_label"] = y_label_new
                        # Set the 'y_ticks' value, if equals to 'Auto'.
                        if y_ticks == "Auto":
                            y_ticks_new = None
                            plotting_settings["y_ticks"] = y_ticks_new
                        # Set the 'y_lim' value, if equals to 'Auto'.
                        if y_lim == "Auto":
                            y_lim_new = None
                            plotting_settings["y_lim"] = y_lim_new
                    # Set the 'x_data' value, if equals to 'Auto'.
                    if x_data == "Auto":
                        x_data_new = "num_available_clients"
                        if metric_name == "num_selected_clients":
                            x_data_new = "comm_round"
                        plotting_settings["x_data"] = x_data_new
                    # Set the 'x_label' value, if equals to 'Auto'.
                    if x_label == "Auto":
                        x_label_new = "Number of available clients"
                        if metric_name == "num_selected_clients":
                            x_label_new = "Communication round"
                        plotting_settings["x_label"] = x_label_new
                    # Set the 'x_ticks' value, if equals to 'Auto'.
                    if x_ticks == "Auto":
                        plotting_settings["x_ticks"] = num_available_clients
                        if metric_name == "num_selected_clients":
                            plotting_settings["x_ticks"] = list(range(0, max(comm_rounds) + 1, 10))
                    # Set the 'hue' value, if equals to 'Auto'.
                    if hue == "Auto":
                        hue_new = "client_selector"
                        plotting_settings["hue"] = hue_new
                    # Set the 'hue_order' value, if equals to 'Auto'.
                    if hue_order == "Auto":
                        hue_order_new = client_selectors
                        plotting_settings["hue_order"] = hue_order_new
                        # Set the 'style' value, if equals to 'Auto'.
                    if style == "Auto":
                        style_new = "client_selector"
                        plotting_settings["style"] = style_new
                        # Set the 'size' value, if equals to 'Auto'.
                    if size == "Auto":
                        size_new = "client_selector"
                        plotting_settings["size"] = size_new
                    # Update the plotting settings.
                    selected_evaluate_clients_history_settings["plotting_settings"] = plotting_settings
                    self._set_attribute("_selected_evaluate_clients_history_settings",
                                        selected_evaluate_clients_history_settings)
                    # Set the 'plotting_df' dataframe (data that will be plotted into the figure).
                    plotting_df = DataFrame(data=plotting_data)
                    self._set_attribute("_plotting_df", plotting_df)
                    # Load the figure settings for plots.
                    self._load_figure_settings(plotting_settings)
                    # Plot data into the figure.
                    self._plot_data(plotting_settings, plotting_df)
                    # Set the figure name.
                    figure_name = "fig_{0}_{1}_tasks_{2}.{3}" \
                                  .format(analysis_name,
                                          n_tasks,
                                          metric_name.lower(),
                                          figure_files_extension)
                    # Save the figure.
                    self._save_figure(plotting_settings, figure_name)

    def _analyze_selected_evaluate_clients_history_file(self) -> None:
        # Get the necessary attributes.
        input_settings = self.get_attribute("_input_settings")
        analysis_settings = self.get_attribute("_analysis_settings")
        selected_evaluate_clients_history_file = \
            Path(input_settings["selected_evaluate_clients_history_file"]).absolute()
        analysis_root_folder = Path(analysis_settings["analysis_root_folder"]).absolute()
        selected_evaluate_clients_history_settings = self.get_attribute("_selected_evaluate_clients_history_settings")
        results_df_settings = selected_evaluate_clients_history_settings["results_df_settings"]
        num_tasks = selected_evaluate_clients_history_settings["num_tasks"]
        num_available_clients = selected_evaluate_clients_history_settings["num_available_clients"]
        comm_rounds = selected_evaluate_clients_history_settings["comm_rounds"]
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
        plotting_settings["line_colors"] = ["#00FFFF", "#FFA500", "#FF4F4F", "#0000FF",
                                            "#7FFFD4", "#228B22", "#635F4C", "#A82378"]
        plotting_settings["line_sizes"] = [2, 2, 2, 2, 2, 2]

        # Set the 'num_tasks' value, if equals to 'Auto'.
        if num_tasks == "Auto":
            num_tasks = list(results_df["num_tasks"].sort_values().unique())
            selected_evaluate_clients_history_settings["num_tasks"] = num_tasks
        # Set the 'num_available_clients' value, if equals to 'Auto'.
        if num_available_clients == "Auto":
            num_available_clients = list(results_df["num_available_clients"].sort_values().unique())
            selected_evaluate_clients_history_settings["num_available_clients"] = num_available_clients
        # Set the 'comm_rounds' value, if equals to 'Auto'.
        if comm_rounds == "Auto":
            comm_rounds = sorted([int(comm_round.replace("comm_round_", ""))
                                  for comm_round in list(results_df["comm_round"].sort_values().unique())])
            selected_evaluate_clients_history_settings["comm_rounds"] = comm_rounds
        # Set the 'client_selectors' value, if equals to 'Auto'.
        if client_selectors == "Auto":
            client_selectors = list(results_df["client_selector"].sort_values().unique())
            selected_evaluate_clients_history_settings["client_selectors"] = client_selectors
        # Set the 'n_colors' value, if equals to 'Auto'.
        if plotting_settings["n_colors"] == "Auto":
            plotting_settings["n_colors"] = len(client_selectors)
        # Update the plotting settings.
        selected_evaluate_clients_history_settings["plotting_settings"] = plotting_settings
        self._set_attribute("_selected_evaluate_clients_history_settings", selected_evaluate_clients_history_settings)
        # Load the theme and palette for plots.
        self._load_theme_and_palette(plotting_settings)
        # Generate the figures.
        self._generate_figures_for_selected_evaluate_clients_history_file()

    def _generate_figures_for_individual_fit_metrics_history_file(self) -> None:
        # Get the necessary attributes.
        analysis_settings = self.get_attribute("_analysis_settings")
        figure_files_extension = analysis_settings["figure_files_extension"]
        selected_fit_clients_history_settings = self.get_attribute("_selected_fit_clients_history_settings")
        num_tasks = selected_fit_clients_history_settings["num_tasks"]
        num_available_clients = selected_fit_clients_history_settings["num_available_clients"]
        comm_rounds = selected_fit_clients_history_settings["comm_rounds"]
        client_selectors = selected_fit_clients_history_settings["client_selectors"]
        metrics_names = selected_fit_clients_history_settings["metrics_names"]
        time_unit_to_output = selected_fit_clients_history_settings["time_unit_to_output"]
        plotting_settings = selected_fit_clients_history_settings["plotting_settings"]
        x_data = plotting_settings["x_data"]
        x_label = plotting_settings["x_label"]
        y_data = plotting_settings["y_data"]
        y_label = plotting_settings["y_label"]
        hue = plotting_settings["hue"]
        hue_order = plotting_settings["hue_order"]
        style = plotting_settings["style"]
        size = plotting_settings["size"]
        analysis_name = self.get_attribute("_analysis_name")
        results_df = self.get_attribute("_results_df")
        # Initialize the time unit symbol.
        time_unit_symbol = None
        # Generate a figure for each (num_tasks, num_available_clients, metric_name) tuple.
        # The figures contain a data point for each (client_selector, comm_round) tuple.
        for n_tasks in num_tasks:
            # Update the plotting settings with the current number of tasks.
            plotting_settings.update({"n_tasks": n_tasks})
            for metric_name in metrics_names:
                # Update the plotting settings with the current metric name.
                plotting_settings.update({"metric_name": metric_name})
                # Iterate through the list of number of available clients.
                for n_available_clients in num_available_clients:
                    # Initialize the plotting data.
                    plotting_data = []
                    # Iterate through the list of communication rounds.
                    for comm_round in comm_rounds:
                        # Iterate through the list of client selectors.
                        for client_selector in client_selectors:
                            # Filter the 'results_df' dataframe considering the current
                            # (num_tasks, num_available_clients, comm_round, client_selector) tuple.
                            filtered_df = results_df[(results_df["num_tasks"] == n_tasks) &
                                                     (results_df["num_available_clients"] == n_available_clients) &
                                                     (results_df["comm_round"] == "comm_round_{0}".format(comm_round)) &
                                                     (results_df["client_selector"] == client_selector)]
                            # If the current metric name equals to 'max_training_elapsed_time'...
                            if metric_name == "max_training_elapsed_time":
                                if "training_elapsed_time" in filtered_df.columns:
                                    # Get the maximum training elapsed time among the participating clients (Makespan).
                                    max_training_elapsed_time = filtered_df["training_elapsed_time"].max()
                                    # Convert the maximum training elapsed time, if needed to.
                                    max_training_elapsed_time, time_unit_symbol \
                                        = self._convert_time(max_training_elapsed_time, time_unit_to_output)
                                    # Append the maximum training elapsed time to the plotting data.
                                    plotting_data.append({"comm_round": comm_round,
                                                          "client_selector": client_selector,
                                                          "max_training_elapsed_time": max_training_elapsed_time})
                            # If the current metric name equals to 'sum_training_energy_cpu'...
                            elif metric_name == "sum_training_energy_cpu":
                                if "training_energy_cpu" in filtered_df.columns:
                                    # Get the sum of CPU energy consumption.
                                    sum_training_energy_cpu = filtered_df["training_energy_cpu"].sum()
                                    # Append the sum of CPU energy consumption to the plotting data.
                                    plotting_data.append({"comm_round": comm_round,
                                                          "client_selector": client_selector,
                                                          "sum_training_energy_cpu": sum_training_energy_cpu})
                            # If the current metric name equals to 'mean_num_training_examples_used'...
                            elif metric_name == "mean_num_training_examples_used":
                                if "num_training_examples_used" in filtered_df.columns:
                                    # Calculate the mean number of training examples used.
                                    mean_num_training_examples_used = filtered_df["num_training_examples_used"].mean()
                                    # Append the mean number of training examples used to the plotting data.
                                    plotting_data.append({"comm_round": comm_round,
                                                          "client_selector": client_selector,
                                                          "mean_num_training_examples_used":
                                                          mean_num_training_examples_used})
                            # If the current metric name equals to 'mean_sparse_categorical_accuracy'...
                            elif metric_name == "mean_sparse_categorical_accuracy":
                                if "sparse_categorical_accuracy" in filtered_df.columns:
                                    # Calculate the mean sparse categorical accuracy.
                                    mean_sparse_categorical_accuracy = filtered_df["sparse_categorical_accuracy"].mean()
                                    # Append the mean sparse categorical accuracy to the plotting data.
                                    plotting_data.append({"comm_round": comm_round,
                                                          "client_selector": client_selector,
                                                          "mean_sparse_categorical_accuracy":
                                                          mean_sparse_categorical_accuracy})
                            # If the current metric name equals to 'weighted_mean_sparse_categorical_accuracy'...
                            elif metric_name == "weighted_mean_sparse_categorical_accuracy":
                                if "sparse_categorical_accuracy" in filtered_df.columns:
                                    # Calculate the weighted mean sparse categorical accuracy.
                                    num_training_examples_used = filtered_df["num_training_examples_used"]
                                    sparse_categorical_accuracy = filtered_df["sparse_categorical_accuracy"]
                                    sum_sparse_categorical_accuracy_product \
                                        = (num_training_examples_used * sparse_categorical_accuracy).sum()
                                    sum_num_training_examples_used = filtered_df["num_training_examples_used"].sum()
                                    weighted_mean_sparse_categorical_accuracy \
                                        = sum_sparse_categorical_accuracy_product / sum_num_training_examples_used
                                    # Append the weighted mean sparse categorical accuracy to the plotting data.
                                    plotting_data.append({"comm_round": comm_round,
                                                          "client_selector": client_selector,
                                                          "weighted_mean_sparse_categorical_accuracy":
                                                          weighted_mean_sparse_categorical_accuracy})
                            # If the current metric name equals to 'max_communication_time'...
                            elif metric_name == "max_communication_time":
                                if "communication_time" in filtered_df.columns:
                                    # Get the maximum communication time among the participating clients.
                                    max_communication_time = filtered_df["communication_time"].max()
                                    # Convert the maximum communication time, if needed to.
                                    max_communication_time, time_unit_symbol \
                                        = self._convert_time(max_communication_time, time_unit_to_output)
                                    # Append the maximum communication time to the plotting data.
                                    plotting_data.append({"comm_round": comm_round,
                                                          "client_selector": client_selector,
                                                          "max_communication_time": max_communication_time})
                    # If there is data to plot...
                    if plotting_data:
                        # If the current metric name equals to 'max_training_elapsed_time'...
                        if metric_name == "max_training_elapsed_time":
                            # Set the 'y_data' value, if equals to 'Auto'.
                            if y_data == "Auto":
                                y_data_new = "max_training_elapsed_time"
                                plotting_settings["y_data"] = y_data_new
                            # Set the 'y_label' value, if equals to 'Auto'.
                            if y_label == "Auto":
                                y_scale_new = None
                                plotting_settings["y_scale"] = y_scale_new
                                y_label_new = "{0} ({1})".format("Makespan", time_unit_symbol)
                                plotting_settings["y_label"] = y_label_new
                        # If the current metric name equals to 'sum_training_energy_cpu'...
                        elif metric_name == "sum_training_energy_cpu":
                            # Set the 'y_data' value, if equals to 'Auto'.
                            if y_data == "Auto":
                                y_data_new = "sum_training_energy_cpu"
                                plotting_settings["y_data"] = y_data_new
                            # Set the 'y_label' value, if equals to 'Auto'.
                            if y_label == "Auto":
                                y_scale_new = None
                                plotting_settings["y_scale"] = y_scale_new
                                y_label_new = "{0} ({1})".format("Sum of CPU energy consumption", "J")
                                plotting_settings["y_label"] = y_label_new
                        # If the current metric name equals to 'mean_num_training_examples_used'...
                        elif metric_name == "mean_num_training_examples_used":
                            # Set the 'y_data' value, if equals to 'Auto'.
                            if y_data == "Auto":
                                y_data_new = "mean_num_training_examples_used"
                                plotting_settings["y_data"] = y_data_new
                            # Set the 'y_label' value, if equals to 'Auto'.
                            if y_label == "Auto":
                                y_scale_new = None
                                plotting_settings["y_scale"] = y_scale_new
                                y_label_new = "{0}".format("Mean number of training examples used")
                                plotting_settings["y_label"] = y_label_new
                        # If the current metric name equals to 'mean_sparse_categorical_accuracy'...
                        elif metric_name == "mean_sparse_categorical_accuracy":
                            # Set the 'y_data' value, if equals to 'Auto'.
                            if y_data == "Auto":
                                y_data_new = "mean_sparse_categorical_accuracy"
                                plotting_settings["y_data"] = y_data_new
                            # Set the 'y_label' value, if equals to 'Auto'.
                            if y_label == "Auto":
                                y_scale_new = None
                                plotting_settings["y_scale"] = y_scale_new
                                y_label_new = "{0}".format("Mean accuracy")
                                plotting_settings["y_label"] = y_label_new
                        # If the current metric name equals to 'weighted_mean_sparse_categorical_accuracy'...
                        elif metric_name == "weighted_mean_sparse_categorical_accuracy":
                            # Set the 'y_data' value, if equals to 'Auto'.
                            if y_data == "Auto":
                                y_data_new = "weighted_mean_sparse_categorical_accuracy"
                                plotting_settings["y_data"] = y_data_new
                            # Set the 'y_label' value, if equals to 'Auto'.
                            if y_label == "Auto":
                                y_scale_new = None
                                plotting_settings["y_scale"] = y_scale_new
                                y_label_new = "{0}".format("Weighted mean accuracy")
                                plotting_settings["y_label"] = y_label_new
                        # If the current metric name equals to 'max_communication_time'...
                        elif metric_name == "max_communication_time":
                            # Set the 'y_data' value, if equals to 'Auto'.
                            if y_data == "Auto":
                                y_data_new = "max_communication_time"
                                plotting_settings["y_data"] = y_data_new
                            # Set the 'y_label' value, if equals to 'Auto'.
                            if y_label == "Auto":
                                y_scale_new = None
                                plotting_settings["y_scale"] = y_scale_new
                                y_label_new = "{0} ({1})".format("Max. Communication Time", time_unit_symbol)
                                plotting_settings["y_label"] = y_label_new
                        # Set the 'x_data' value, if equals to 'Auto'.
                        if x_data == "Auto":
                            x_data_new = "comm_round"
                            plotting_settings["x_data"] = x_data_new
                        # Set the 'x_label' value, if equals to 'Auto'.
                        if x_label == "Auto":
                            x_label_new = "Communication round"
                            plotting_settings["x_label"] = x_label_new
                        # Set the 'hue' value, if equals to 'Auto'.
                        if hue == "Auto":
                            hue_new = "client_selector"
                            plotting_settings["hue"] = hue_new
                        # Set the 'hue_order' value, if equals to 'Auto'.
                        if hue_order == "Auto":
                            hue_order_new = client_selectors
                            plotting_settings["hue_order"] = hue_order_new
                            # Set the 'style' value, if equals to 'Auto'.
                        if style == "Auto":
                            style_new = "client_selector"
                            plotting_settings["style"] = style_new
                            # Set the 'size' value, if equals to 'Auto'.
                        if size == "Auto":
                            size_new = "client_selector"
                            plotting_settings["size"] = size_new
                        # Update the plotting settings.
                        selected_fit_clients_history_settings["plotting_settings"] = plotting_settings
                        self._set_attribute("_selected_fit_clients_history_settings",
                                            selected_fit_clients_history_settings)
                        # Set the 'plotting_df' dataframe (data that will be plotted into the figure).
                        plotting_df = DataFrame(data=plotting_data)
                        self._set_attribute("_plotting_df", plotting_df)
                        # Load the figure settings for plots.
                        self._load_figure_settings(plotting_settings)
                        # Plot data into the figure.
                        self._plot_data(plotting_settings, plotting_df)
                        # Set the figure name.
                        figure_name = "fig_{0}_{1}_tasks_{2}_clients_{3}.{4}" \
                                      .format(analysis_name,
                                              n_tasks,
                                              n_available_clients,
                                              metric_name.lower(),
                                              figure_files_extension)
                        # Save the figure.
                        self._save_figure(plotting_settings, figure_name)

    def _analyze_individual_fit_metrics_history_file(self) -> None:
        # Get the necessary attributes.
        input_settings = self.get_attribute("_input_settings")
        analysis_settings = self.get_attribute("_analysis_settings")
        individual_fit_metrics_history_file = Path(input_settings["individual_fit_metrics_history_file"]).absolute()
        analysis_root_folder = Path(analysis_settings["analysis_root_folder"]).absolute()
        individual_fit_metrics_history_settings = self.get_attribute("_individual_fit_metrics_history_settings")
        results_df_settings = individual_fit_metrics_history_settings["results_df_settings"]
        num_tasks = individual_fit_metrics_history_settings["num_tasks"]
        num_available_clients = individual_fit_metrics_history_settings["num_available_clients"]
        comm_rounds = individual_fit_metrics_history_settings["comm_rounds"]
        client_selectors = individual_fit_metrics_history_settings["client_selectors"]
        plotting_settings = individual_fit_metrics_history_settings["plotting_settings"]
        logger = self.get_attribute("_logger")
        # Log an 'analyzing the results' message.
        message = "Analyzing the results in the '{0}' file...".format(individual_fit_metrics_history_file)
        log_message(logger, message, "INFO")
        # Set the analysis name.
        analysis_name = "individual_fit_metrics_history"
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
        results_df = read_csv(filepath_or_buffer=individual_fit_metrics_history_file,
                              comment=results_df_settings["comments_prefix"])
        # Order the dataframe by client selector's name in ascending order.
        results_df = results_df.sort_values(by=results_df_settings["sort_by"],
                                            ascending=results_df_settings["sort_ascending_order"])
        # Set the 'results_df' dataframe.
        self._set_attribute("_results_df", results_df)

        # TODO: Fix Parser for dict and tuple values for dictionary keys.
        plotting_settings["figure_size"] = (6, 5)
        plotting_settings["line_colors"] = ["#00FFFF", "#FFA500", "#FF4F4F", "#0000FF",
                                            "#7FFFD4", "#228B22", "#635F4C", "#A82378"]
        plotting_settings["line_sizes"] = [2, 2, 2, 2, 2, 2]

        # Set the 'num_tasks' value, if equals to 'Auto'.
        if num_tasks == "Auto":
            num_tasks = list(results_df["num_tasks"].sort_values().unique())
            individual_fit_metrics_history_settings["num_tasks"] = num_tasks
        # Set the 'num_available_clients' value, if equals to 'Auto'.
        if num_available_clients == "Auto":
            num_available_clients = list(results_df["num_available_clients"].sort_values().unique())
            individual_fit_metrics_history_settings["num_available_clients"] = num_available_clients
        # Set the 'comm_rounds' value, if equals to 'Auto'.
        if comm_rounds == "Auto":
            comm_rounds = sorted([int(comm_round.replace("comm_round_", ""))
                                  for comm_round in list(results_df["comm_round"].sort_values().unique())])
            individual_fit_metrics_history_settings["comm_rounds"] = comm_rounds
        # Set the 'client_selectors' value, if equals to 'Auto'.
        if client_selectors == "Auto":
            client_selectors = list(results_df["client_selector"].sort_values().unique())
            individual_fit_metrics_history_settings["client_selectors"] = client_selectors
        # Set the 'n_colors' value, if equals to 'Auto'.
        if plotting_settings["n_colors"] == "Auto":
            plotting_settings["n_colors"] = len(client_selectors)
        # Set the 'x_ticks' value, if equals to 'Auto'.
        if plotting_settings["x_ticks"] == "Auto":
            plotting_settings["x_ticks"] = list(range(0, max(comm_rounds)+1, 10))
        # Update the plotting settings.
        individual_fit_metrics_history_settings["plotting_settings"] = plotting_settings
        self._set_attribute("_selected_fit_clients_history_settings", individual_fit_metrics_history_settings)
        # Load the theme and palette for plots.
        self._load_theme_and_palette(plotting_settings)
        # Generate the figures.
        self._generate_figures_for_individual_fit_metrics_history_file()

    def _generate_figures_for_individual_evaluate_metrics_history_file(self) -> None:
        # Get the necessary attributes.
        analysis_settings = self.get_attribute("_analysis_settings")
        figure_files_extension = analysis_settings["figure_files_extension"]
        individual_evaluate_metrics_history_settings = \
            self.get_attribute("_individual_evaluate_metrics_history_settings")
        num_tasks = individual_evaluate_metrics_history_settings["num_tasks"]
        num_available_clients = individual_evaluate_metrics_history_settings["num_available_clients"]
        comm_rounds = individual_evaluate_metrics_history_settings["comm_rounds"]
        client_selectors = individual_evaluate_metrics_history_settings["client_selectors"]
        metrics_names = individual_evaluate_metrics_history_settings["metrics_names"]
        time_unit_to_output = individual_evaluate_metrics_history_settings["time_unit_to_output"]
        plotting_settings = individual_evaluate_metrics_history_settings["plotting_settings"]
        x_data = plotting_settings["x_data"]
        x_label = plotting_settings["x_label"]
        y_data = plotting_settings["y_data"]
        y_label = plotting_settings["y_label"]
        hue = plotting_settings["hue"]
        hue_order = plotting_settings["hue_order"]
        style = plotting_settings["style"]
        size = plotting_settings["size"]
        analysis_name = self.get_attribute("_analysis_name")
        results_df = self.get_attribute("_results_df")
        # Initialize the time unit symbol.
        time_unit_symbol = None
        # Generate a figure for each (num_tasks, num_available_clients, metric_name) tuple.
        # The figures contain a data point for each (client_selector, comm_round) tuple.
        for n_tasks in num_tasks:
            # Update the plotting settings with the current number of tasks.
            plotting_settings.update({"n_tasks": n_tasks})
            for metric_name in metrics_names:
                # Update the plotting settings with the current metric name.
                plotting_settings.update({"metric_name": metric_name})
                # Iterate through the list of number of available clients.
                for n_available_clients in num_available_clients:
                    # Initialize the plotting data.
                    plotting_data = []
                    # Iterate through the list of communication rounds.
                    for comm_round in comm_rounds:
                        # Iterate through the list of client selectors.
                        for client_selector in client_selectors:
                            # Filter the 'results_df' dataframe considering the current
                            # (num_tasks, num_available_clients, comm_round, client_selector) tuple.
                            filtered_df = results_df[(results_df["num_tasks"] == n_tasks) &
                                                     (results_df["num_available_clients"] == n_available_clients) &
                                                     (results_df["comm_round"] == "comm_round_{0}".format(comm_round)) &
                                                     (results_df["client_selector"] == client_selector)]
                            # If the current metric name equals to 'max_testing_elapsed_time'...
                            if metric_name == "max_testing_elapsed_time":
                                if "testing_elapsed_time" in filtered_df.columns:
                                    # Get the maximum testing elapsed time among the participating clients (Makespan).
                                    max_testing_elapsed_time = filtered_df["testing_elapsed_time"].max()
                                    # Convert the maximum testing elapsed time, if needed to.
                                    max_testing_elapsed_time, time_unit_symbol \
                                        = self._convert_time(max_testing_elapsed_time, time_unit_to_output)
                                    # Append the maximum testing elapsed time to the plotting data.
                                    plotting_data.append({"comm_round": comm_round,
                                                          "client_selector": client_selector,
                                                          "max_testing_elapsed_time": max_testing_elapsed_time})
                            # If the current metric name equals to 'sum_testing_energy_cpu'...
                            elif metric_name == "sum_testing_energy_cpu":
                                if "testing_energy_cpu" in filtered_df.columns:
                                    # Get the sum of CPU energy consumption.
                                    sum_testing_energy_cpu = filtered_df["testing_energy_cpu"].sum()
                                    # Append the sum of CPU energy consumption to the plotting data.
                                    plotting_data.append({"comm_round": comm_round,
                                                          "client_selector": client_selector,
                                                          "sum_testing_energy_cpu": sum_testing_energy_cpu})
                            # If the current metric name equals to 'mean_num_testing_examples_used'...
                            elif metric_name == "mean_num_testing_examples_used":
                                if "num_testing_examples_used" in filtered_df.columns:
                                    # Calculate the mean number of testing examples used.
                                    mean_num_testing_examples_used = filtered_df["num_testing_examples_used"].mean()
                                    # Append the mean number of testing examples used to the plotting data.
                                    plotting_data.append({"comm_round": comm_round,
                                                          "client_selector": client_selector,
                                                          "mean_num_testing_examples_used":
                                                          mean_num_testing_examples_used})
                            # If the current metric name equals to 'mean_sparse_categorical_accuracy'...
                            elif metric_name == "mean_sparse_categorical_accuracy":
                                if "sparse_categorical_accuracy" in filtered_df.columns:
                                    # Calculate the mean sparse categorical accuracy.
                                    mean_sparse_categorical_accuracy = filtered_df["sparse_categorical_accuracy"].mean()
                                    # Append the mean sparse categorical accuracy to the plotting data.
                                    plotting_data.append({"comm_round": comm_round,
                                                          "client_selector": client_selector,
                                                          "mean_sparse_categorical_accuracy":
                                                          mean_sparse_categorical_accuracy})
                            # If the current metric name equals to 'weighted_mean_sparse_categorical_accuracy'...
                            elif metric_name == "weighted_mean_sparse_categorical_accuracy":
                                if "sparse_categorical_accuracy" in filtered_df.columns:
                                    # Calculate the weighted mean sparse categorical accuracy.
                                    num_testing_examples_used = filtered_df["num_testing_examples_used"]
                                    sparse_categorical_accuracy = filtered_df["sparse_categorical_accuracy"]
                                    sum_sparse_categorical_accuracy_product \
                                        = (num_testing_examples_used * sparse_categorical_accuracy).sum()
                                    sum_num_testing_examples_used = filtered_df["num_testing_examples_used"].sum()
                                    weighted_mean_sparse_categorical_accuracy \
                                        = sum_sparse_categorical_accuracy_product / sum_num_testing_examples_used
                                    # Append the weighted mean sparse categorical accuracy to the plotting data.
                                    plotting_data.append({"comm_round": comm_round,
                                                          "client_selector": client_selector,
                                                          "weighted_mean_sparse_categorical_accuracy":
                                                          weighted_mean_sparse_categorical_accuracy})
                            # If the current metric name equals to 'max_communication_time'...
                            elif metric_name == "max_communication_time":
                                if "communication_time" in filtered_df.columns:
                                    # Get the maximum communication time among the participating clients.
                                    max_communication_time = filtered_df["communication_time"].max()
                                    # Convert the maximum communication time, if needed to.
                                    max_communication_time, time_unit_symbol \
                                        = self._convert_time(max_communication_time, time_unit_to_output)
                                    # Append the maximum communication time to the plotting data.
                                    plotting_data.append({"comm_round": comm_round,
                                                          "client_selector": client_selector,
                                                          "max_communication_time": max_communication_time})
                    # If there is data to plot...
                    if plotting_data:
                        # If the current metric name equals to 'max_testing_elapsed_time'...
                        if metric_name == "max_testing_elapsed_time":
                            # Set the 'y_data' value, if equals to 'Auto'.
                            if y_data == "Auto":
                                y_data_new = "max_testing_elapsed_time"
                                plotting_settings["y_data"] = y_data_new
                            # Set the 'y_label' value, if equals to 'Auto'.
                            if y_label == "Auto":
                                y_scale_new = None
                                plotting_settings["y_scale"] = y_scale_new
                                y_label_new = "{0} ({1})".format("Makespan", time_unit_symbol)
                                plotting_settings["y_label"] = y_label_new
                        # If the current metric name equals to 'sum_testing_energy_cpu'...
                        elif metric_name == "sum_testing_energy_cpu":
                            # Set the 'y_data' value, if equals to 'Auto'.
                            if y_data == "Auto":
                                y_data_new = "sum_testing_energy_cpu"
                                plotting_settings["y_data"] = y_data_new
                            # Set the 'y_label' value, if equals to 'Auto'.
                            if y_label == "Auto":
                                y_scale_new = None
                                plotting_settings["y_scale"] = y_scale_new
                                y_label_new = "{0} ({1})".format("Sum of CPU energy consumption", "J")
                                plotting_settings["y_label"] = y_label_new
                        # If the current metric name equals to 'mean_num_testing_examples_used'...
                        elif metric_name == "mean_num_testing_examples_used":
                            # Set the 'y_data' value, if equals to 'Auto'.
                            if y_data == "Auto":
                                y_data_new = "mean_num_testing_examples_used"
                                plotting_settings["y_data"] = y_data_new
                            # Set the 'y_label' value, if equals to 'Auto'.
                            if y_label == "Auto":
                                y_scale_new = None
                                plotting_settings["y_scale"] = y_scale_new
                                y_label_new = "{0}".format("Mean number of testing examples used")
                                plotting_settings["y_label"] = y_label_new
                        # If the current metric name equals to 'mean_sparse_categorical_accuracy'...
                        elif metric_name == "mean_sparse_categorical_accuracy":
                            # Set the 'y_data' value, if equals to 'Auto'.
                            if y_data == "Auto":
                                y_data_new = "mean_sparse_categorical_accuracy"
                                plotting_settings["y_data"] = y_data_new
                            # Set the 'y_label' value, if equals to 'Auto'.
                            if y_label == "Auto":
                                y_scale_new = None
                                plotting_settings["y_scale"] = y_scale_new
                                y_label_new = "{0}".format("Mean accuracy")
                                plotting_settings["y_label"] = y_label_new
                        # If the current metric name equals to 'weighted_mean_sparse_categorical_accuracy'...
                        elif metric_name == "weighted_mean_sparse_categorical_accuracy":
                            # Set the 'y_data' value, if equals to 'Auto'.
                            if y_data == "Auto":
                                y_data_new = "weighted_mean_sparse_categorical_accuracy"
                                plotting_settings["y_data"] = y_data_new
                            # Set the 'y_label' value, if equals to 'Auto'.
                            if y_label == "Auto":
                                y_scale_new = None
                                plotting_settings["y_scale"] = y_scale_new
                                y_label_new = "{0}".format("Weighted mean accuracy")
                                plotting_settings["y_label"] = y_label_new
                        # If the current metric name equals to 'max_communication_time'...
                        elif metric_name == "max_communication_time":
                            # Set the 'y_data' value, if equals to 'Auto'.
                            if y_data == "Auto":
                                y_data_new = "max_communication_time"
                                plotting_settings["y_data"] = y_data_new
                            # Set the 'y_label' value, if equals to 'Auto'.
                            if y_label == "Auto":
                                y_scale_new = None
                                plotting_settings["y_scale"] = y_scale_new
                                y_label_new = "{0} ({1})".format("Max. Communication Time", time_unit_symbol)
                                plotting_settings["y_label"] = y_label_new
                        # Set the 'x_data' value, if equals to 'Auto'.
                        if x_data == "Auto":
                            x_data_new = "comm_round"
                            plotting_settings["x_data"] = x_data_new
                        # Set the 'x_label' value, if equals to 'Auto'.
                        if x_label == "Auto":
                            x_label_new = "Communication round"
                            plotting_settings["x_label"] = x_label_new
                        # Set the 'hue' value, if equals to 'Auto'.
                        if hue == "Auto":
                            hue_new = "client_selector"
                            plotting_settings["hue"] = hue_new
                        # Set the 'hue_order' value, if equals to 'Auto'.
                        if hue_order == "Auto":
                            hue_order_new = client_selectors
                            plotting_settings["hue_order"] = hue_order_new
                            # Set the 'style' value, if equals to 'Auto'.
                        if style == "Auto":
                            style_new = "client_selector"
                            plotting_settings["style"] = style_new
                            # Set the 'size' value, if equals to 'Auto'.
                        if size == "Auto":
                            size_new = "client_selector"
                            plotting_settings["size"] = size_new
                        # Update the plotting settings.
                        individual_evaluate_metrics_history_settings["plotting_settings"] = plotting_settings
                        self._set_attribute("_individual_evaluate_metrics_history_settings",
                                            individual_evaluate_metrics_history_settings)
                        # Set the 'plotting_df' dataframe (data that will be plotted into the figure).
                        plotting_df = DataFrame(data=plotting_data)
                        self._set_attribute("_plotting_df", plotting_df)
                        # Load the figure settings for plots.
                        self._load_figure_settings(plotting_settings)
                        # Plot data into the figure.
                        self._plot_data(plotting_settings, plotting_df)
                        # Set the figure name.
                        figure_name = "fig_{0}_{1}_tasks_{2}_clients_{3}.{4}" \
                                      .format(analysis_name,
                                              n_tasks,
                                              n_available_clients,
                                              metric_name.lower(),
                                              figure_files_extension)
                        # Save the figure.
                        self._save_figure(plotting_settings, figure_name)

    def _analyze_individual_evaluate_metrics_history_file(self) -> None:
        # Get the necessary attributes.
        input_settings = self.get_attribute("_input_settings")
        analysis_settings = self.get_attribute("_analysis_settings")
        individual_evaluate_metrics_history_file = \
            Path(input_settings["individual_evaluate_metrics_history_file"]).absolute()
        analysis_root_folder = Path(analysis_settings["analysis_root_folder"]).absolute()
        individual_evaluate_metrics_history_settings = \
            self.get_attribute("_individual_evaluate_metrics_history_settings")
        results_df_settings = individual_evaluate_metrics_history_settings["results_df_settings"]
        num_tasks = individual_evaluate_metrics_history_settings["num_tasks"]
        num_available_clients = individual_evaluate_metrics_history_settings["num_available_clients"]
        comm_rounds = individual_evaluate_metrics_history_settings["comm_rounds"]
        client_selectors = individual_evaluate_metrics_history_settings["client_selectors"]
        plotting_settings = individual_evaluate_metrics_history_settings["plotting_settings"]
        logger = self.get_attribute("_logger")
        # Log an 'analyzing the results' message.
        message = "Analyzing the results in the '{0}' file...".format(individual_evaluate_metrics_history_file)
        log_message(logger, message, "INFO")
        # Set the analysis name.
        analysis_name = "individual_evaluate_metrics_history"
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
        results_df = read_csv(filepath_or_buffer=individual_evaluate_metrics_history_file,
                              comment=results_df_settings["comments_prefix"])
        # Order the dataframe by client selector's name in ascending order.
        results_df = results_df.sort_values(by=results_df_settings["sort_by"],
                                            ascending=results_df_settings["sort_ascending_order"])
        # Set the 'results_df' dataframe.
        self._set_attribute("_results_df", results_df)

        # TODO: Fix Parser for dict and tuple values for dictionary keys.
        plotting_settings["figure_size"] = (6, 5)
        plotting_settings["line_colors"] = ["#00FFFF", "#FFA500", "#FF4F4F", "#0000FF",
                                            "#7FFFD4", "#228B22", "#635F4C", "#A82378"]
        plotting_settings["line_sizes"] = [2, 2, 2, 2, 2, 2]

        # Set the 'num_tasks' value, if equals to 'Auto'.
        if num_tasks == "Auto":
            num_tasks = list(results_df["num_tasks"].sort_values().unique())
            individual_evaluate_metrics_history_settings["num_tasks"] = num_tasks
        # Set the 'num_available_clients' value, if equals to 'Auto'.
        if num_available_clients == "Auto":
            num_available_clients = list(results_df["num_available_clients"].sort_values().unique())
            individual_evaluate_metrics_history_settings["num_available_clients"] = num_available_clients
        # Set the 'comm_rounds' value, if equals to 'Auto'.
        if comm_rounds == "Auto":
            comm_rounds = sorted([int(comm_round.replace("comm_round_", ""))
                                  for comm_round in list(results_df["comm_round"].sort_values().unique())])
            individual_evaluate_metrics_history_settings["comm_rounds"] = comm_rounds
        # Set the 'client_selectors' value, if equals to 'Auto'.
        if client_selectors == "Auto":
            client_selectors = list(results_df["client_selector"].sort_values().unique())
            individual_evaluate_metrics_history_settings["client_selectors"] = client_selectors
        # Set the 'n_colors' value, if equals to 'Auto'.
        if plotting_settings["n_colors"] == "Auto":
            plotting_settings["n_colors"] = len(client_selectors)
        # Set the 'x_ticks' value, if equals to 'Auto'.
        if plotting_settings["x_ticks"] == "Auto":
            plotting_settings["x_ticks"] = list(range(0, max(comm_rounds)+1, 10))
        # Update the plotting settings.
        individual_evaluate_metrics_history_settings["plotting_settings"] = plotting_settings
        self._set_attribute("_individual_evaluate_metrics_history_settings",
                            individual_evaluate_metrics_history_settings)
        # Load the theme and palette for plots.
        self._load_theme_and_palette(plotting_settings)
        # Generate the figures.
        self._generate_figures_for_individual_evaluate_metrics_history_file()

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
