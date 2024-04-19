from pathlib import Path
from warnings import filterwarnings

from goffls.util.config_parser_util import parse_config_section
from goffls.util.logger_util import load_logger


class ResultAnalyzer:
    def __init__(self,
                 config_file: Path) -> None:
        self._config_file = config_file
        self._logging_settings = None
        self._input_settings = None
        self._analysis_settings = None
        self._logger = None
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

    def _set_logger(self) -> None:
        logging_settings = self._logging_settings
        logger_name = type(self).__name__ + "_Logger"
        logger = load_logger(logging_settings, logger_name)
        self._logger = logger

    def _analyze_selected_fit_clients_history_file(self) -> None:
        pass

    def _analyze_selected_evaluate_clients_history_file(self) -> None:
        pass

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
