import os
import sys

# Add the path of the log_analyzer package to sys.path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import magic
from log_analyzer.pattern_matcher import create_pipeline, train_pipeline, predict_anomalies
from log_analyzer.log_scanner import process_log_file, create_output_dir, scan_logs_for_patterns
from log_analyzer.pattern_writer import write_patterns_to_file
from log_analyzer.anomaly_writer import write_anomalies_to_file


if __name__ == '__main__':
    scan_logs_for_patterns()
