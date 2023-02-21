import os
import magic
import tarfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import IsolationForest

# from Log_Pattern_Generator.log_analyzer.anomaly_writer import write_anomalies_to_file
# from Log_Pattern_Generator.log_analyzer.pattern_matcher import create_pipeline, predict_anomalies, train_pipeline
# from Log_Pattern_Generator.log_analyzer.pattern_writer import write_patterns_to_file
from anomaly_writer import write_anomalies_to_file
from pattern_matcher import create_pipeline, predict_anomalies, train_pipeline
from pattern_writer import write_patterns_to_file


def process_log_file(log_file, pipeline):
    if log_file.endswith('.tgz'):
        # Extract the log file from the .tgz archive
        with tarfile.open(log_file, 'r:gz') as tar:
            log_filename = os.path.basename(log_file).replace('.tgz', '')
            tar.extract(log_filename, path=os.path.dirname(log_file))
            log_file = os.path.join(os.path.dirname(log_file), log_filename)

    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        messages = [line for line in f]

    # Check for an empty list of messages
    if not messages:
        print(f'Warning: Empty file {log_file}')
        return [], [], [], {}

    # Fit the pipeline on the log messages
    pipeline = train_pipeline(pipeline, messages)

    if not pipeline:
        return messages, [], [], {}

    # Predict the anomaly score for each message and return anomalous messages and their indices
    is_anomaly = predict_anomalies(pipeline, messages)
    anomalies = [message for i, message in enumerate(messages) if is_anomaly[i]]
    anomaly_indices = [i for i, is_anomaly in enumerate(is_anomaly) if is_anomaly]

    # Count the frequency of observed log messages that are not anomalies
    observed_messages = {}
    for i, message in enumerate(messages):
        if not is_anomaly[i]:
            if message not in observed_messages:
                observed_messages[message] = 1
            else:
                observed_messages[message] += 1

    return messages, anomalies, anomaly_indices, observed_messages


def create_output_dir(log_dir):
    output_dir = os.path.join(log_dir, 'Log_Patterns')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print(f'Created {output_dir}')
    return output_dir


def scan_logs_for_patterns():
    # Prompt the user to input the directory where the log files are located
    log_dir = input('Enter the path to the log file directory: ')

    # Create the Log_Patterns directory if it doesn't exist
    output_dir = create_output_dir(log_dir)

    # Create the pipeline
    pipeline = create_pipeline()

    # Loop over all files in the directory
    for filename in os.listdir(log_dir):
        filepath = os.path.join(log_dir, filename)
        if not os.path.isfile(filepath):
            continue

        # Determine the file type using the magic library
        file_type = magic.from_file(filepath, mime=True)

        # Handle the file based on its type
        if not file_type.startswith('text/'):
            continue

        print(f'Analyzing {filename}...')

        # Extract the base filename (i.e. remove the extension and digits after the dot)
        base_filename = os.path.splitext(filename)[0].rsplit('.', 1)[0]

        # Process the log file and get the anomalous messages, their indices, and the observed messages
        messages, anomalies, anomaly_indices, observed_messages = process_log_file(filepath, pipeline)

        if not messages:
            # Empty file
            continue

        # Write patterns to file
        pattern_filename = os.path.join(output_dir, base_filename + '_patterns.txt')
        write_patterns_to_file(pattern_filename, observed_messages)

        if anomalies:
            # Write the anomalous messages to the output file
            anomaly_output_filename = os.path.join(output_dir, base_filename + '_anomalies.txt')
            write_anomalies_to_file(anomaly_output_filename, anomalies)

    print('Done scanning logs for patterns.')
