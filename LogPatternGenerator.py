import os
import gzip
import sys
import magic
import tarfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import IsolationForest
import pickle

def check_dependencies():
    # Check if all required packages are installed
    try:
        import magic
        import sklearn
    except ImportError:
        print('Error: Required packages are missing. Please run "pip install -r requirements.txt" to install the required packages.')
        sys.exit(1)

def scan_log_file(log_file):
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        messages = [line for line in f]

    # Calculate the number of unique words in the log file
    unique_words = set(' '.join(messages).split())

    # Calculate the total number of words in the log file
    total_words = sum(len(line.split()) for line in messages)

    # Calculate the average length of each message in the log file
    avg_length = sum(len(line) for line in messages) / len(messages)

    return unique_words, total_words, avg_length


def create_pipeline(log_file=None):
    # Create a TfidfVectorizer and a TruncatedSVD transformer to reduce the dimensionality of the data
    vectorizer = TfidfVectorizer(max_features=8, min_df=1, stop_words='english')
    
    # Set the max_features and n_components based on the number of log messages in the file
    if log_file is not None:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            messages = [line for line in f]
        n_messages = len(messages)
        if n_messages < vectorizer.max_features:
            max_features = n_messages
        else:
            max_features = vectorizer.max_features
        n_components = min(max_features, 10)
        vectorizer = TfidfVectorizer(max_features=max_features, min_df=1, stop_words='english')
    else:
        n_components = min(vectorizer.max_features, 10)

    svd = TruncatedSVD(n_components=n_components)

    # Create a pipeline that applies the vectorizer, SVD, and IsolationForest model
    pipeline = make_pipeline(vectorizer, svd, IsolationForest(contamination=0.01))

    return pipeline

def train_pipeline(pipeline, messages):
    # Check for an empty list of messages
    if not messages:
        print('Error: Empty list of messages')
        return None

    # Fit the pipeline on the log messages
    try:
        pipeline.fit(messages)
    except ValueError as e:
        if "empty vocabulary" in str(e):
            print('Error: No terms remain after pruning. Try a lower min_df or a higher max_df')
        elif "contain stop words" in str(e):
            print('Error: Only stop words are found in the document. Try lowering the max_df')
        else:
            print('Error:', e)
        return None

    return pipeline

def predict_anomalies(pipeline, messages):
    anomaly_scores = pipeline.decision_function(messages)
    is_anomaly = anomaly_scores < 0

    return is_anomaly


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

def write_patterns_to_file(filename, patterns):
    if any(char.isdigit() for char in filename):
        filename = ''.join([i for i in filename if not i.isdigit()])
    with open(filename, 'w', encoding='utf-8', errors='ignore') as f:
        for pattern in patterns:
            # Remove the timestamp from the message before writing to file
            message = pattern['message'][24:]
            f.write(f"{pattern['count']}\t{message}\n")

def create_output_dir(log_dir):
    output_dir = os.path.join(log_dir, 'Log_Patterns')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print(f'Created {output_dir}')
    return output_dir

def write_patterns_and_anomalies(output_dir, base_filename, observed_messages, anomalies, anomaly_indices, processed_files):
    if base_filename[-1].isdigit():
        base_filename = ''.join(filter(lambda x: not x.isdigit(), base_filename))
    
    # Check if this log file has been processed before
    if base_filename in processed_files:
        # Append the observed messages to the existing pattern file
        pattern_filename = processed_files[base_filename]['pattern_file']
        with open(pattern_filename, 'a', encoding='utf-8', errors='ignore') as output_file:
            for message, count in observed_messages.items():
                output_file.write(f'{count}\t{message}\n')

        # Append the anomalous messages to the existing anomaly file
        anomaly_filename = processed_files[base_filename]['anomaly_file']
        with open(anomaly_filename, 'r', encoding='utf-8', errors='ignore') as anomaly_input_file:
            existing_anomalies = set(anomaly_input_file.read().splitlines())

        with open(anomaly_filename, 'a', encoding='utf-8', errors='ignore') as anomaly_output_file:
            for i in anomaly_indices:
                if i < len(anomalies):
                    new_anomaly = anomalies[i].strip()
                    if new_anomaly not in existing_anomalies:
                        anomaly_output_file.write(f"{new_anomaly}\n")
                else:
                    print(f"Warning: anomaly index {i} is out of range for file {base_filename}")
    else:
        # Create new pattern and anomaly files for this log file
        pattern_filename = os.path.join(output_dir, base_filename + '_patterns.txt')
        anomaly_filename = os.path.join(output_dir, base_filename + '_anomalies.txt')

        with open(pattern_filename, 'w', encoding='utf-8', errors='ignore') as output_file:
            for message, count in observed_messages.items():
                output_file.write(f'{count}\t{message}\n')

        with open(anomaly_filename, 'w', encoding='utf-8', errors='ignore') as anomaly_output_file:
            for i in anomaly_indices:
                if i < len(anomalies):
                    anomaly_output_file.write(f"{anomalies[i].strip()}\n")
                else:
                    print(f"Warning: anomaly index {i} is out of range for file {base_filename}")

        # Add this log file and its pattern and anomaly files to the processed files dictionary
        processed_files[base_filename] = {'pattern_file': pattern_filename, 'anomaly_file': anomaly_filename}

    return processed_files

def scan_logs_for_patterns():
    # Prompt the user to input the directory where the log files are located
    log_dir = input('Enter the path to the log file directory: ')

    # Create the Log_Patterns directory if it doesn't exist
    output_dir = create_output_dir(log_dir)

    # Create the pipeline
    pipeline = create_pipeline()

    # Store patterns and anomalies in a dictionary with base filename as key
    pattern_dict = {}

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

        if base_filename in pattern_dict:
            # Append the observed messages and anomalies to existing pattern files
            observed_messages = pattern_dict[base_filename]['observed_messages']
            observed_patterns = [{'count': observed_messages[message], 'message': message} for message in sorted(observed_messages)]
            anomalies = pattern_dict[base_filename]['anomalies'] + anomalies
        else:
            # Create a new entry in the pattern dictionary
            observed_patterns = [{'count': count, 'message': message} for message, count in observed_messages.items()]
            anomalies = anomalies

        # Add observed messages and anomalies to pattern dictionary
        pattern_dict[base_filename] = {
            'observed_messages': observed_messages,
            'observed_patterns': observed_patterns,
            'anomalies': anomalies,
        }

    # Write patterns and anomalies to files for each base filename
    for base_filename, pattern_data in pattern_dict.items():
        observed_patterns = pattern_data['observed_patterns']
        anomalies = pattern_data['anomalies']

        # Write patterns to file
        output_filename = os.path.join(output_dir, base_filename + '_patterns.txt')
        write_patterns_to_file(output_filename, observed_patterns)

        if anomalies:
            # Write the anomalous messages to the output file
            anomaly_output_filename = os.path.join(output_dir, base_filename + '_anomalies.txt')
            with open(anomaly_output_filename, 'w', encoding='utf-8', errors='ignore') as anomaly_output_file:
                for anomaly in anomalies:
                    anomaly_output_file.write(anomaly)

    print('Done scanning logs for patterns.')

scan_logs_for_patterns()

