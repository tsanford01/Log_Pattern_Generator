import os


def write_anomalies_to_file(output_dir, base_filename, anomalies):
    if base_filename[-1].isdigit():
        base_filename = ''.join(filter(lambda x: not x.isdigit(), base_filename))

    # Write the anomalous messages to the output file
    anomaly_output_filename = os.path.join(output_dir, base_filename + '_anomalies.txt')
    with open(anomaly_output_filename, 'w', encoding='utf-8', errors='ignore') as anomaly_output_file:
        for anomaly in anomalies:
            anomaly_output_file.write(anomaly)

    return anomaly_output_filename
