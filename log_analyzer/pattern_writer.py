import os


def write_patterns_to_file(filename, patterns):
    """
    Write the observed log patterns to a file.

    Args:
        filename (str): The name of the output file.
        patterns (list): A list of dictionaries representing log patterns and their counts.

    Returns:
        None
    """
    if any(char.isdigit() for char in filename):
        filename = ''.join([i for i in filename if not i.isdigit()])
    with open(filename, 'w', encoding='utf-8', errors='ignore') as f:
        for pattern in patterns:
            # Remove the timestamp from the message before writing to file
            message = pattern['message'][24:]
            f.write(f"{pattern['count']}\t{message}\n")
