import json

def write_json(path: str, data: dict) -> None:
    """
    It writes a dictionary to a file
    
    :param path: The path to the file to write to
    :type path: str
    :param data: the data to be written to the file
    :type data: dict
    """
    ctx = json.dumps(data)
    with open(path, 'w') as f:
        f.write(ctx)

    
def read_json(path: str) -> dict:
    """
    `read_json` reads a JSON file and returns a dictionary
    
    :param path: The path to the JSON file
    :type path: str
    :return: A dictionary
    """
    return json.loads(open(path))
