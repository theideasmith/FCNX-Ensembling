import os
def unix_basename(path):
    """
    Returns the base name of a path, mimicking the behavior of the Unix basename command.
    """
    if path.endswith(os.sep):
        path = path.rstrip(os.sep)
    return os.path.basename(path)
