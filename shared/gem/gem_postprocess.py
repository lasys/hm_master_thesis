# Copied function from https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/result/postprocess.py 
def _hex_to_bin(hexstring):
    """Convert hexadecimal readouts (memory) to binary readouts."""
    return str(bin(int(hexstring, 16)))[2:]

def _pad_zeros(bitstring, memory_slots):
    """If the bitstring is truncated, pad extra zeros to make its
    length equal to memory_slots"""
    return format(int(bitstring, 2), f"0{memory_slots}b")

def _separate_bitstring(bitstring, creg_sizes):
    """Separate a bitstring according to the registers defined in the result header."""
    substrings = []
    running_index = 0
    for _, size in reversed(creg_sizes):
        substrings.append(bitstring[running_index : running_index + size])
        running_index += size
    return " ".join(substrings)

def _format_counts_memory(shot_memory, header=None):
    """
    Format a single bitstring (memory) from a single shot experiment.
    - The hexadecimals are expanded to bitstrings
    - Spaces are inserted at register divisions.
    Args:
        shot_memory (str): result of a single experiment.
        header (dict): the experiment header dictionary containing
            useful information for postprocessing. creg_sizes
            are a nested list where the inner element is a list
            of creg name, creg size pairs. memory_slots is an integers
            specifying the number of total memory_slots in the experiment.
    Returns:
        dict: a formatted memory
    """
    if shot_memory.startswith("0x"):
        shot_memory = _hex_to_bin(shot_memory)
    if header:
        creg_sizes = header.creg_sizes
        memory_slots = header.memory_slots
        if memory_slots:
            shot_memory = _pad_zeros(shot_memory, memory_slots)
        if creg_sizes and memory_slots:
            shot_memory = _separate_bitstring(shot_memory, creg_sizes)
    return shot_memory

def format_counts(counts, header=None):
    """Format a single experiment result coming from backend to present
    to the Qiskit user.
    Args:
        counts (dict): counts histogram of multiple shots
        header (dict): the experiment header dictionary containing
            useful information for postprocessing.
    Returns:
        dict: a formatted counts
    """
    counts_dict = {}
    for key, val in counts.items():
        key = _format_counts_memory(key, header)
        counts_dict[key] = val
    return counts_dict