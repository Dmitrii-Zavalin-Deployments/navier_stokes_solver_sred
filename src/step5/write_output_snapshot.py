# src/step5/write_output_snapshot.py

def write_output_snapshot(state, current_time, step_index):
    """
    Optional output writer.

    This is intentionally minimal and does not write files.
    It only records that a snapshot *would* have been written.
    """

    if state.history is None:
        state.history = {"output_file_pairs": []}
    else:
        state.history.setdefault("output_file_pairs", [])

    # Placeholder entry — real implementation may write JSON/VTK
    state.history["output_file_pairs"].append({
        "time": current_time,
        "step": step_index,
        "json": None,
        "vti": None,
    })
