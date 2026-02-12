# src/step4/utils_geometry.py

def get_face_slices(face):
    """
    Return slicing tuples for selecting the ghost layer on a given face.

    Example:
        face = "x_min" → (0, :, :)
        face = "y_max" → (:, -1, :)
        face = "z_min" → (:, :, 0)

    This is used by pressure and velocity BC modules.
    """

    if face == "x_min":
        return (0, slice(None), slice(None))
    if face == "x_max":
        return (-1, slice(None), slice(None))

    if face == "y_min":
        return (slice(None), 0, slice(None))
    if face == "y_max":
        return (slice(None), -1, slice(None))

    if face == "z_min":
        return (slice(None), slice(None), 0)
    if face == "z_max":
        return (slice(None), slice(None), -1)

    raise ValueError(f"Unknown face: {face}")


def get_normal_direction(face):
    """
    Return the axis normal to the given face: "x", "y", or "z".

    Example:
        face = "x_min" → "x"
        face = "y_max" → "y"
        face = "z_min" → "z"
    """

    if face.startswith("x_"):
        return "x"
    if face.startswith("y_"):
        return "y"
    if face.startswith("z_"):
        return "z"

    raise ValueError(f"Unknown face: {face}")


def get_tangential_directions(face):
    """
    Return the two tangential axes for a given face.

    Example:
        face = "x_min" → ("y", "z")
        face = "y_max" → ("x", "z")
        face = "z_min" → ("x", "y")
    """

    normal = get_normal_direction(face)

    if normal == "x":
        return ("y", "z")
    if normal == "y":
        return ("x", "z")
    if normal == "z":
        return ("x", "y")

    raise ValueError(f"Unknown face: {face}")


def get_face_coordinates(nx, ny, nz, face):
    """
    Return the (i, j, k) index of the boundary face in the NON-extended grid.

    This is used only in boundary-fluid logic and tests that check
    correct face identification.

    Example:
        face = "x_min" → i = 0
        face = "x_max" → i = nx - 1
    """

    if face == "x_min":
        return 0, None, None
    if face == "x_max":
        return nx - 1, None, None

    if face == "y_min":
        return None, 0, None
    if face == "y_max":
        return None, ny - 1, None

    if face == "z_min":
        return None, None, 0
    if face == "z_max":
        return None, None, nz - 1

    raise ValueError(f"Unknown face: {face}")
