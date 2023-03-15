import numpy as np

from ratinabox.Environment import Environment
from ratinabox.Agent import Agent


def random_angle_steps(steps: int, irregularity: float):
    """Generates the division of a circumference in random angles.

    Parameters
    ----------
    steps (int):
        the number of angles to generate.
    irregularity (float):
        variance of the spacing of the angles between consecutive vertices.

    Returns
    -------
    The list of random angles.
    """
    # Generate n angle steps
    angles = []
    lower = (2 * np.pi / steps) - irregularity
    upper = (2 * np.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = np.random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # Normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * np.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles


def generate_random_polygon(avg_radius: float, irregularity: float, spikiness: float, max_num_vertices: int):
    """Creates the polygon by sampling points on a circle around the center.
    Angular spacing between sequential points, and radial distance of each
    point from the centre are varied randomly.

    Parameters
    ----------
    avg_radius: float
        The average radius (distance of each generated vertex to the center of the circumference)
        used to generate points with a normal distribution.
    irregularity: float
        Variance of the spacing of the angles between consecutive vertices.
    spikiness: float
        Variance of the distance of each vertex to the center of the circumference.
    max_num_vertices: int
        Maximum number of vertices the polygon can have. The final number will be randomly drawn between 3 and this number.

    Returns
    -------
    List of vertices, in CCW order.
    """
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    # Generate random number of vertices
    num_vertices = np.random.randint(3, max_num_vertices + 1)

    irregularity *= 2 * np.pi / num_vertices
    spikiness *= avg_radius

    # Generate angular steps
    angle_steps = random_angle_steps(num_vertices, irregularity)

    # Generate points
    points = []
    angle = np.random.uniform(0, 2 * np.pi)
    for i in range(num_vertices):
        radius = np.clip(np.random.normal(avg_radius, spikiness), 0, 2 * avg_radius)
        point = [radius * np.cos(angle), radius *  np.sin(angle)]
        points.append(point)
        angle += angle_steps[i]

    return points


def generate_random_polygon_envs(num_envs, **params):
    """Generates random polygon environments.

    Parameters
    ----------
    num_envs: int
        Number of environments to generate.

    Notes
    -----
    For other parameters see 'generate_random_polygon' method. All additional parameters will be passed to it.

    Returns
    -------
    List of environments.
    """
    envs = []
    for i in range(num_envs):

        # Create random polygon
        random_polygon = generate_random_polygon(**params)

        # Create environment
        env = Environment({"boundary": random_polygon})

        # Append
        envs.append(env)

    return envs


def generate_random_trajectories(envs, traj_per_env, traj_len, **params):
    """Generates random trajectories.

    Parameters
    ----------
    envs: list
        List of environments.
    traj_per_env: int
        Number of trajectories per environment.
    traj_len: int
        Length of each trajectory.

    Returns
    -------
    List of trajectories.
    """
    trajectories = []
    for env in envs:

        for i in range(traj_per_env):

            # Create agent
            agent = Agent(env, params)

            # Generate trajectory
            for i in range(traj_len):
                agent.update()

            # Append
            trajectories.append(agent.history['pos'])

    return trajectories
