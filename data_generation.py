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


def generate_random_polygon(avg_radius=1, irregularity=0.5, spikiness=0.5, max_num_vertices=10):
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


def generate_random_polygon_env(**params):
    """Generates random polygon environments.

    Notes
    -----
    For other parameters see 'generate_random_polygon' method. All additional parameters will be passed to it.

    Returns
    -------
    List of environments.
    """
    # Generate random polygon
    random_polygon = generate_random_polygon(**params)

    # Create environment
    env = Environment({"boundary": random_polygon})

    return env


def generate_random_trajectories(env, n_traj, traj_len, **params):
    """Generates random trajectories.

    Parameters
    ----------
    env: Environment
        The environment in which the trajectories will be generated.
    n_traj: int
        Number of trajectories to be generated.
    traj_len: int
        Length of each trajectory.

    Returns
    -------
    List of trajectories.
    """
    trajectories = []
    for i in range(n_traj):

        # Create agent
        agent = Agent(env, params)

        # Generate trajectory
        for i in range(traj_len):
            agent.update()

        # Append
        trajectories.append(agent.history['pos'])

    return trajectories


def generate_random_samples(env, n_samples):
    """Generates random samples in the environment.
    """
    return env.sample_positions(n=n_samples, method='random')


def add_random_walls(env, n_walls, min_gap=0.1, min_len=0.2):
    """Adds n_walls random walls to the environment.

    Parameters
    ----------
    env : Environment
        Environment to add walls to.
    n_walls : int
        Number of walls to add.
    min_gap : float, optional
        Minimum gap size in the opposite direction, by default 0.1
    min_len : float, optional
        Minimum wall length, by default 0.2

    Returns
    -------
    Environment
        Environment with added walls.
    """

    added_walls = 0
    while added_walls < n_walls:

        # Sample a random point in the environment
        p = env.sample_positions(n=1, method="random")[0]

        p2 = p
        l = 0
        direction = 0
        found = False
        while not found:

            # Increase length
            l += 0.01

            for d in np.arange(0, 2 * np.pi, 0.01):

                # Determine new point in direction d
                p2 = [p[0] + l * np.cos(d), p[1] + l * np.sin(d)]

                # Calculate wall collisions
                _, wall_collision = env.check_wall_collisions(np.array([p, p2]))

                if np.any(wall_collision) or not env.check_if_position_is_in_environment(p2):
                    found = True
                    direction = d
                    break

        # Determine point with minimum gap size in opposite direction
        p3 = [p[0] + min_gap * np.cos(direction + np.pi), p[1] + min_gap * np.sin(direction + np.pi)]

        # Calculate wall collisions
        _, wall_collision = env.check_wall_collisions(np.array([p, p3]))

        # Check if the gap is large enough
        if env.check_if_position_is_in_environment(p3) and not np.any(wall_collision):

            # Add wall if p2 is not equal to p and l is large enough
            if not np.all(p == p2) and l > min_len:
                env.add_wall([p, p2])
                added_walls += 1

    return env