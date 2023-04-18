import numpy as np
import torch
from multiprocessing import Pool
from shapely import Point, Polygon

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


def generate_random_polygon(avg_radius=1, irregularity=0.5, spikiness=0.5, max_num_vertices=8):
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


def generate_random_polygon_env(n_env=1, add_walls=True, add_holes=True, **params):
    """Generates random polygon environments.

    Parameters
    ----------
    n_env: int
        Number of environments to be generated.
    add_walls: bool
        Whether to add walls to the environment.
    add_holes: bool
        Whether to add holes to the environment.

    Notes
    -----
    For other parameters see 'generate_random_polygon' method. All additional parameters will be passed to it.

    Returns
    -------
    List of environments.
    """
    envs = []
    for i in range(n_env):

        # Generate random polygon
        random_polygon = generate_random_polygon(**params)

        holes = []
        if add_holes:
            holes, _ = generate_random_holes(random_polygon, n_holes=np.random.randint(0, 2))

        # Create environment
        env = Environment({'boundary': random_polygon, 'holes': holes})

        # Add walls
        if add_walls:
            env = add_random_walls(env, np.random.randint(0, 2))

        envs.append(env)

    if n_env == 1:
        return envs[0]

    return envs


def add_random_walls_new(env, n_walls, min_gap=0.1, min_len=0.2):
    """Adds n_walls random walls to the environment.
    """
    added_walls = 0
    while added_walls < n_walls:

        # Sample a random point in the environment
        p = env.sample_positions(n=1, method="random")[0]

        # Get list of candidates
        candidates = env.vectors_from_walls(p)

        # Get norms of candidates
        norms = np.linalg.norm(candidates, axis=1)

        if np.min(norms) > min_gap:

            # Get subset of candidates that are long enough
            candidates = candidates[norms > min_len]

            env.add_wall([p, p - candidates[0]])

            added_walls += 1

            # if candidates.shape[0] > 0:
            #
            #     # Choose random candidate
            #     idx = np.random.randint(0, candidates.shape[0])
            #
            #     # Add wall
            #     env.add_wall([p, p + candidates[idx]])
            #     added_walls += 1

    return env


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
        l = min_len
        direction = 0
        found = False
        while not found:

            # Increase length
            l += 0.05

            for d in np.arange(0, 2 * np.pi, 0.1):

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


def generate_random_holes(polygon, n_holes, min_hole_points=3, max_hole_points=4, min_dist=0.3):
    """Adds random holes to a Shapely polygon object.

    Parameters
    ----------
    polygon: shapely.geometry.Polygon or list of points
        The polygon to which holes will be added.
    n_holes: int
        The number of holes to add.
    min_hole_points: int
        The minimum number of points in each hole.
    max_hole_points: int
        The maximum number of points in each hole.
    min_dist: float
        Minimum distance from the polygon boundary.

    Todo
        * make sure that holes do not overlap.

    Returns
    -------
    shapely.geometry.Polygon
        The polygon object with the holes added.
    """
    if isinstance(polygon, list):
        polygon = Polygon(polygon)

    if not isinstance(polygon, Polygon):
        raise ValueError("Polygon must be a shapely.geometry.Polygon object or a list of points.")

    holes = []
    for _ in range(n_holes):

        # Generate a random number of points for the hole
        if max_hole_points == min_hole_points:
            n_points = max_hole_points
        else:
            n_points = np.random.randint(min_hole_points, max_hole_points)

        # Generate random points within the polygon
        hole_points = []
        while len(hole_points) < n_points:
            point = Point(
                np.random.uniform(
                    polygon.bounds[0] - np.sign(polygon.bounds[0]) * min_dist,
                    polygon.bounds[2] - np.sign(polygon.bounds[2]) * min_dist),
                np.random.uniform(
                    polygon.bounds[1] - np.sign(polygon.bounds[1]) * min_dist,
                    polygon.bounds[3] - np.sign(polygon.bounds[3]) * min_dist
                )
            )
            if polygon.contains(point):
                hole_points.append(point)

        # Create a hole polygon from the generated points
        points = [point.coords[0] for point in hole_points]
        hole_polygon = Polygon(points)

        # Subtract the hole polygon from the original polygon
        polygon = polygon.difference(hole_polygon)

        holes.append(points)

    return holes, polygon


def generate_trajectory(envs, timesteps, params):
    """Generates a single trajectory."""

    # Select environment
    if isinstance(envs, list):
        env = np.random.choice(envs)
    else:
        env = envs

    # Create agent
    agent = Agent(env, params)

    # Generate trajectory
    for _ in range(timesteps):
        agent.update()

    # Return trajectory
    return agent.history['pos']


def generate_random_trajectories(envs, n_traj, timesteps, **params):
    """Generates random trajectories.

    Parameters
    ----------
    envs: Environment or list of Environments
        The environment in which the trajectories will be generated.
    n_traj: int
        Number of trajectories to be generated.
    timesteps: int
        Length of each trajectory.

    Returns
    -------
    Tensor of shape.
    """
    # Create a Pool of worker processes
    pool = Pool()

    # Use the Pool to parallelize trajectory generation
    results = pool.starmap(generate_trajectory, [(envs, timesteps, params) for i in range(n_traj)])

    # Close the Pool to prevent further submissions
    pool.close()

    # Wait for all worker processes to finish
    pool.join()

    # Convert list of trajectories to a numpy array
    p = np.array(results).astype('float32')

    # Compute velocity
    v = np.diff(p, axis=1)

    # Convert trajectory arrays to torch Tensors
    p = torch.Tensor(p)
    v = torch.Tensor(v)

    return p, v


def generate_random_samples(env, n_samples):
    """Generates random samples in the environment.
    """
    return torch.Tensor(env.sample_positions(n=n_samples, method='random').astype('float32'))
