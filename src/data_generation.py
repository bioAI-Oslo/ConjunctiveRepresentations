import numpy as np
import torch
from multiprocessing import Pool
from shapely import Point, Polygon, MultiPoint, LineString
from scipy.stats import rayleigh, vonmises
import matplotlib.pyplot as plt

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


def generate_random_polygon(avg_radius=1, irregularity=0.5, spikiness=0.2, max_num_vertices=8, **params):
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
        point = [radius * np.cos(angle), radius * np.sin(angle)]
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
        points = generate_random_polygon(**params)

        holes = []
        if add_holes:
            holes, _ = generate_random_holes(points, n_holes=np.random.randint(0, 2))

        # Create environment
        env = Environment({'boundary': points, 'holes': holes})

        # Add walls
        if add_walls:
            env = add_random_walls(env, np.random.randint(1, 3))

        envs.append(env)

    if n_env == 1:
        return envs[0]

    return envs


def add_random_walls(env, n_walls, min_gap=0.1, min_len=0.2):
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

            if candidates.shape[0] > 0:

                # Choose random candidate
                idx = np.random.randint(0, candidates.shape[0])

                # Get second point
                p2 = p - candidates[idx]

                # Check for collisions
                _, wall_collision = env.check_wall_collisions(np.array([p, p2]))

                # Add wall
                if not np.any(wall_collision):
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
    if isinstance(polygon, (list, np.ndarray)):
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

    while True:

        # Create agent
        agent = Agent(env, params)

        # Generate trajectory
        for t in range(timesteps):

            agent.update()

            # Check if the agent had a collision with the environment
            if len(agent.history["pos"]) > 1:

                w, wc = agent.Environment.check_wall_collisions(np.array([agent.history["pos"][-2], agent.history["pos"][-1]]))

                # If there was a collision, break the for-loop
                if np.any(wc):
                    break

        # Return agent
        if t == timesteps - 1:
            return agent


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
    p = np.array([agent.history['pos'] for agent in results]).astype('float32')

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


def shortest_vectors_from_points_to_lines(positions, vectors):
    """
    Takes a list of positions and a list of vectors (line segments) and returns the pairwise  vectors of shortest distance
    FROM the vector segments TO the positions.
    Suppose we have a list of N_p positions and a list of N_v line segments (or vectors). Each position is a point like [x_p,y_p], or p_p as a vector.
    Each vector is defined by two points [[x_v_0,y_v_0],[x_v_1,y_v_1]], or [p_v_0,p_v_1]. Thus
        positions.shape = (N_p,2)
        vectors.shape = (N_v,2,2)

    Each vector defines an infinite line, parameterised by line_v = p_v_0 + l_v . (p_v_1 - p_v_0).
    We want to solve for the l_v defining the point on the line with the shortest distance to p_p. This is given by:
        l_v = dot((p_p-p_v_0),(p_v_1-p_v_0)/dot((p_v_1-p_v_0),(p_v_1-p_v_0)).
    Or, using a diferrent notation
        l_v = dot(d,s)/dot(s,s)
    where
        d = p_p-p_v_0
        s = p_v_1-p_v_0"""

    vectors = np.array([np.asarray(line.coords) for line in vectors])

    assert (positions.shape[-1] == 2) and (
        vectors.shape[-2:] == (2, 2)
    ), "positions and vectors must have shapes (_,2) and (_,2,2) respectively. _ is optional"
    positions = positions.reshape(-1, 2)
    vectors = vectors.reshape(-1, 2, 2)
    positions = positions + np.random.normal(scale=1e-6, size=positions.shape)
    vectors = vectors + np.random.normal(scale=1e-6, size=vectors.shape)

    N_p = positions.shape[0]
    N_v = vectors.shape[0]

    d = np.expand_dims(positions, axis=1) - np.expand_dims(
        vectors[:, 0, :], axis=0
    )  # d.shape = (N_p,N_v,2)
    s = vectors[:, 1, :] - vectors[:, 0, :]  # vectors.shape = (N_v,2)

    """in order to do the dot product we must reshaope s to be d's shape."""
    s_ = np.tile(
        np.expand_dims(s.copy(), axis=0), reps=(N_p, 1, 1)
    )  # s_.shape = (N_p,N_v,2)
    """now do the dot product by broadcast multiplying the arraays then summing over the last axis"""

    l_v = (d * s).sum(axis=-1) / (s * s).sum(axis=-1)  # l_v.shape = (N_p,N_v)

    """
    Now we can actually find the vector of shortest distance from the line segments to the points which is given by the size of the perpendicular
        perp = p_p - (p_v_0 + l_v.s_)

    But notice that if l_v > 1 then the perpendicular drops onto a part of the line which doesn't exist.
    In fact the shortest distance is to the point on the line segment where l_v = 1. Likewise for l_v < 0.
    To fix this we should limit l_v to be between 1 and 0
    """
    l_v[l_v > 1] = 1
    l_v[l_v < 0] = 0

    """we must reshape p_p and p_v_0 to be shape (N_p,N_v,2), also reshape l_v to be shape (N_p, N_v,1) so we can broadcast multiply it wist s_"""
    p_p = np.tile(
        np.expand_dims(positions, axis=1), reps=(1, N_v, 1)
    )  # p_p.shape = (N_p,N_v,2)
    p_v_0 = np.tile(
        np.expand_dims(vectors[:, 0, :], axis=0), reps=(N_p, 1, 1)
    )  # p_v_0.shape = (N_p,N_v,2)
    l_v = np.expand_dims(l_v, axis=-1)

    perp = p_p - (p_v_0 + l_v * s_)  # perp.shape = (N_p,N_v,2)

    return perp


class PolygonEnvironment:

    def __init__(self, points=None, add_holes=True, add_walls=True, **params):

        # Generate points for random polygon
        if points is None:
            points = generate_random_polygon(**params)

        # Constructs walls from points on polygon
        self.walls = [
            LineString([points[(i + 1) if (i + 1) < len(points) else 0], points[i]])
            for i in range(len(points))
        ]

        # Create polygon
        self.polygon = Polygon(points)

        # Create polygon
        if add_holes:
            # Generate random holes
            self.add_random_holes(n_holes=np.random.randint(0, 1), **params)

        if add_walls:
            self.add_random_walls(n_walls=np.random.randint(0, 3))

    def crosses_wall(self, start, stop):
        step_line = LineString([start, stop])
        for wall in self.walls:
            if step_line.crosses(wall):
                return True
        return False

    def crosses_which_wall(self, start, stop):
        """Checks if the line between a start, and an end point crosses any wall of this environment.
        """
        # Create a line from the start and stop points
        step_line = LineString([start, stop])

        for wall in self.walls:
            if step_line.crosses(wall):

                # Find the intersection point
                intersection_point = step_line.intersection(wall)

                if isinstance(intersection_point, Point):

                    # Compute the direction of wall crossing
                    wall_direction = np.array(wall.coords[1]) - np.array(wall.coords[0])
                    crossing_direction = np.array(stop) - np.array(start)

                    # Check if the wall was crossed in x, y, or both directions
                    x_crossed = (np.sign(wall_direction[0]) == np.sign(crossing_direction[0])) and (
                                abs(crossing_direction[0]) > 0)
                    y_crossed = (np.sign(wall_direction[1]) == np.sign(crossing_direction[1])) and (
                                abs(crossing_direction[1]) > 0)

                    # Return the direction(s) in which the wall was crossed
                    if x_crossed and y_crossed:
                        return np.array([True, True])
                    elif x_crossed:
                        return np.array([True, False])
                    elif y_crossed:
                        return np.array([False, True])

        return np.array([False, False])

    @staticmethod
    def is_angle_larger(v1, v2, angle=180):
        # Calculate the cosine of the angle between the vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # Check if the angle is larger than the specified one
        return cos_angle > np.cos(np.radians(angle))

    @staticmethod
    def perpendicular_vec(vec):
        """Finds a normalized vector to vec that is normalized.
        """
        perp_vec = np.ones(2)
        perp_vec[0] = - vec[1] / vec[0]
        return perp_vec / np.linalg.norm(perp_vec)

    def bounce(self, positions, prop_v):
        """Checks if a proposed step violates any boundaries of the environment and if so, changes it such that
        the proposed step is a valid move.
        """

        # Check if a wall was crossed
        wall_crossed = np.array([self.crosses_which_wall(p, p + pv) for p, pv in zip(positions, prop_v)])

        # Invert proposed vectors for the ones that violate boundary conditions
        prop_v[wall_crossed] *= -1

        # Re-check boundary conditions and wall crossing after inverting
        wall_crossed = np.array([self.crosses_wall(p, p + pv) for p, pv in zip(positions, prop_v)])

        # Rotate vectors that still don't satisfy boundary conditions
        angle_step = 0.1
        speed_decrease = 0.9
        while np.any(wall_crossed):
            # Rotate vector clockwise, decrease speed with every step
            cos_angle = np.cos(angle_step)
            sin_angle = np.sin(angle_step)
            prop_v[wall_crossed] = np.column_stack((
                cos_angle * prop_v[wall_crossed, 0] - sin_angle * prop_v[wall_crossed, 1],
                sin_angle * prop_v[wall_crossed, 0] + cos_angle * prop_v[wall_crossed, 1]
            )) * speed_decrease

            # Update conditions
            wall_crossed = np.array([self.crosses_wall(p, p + pv) for p, pv in zip(positions, prop_v)])

            # Increment angle
            angle_step += 0.1

        return prop_v

    def generate_random_trajectories(self, n_steps, n_traj, rayleigh_scale=0.05, vonmises_kappa=4 * np.pi):
        """Generates a number of trajectories of specified length in this environment.
        """

        # Generate random speeds and directions
        speeds = rayleigh.rvs(scale=rayleigh_scale, size=(n_traj, n_steps))
        prev_hd = np.random.uniform(0, 2 * np.pi, n_traj)
        positions = np.zeros((n_traj, n_steps, 2))

        # Initialize the trajectories with random starting points
        for i in range(n_traj):
            while True:
                x = np.random.uniform(self.polygon.bounds[0], self.polygon.bounds[2])
                y = np.random.uniform(self.polygon.bounds[1], self.polygon.bounds[3])
                if self.polygon.contains(Point(x, y)):
                    positions[i, 0] = [x, y]
                    break

        # Generate the trajectories
        for step in range(1, n_steps):
            # Compute head direction
            hd = np.random.vonmises(prev_hd, vonmises_kappa, n_traj)

            # Proposed vector
            prop_v = speeds[:, step, None] * np.stack((np.cos(hd), np.sin(hd)), axis=-1)

            # Calculate vector that respects boundary conditions of env
            v = self.bounce(positions[:, step - 1], prop_v)

            # Update head direction based on actually taken step
            prev_hd = np.arctan2(v[:, 1], v[:, 0])

            # Update position using dt = 1
            positions[:, step] = positions[:, step - 1] + v

        # Determine speed
        speeds = np.diff(positions, axis=1).astype(np.float32)

        return torch.Tensor(positions.astype(np.float32)), torch.Tensor(speeds)

    def generate_random_samples(self, n_samples):
        """Generates n_samples random samples in this environment.
        """
        samples = []
        while len(samples) < n_samples:
            x = np.random.uniform(self.polygon.bounds[0], self.polygon.bounds[2])
            y = np.random.uniform(self.polygon.bounds[1], self.polygon.bounds[3])
            if self.polygon.contains(Point(x, y)):
                samples.append([x, y])
        return torch.Tensor(np.array(samples).astype(np.float64))

    def add_random_walls(self, n_walls, min_gap=0.1, min_len=0.2):
        """Adds n_walls random walls to the environment.
        """
        added_walls = 0
        while added_walls < n_walls:

            # Sample a random point in the environment
            while True:
                x, y = np.random.uniform(self.polygon.bounds[0], self.polygon.bounds[2]), np.random.uniform(
                    self.polygon.bounds[1], self.polygon.bounds[3])
                p1 = np.array([x, y])
                if self.polygon.contains(Point(p1)):
                    break

            # Get list of candidates
            candidates = shortest_vectors_from_points_to_lines(p1, self.walls)[0]

            # Get norms of candidates
            norms = np.linalg.norm(candidates, axis=1)

            if np.min(norms) > min_gap:

                # Get subset of candidates that are long enough
                candidates = candidates[norms > min_len]

                if candidates.shape[0] > 0:

                    # Choose random candidate
                    idx = np.random.randint(0, candidates.shape[0])

                    # Get second point
                    p2 = p1 - candidates[idx]

                    # Check for collisions
                    if not np.any(self.crosses_wall(p1, p2)):
                        # Get normalized perpendicular vector
                        perp_vec = self.perpendicular_vec(p2 - p1)

                        # Generate point 3 and 4
                        p3 = p2 + perp_vec * 0.05
                        p4 = p1 + perp_vec * 0.05

                        self.polygon -= Polygon([p1, p2, p3, p4, p1])

                        self.walls.append(LineString([p1, p2]))
                        self.walls.append(LineString([p2, p3]))
                        self.walls.append(LineString([p3, p4]))
                        self.walls.append(LineString([p4, p1]))
                        added_walls += 1

    def add_random_holes(self, n_holes, min_hole_points=3, max_hole_points=4, min_dist=0.3, **params):
        """Adds random holes to the Shapely polygon object.

        Parameters
        ----------
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
                        self.polygon.bounds[0] - np.sign(self.polygon.bounds[0]) * min_dist,
                        self.polygon.bounds[2] - np.sign(self.polygon.bounds[2]) * min_dist),
                    np.random.uniform(
                        self.polygon.bounds[1] - np.sign(self.polygon.bounds[1]) * min_dist,
                        self.polygon.bounds[3] - np.sign(self.polygon.bounds[3]) * min_dist
                    )
                )
                if self.polygon.contains(point):
                    hole_points.append(point)

            # Create a hole polygon from the generated points
            points = [point.coords[0] for point in hole_points]
            hole_polygon = Polygon(points)

            # Subtract the hole polygon from the original polygon
            self.polygon = self.polygon.difference(hole_polygon)

            # Add to walls
            for i in range(len(points)):
                self.walls.append(LineString([points[(i + 1) if (i + 1) < len(points) else 0], points[i]]))

    def plot_env(self, ax=None, show=False, show_scale=False, figsize=(5, 5)):
        """Plots the environment.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            The axes to plot on.
        show: bool
            Whether to show the plot.
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Plot the polygon
        x, y = self.polygon.exterior.xy
        ax.plot(x, y, color='black')

        # Plot the walls
        for wall in self.walls:
            x, y = wall.xy
            ax.plot(x, y, color='black')

        # Remove top and right spines
        if not show_scale:
            ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.spines[['top', 'right']].set_visible(False)

        if show:
            plt.show()

        return fig, ax