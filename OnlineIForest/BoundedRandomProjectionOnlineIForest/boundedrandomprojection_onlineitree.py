from ..onlineitree import OnlineITree
from ..onlineinode import OnlineINode
from numpy import argsort, empty, inf, ndarray, ones, sort, split, vstack, zeros
from numpy.linalg import norm
from numpy.random import choice, normal, random, uniform


class BoundedRandomProjectionOnlineITree(OnlineITree):
    def __init__(self, max_leaf_samples: int, type: str, subsample: float, branching_factor: int, data_size: int,
                 metric: str = 'axisparallel', depth_limit: str = 'code'):
        super().__init__(max_leaf_samples, type, subsample, branching_factor, data_size)
        self.metric: str = metric
        # [Modified] Depth limit δ. The paper (Sec. 4.1, Table 3) sets δ = log2(ω/η), but the released code computes
        # it as log(ω/η)/log(2·branching_factor) = log4(ω/η), i.e. half as deep (3 instead of 6 for ω=2048, η=32).
        # 'code' keeps the released behaviour; 'paper' uses log2(ω/η) as written in the paper.
        # get_random_path_length divides by log(2·b), so passing b = 1 gives log base 2.
        if depth_limit not in ('code', 'paper'):
            raise ValueError('Bad depth_limit {}'.format(depth_limit))
        self.depth_limit_branching: float = 1 if depth_limit == 'paper' else branching_factor

    def learn(self, data: ndarray) -> 'BoundedRandomProjectionOnlineITree':
        # Subsample data in order to improve diversity among trees
        data: ndarray = data[random(data.shape[0]) < self.subsample]
        if data.shape[0] >= 1:
            # Update the counter of data seen so far
            self.data_size += data.shape[0]
            # Adjust depth limit according to data seen so far and branching factor
            self.depth_limit: float = OnlineITree.get_random_path_length(self.depth_limit_branching,
                                                                         self.max_leaf_samples, self.data_size)
            # Recursively update the tree
            self.next_node_index, self.root = self.recursive_build(data) if self.root is None else self.recursive_learn(self.root, data, self.next_node_index)
        return self

    def recursive_learn(self, node: OnlineINode, data: ndarray, node_index: int,
                        constraints: tuple = ()) -> (int, OnlineINode):
        # [Modified] constraints: the oblique cuts on the path from the root to this node, as
        # (projection_vector, split_values, child_index) triples. Only used by 'oblique_constrained'.
        # Update the number of data seen so far by the current node
        node.data_size += data.shape[0]
        # Update the vectors of minimum and maximum values seen so far by the current node
        node.min_values: ndarray = vstack([data, node.min_values]).min(axis=0)
        node.max_values: ndarray = vstack([data, node.max_values]).max(axis=0)
        # If the current node is a leaf, try to split it
        if node.children is None:
            # If there are enough samples to be split according to the max leaf samples and the depth limit has not been
            # reached yet, split the node
            if node.data_size >= self.max_leaf_samples*OnlineITree.get_multiplier(self.type, node.depth) and node.depth < self.depth_limit:
                # Sample data_size points uniformly at random within the bounding box defined by the vectors of minimum
                # and maximum values of data seen so far by the current node
                if self.metric == 'oblique_constrained' and constraints:
                    data_sampled: ndarray = self.sample_in_region(node, constraints, data.shape[1])
                else:
                    data_sampled: ndarray = uniform(node.min_values, node.max_values, size=(node.data_size, data.shape[1]))
                return self.recursive_build(data_sampled, depth=node.depth, node_index=node_index)
            else:
                return node_index, node
        # If the current node is not a leaf, recursively update all its children
        else:
            # Partition data
            partition_indices: list[ndarray[int]] = self.split_data(data, node.projection_vector, node.split_values)
            # Recursively update children
            for i, indices in enumerate(partition_indices):
                child_constraints: tuple = constraints + ((node.projection_vector, node.split_values, i),) \
                    if self.metric == 'oblique_constrained' else ()
                node_index, node.children[i] = self.recursive_learn(node.children[i], data[indices], node_index,
                                                                    child_constraints)
            return node_index, node

    def sample_in_region(self, node: OnlineINode, constraints: tuple, dim: int) -> ndarray:
        # [Modified] With oblique cuts, a node's region is (bounding box) ∩ (half-spaces of all ancestor cuts).
        # The axis-aligned bounding box alone over-covers that region, so uniform sampling in the box puts
        # synthetic points where no data can be. We rejection-sample: draw in the box, keep only points that
        # fall on the correct side of every ancestor cut.
        n: int = node.data_size
        kept: list = []
        total: int = 0
        for _ in range(50):
            candidates: ndarray = uniform(node.min_values, node.max_values, size=(n, dim))
            mask: ndarray = ones(n, dtype=bool)
            for projection_vector, split_values, child_index in constraints:
                projected: ndarray = candidates @ projection_vector
                if child_index > 0:
                    mask &= projected >= split_values[child_index - 1]
                if child_index < len(split_values):
                    mask &= projected < split_values[child_index]
            kept.append(candidates[mask])
            total += int(mask.sum())
            if total >= n:
                break
        points: ndarray = vstack(kept)
        if points.shape[0] == 0:
            # Region too thin to hit: fall back to the original behaviour
            return uniform(node.min_values, node.max_values, size=(n, dim))
        if points.shape[0] < n:
            points = points[choice(points.shape[0], n)]
        return points[:n]

    def recursive_build(self, data: ndarray, depth: int = 0, node_index: int = 0) -> (int, OnlineINode):
        # If there aren't enough samples to be split according to the max leaf samples or the depth limit has been
        # reached, build a leaf node
        if data.shape[0] < self.max_leaf_samples*OnlineITree.get_multiplier(self.type, depth) or depth >= self.depth_limit:
            return node_index + 1, OnlineINode.create('boundedrandomprojectiononlineinode',
                                                      data_size=data.shape[0], children=None, depth=depth,
                                                      node_index=node_index, min_values=data.min(axis=0, initial=inf),
                                                      max_values=data.max(axis=0, initial=-inf), projection_vector=None,
                                                      split_values=None)
        else:
            # Sample projection vector
            if self.metric == 'axisparallel':
                projection_vector: ndarray[float] = zeros(data.shape[1])
                projection_vector[choice(projection_vector.shape[0])]: float = 1.0
            elif self.metric in ('oblique', 'oblique_constrained'):
                # [Modified] Oblique split: random direction instead of a single axis.
                # Each component is divided by the node's extent along that axis so that features with
                # large numeric ranges do not dominate the direction (scale invariance, computed online
                # from the node's own bounding box, no global normalisation needed).
                extent: ndarray[float] = data.max(axis=0) - data.min(axis=0)
                active: ndarray[bool] = extent > 0
                projection_vector: ndarray[float] = zeros(data.shape[1])
                if active.any():
                    projection_vector[active] = normal(size=active.sum()) / extent[active]
                else:
                    projection_vector[choice(projection_vector.shape[0])] = 1.0
            else:
                raise ValueError('Bad metric {}'.format(self.metric))
            projection_vector: ndarray[float] = projection_vector / norm(projection_vector)
            # Project sampled data using projection vector
            projected_data: ndarray = data @ projection_vector
            # Sample split values
            split_values: ndarray[float] = sort(uniform(min(projected_data), max(projected_data),
                                                        size=self.branching_factor - 1))
            # Partition sampled data
            partition_indices: list[ndarray[int]] = self.split_data(data, projection_vector, split_values)
            # Generate recursively children nodes
            children: ndarray[OnlineINode] = empty(shape=(self.branching_factor,), dtype=OnlineINode)
            for i, indices in enumerate(partition_indices):
                node_index, children[i] = self.recursive_build(data[indices], depth + 1, node_index)
            return node_index + 1, OnlineINode.create('boundedrandomprojectiononlineinode',
                                                      data_size=data.shape[0], children=children, depth=depth,
                                                      node_index=node_index, min_values=data.min(axis=0),
                                                      max_values=data.max(axis=0), projection_vector=projection_vector,
                                                      split_values=split_values)

    def unlearn(self, data: ndarray) -> 'BoundedRandomProjectionOnlineITree':
        # Subsample data in order to improve diversity among trees
        data: ndarray = data[random(data.shape[0]) < self.subsample]
        if data.shape[0] >= 1:
            # Update the counter of data seen so far
            self.data_size -= data.shape[0]
            # Adjust depth limit according to data seen so far and branching factor
            self.depth_limit: float = OnlineITree.get_random_path_length(self.depth_limit_branching,
                                                                         self.max_leaf_samples, self.data_size)
            # Recursively update the tree
            self.root: OnlineINode = self.recursive_unlearn(self.root, data)
        return self

    def recursive_unlearn(self, node: OnlineINode, data: ndarray) -> OnlineINode:
        # Update the number of data seen so far by the current node
        node.data_size -= data.shape[0]
        # If the current node is a leaf, return it
        if node.children is None:
            return node
        # If the current node is not a leaf, try to unsplit it
        else:
            # If there are not enough samples according to max leaf samples, unsplit the node
            if node.data_size < self.max_leaf_samples*OnlineITree.get_multiplier(self.type, node.depth):
                return self.recursive_unbuild(node)
            # If there are enough samples according to max leaf samples, recursively update all its children
            else:
                # Partition data
                partition_indices: list[ndarray[int]] = self.split_data(data, node.projection_vector, node.split_values)
                # Recursively update children
                for i, indices in enumerate(partition_indices):
                    node.children[i]: OnlineINode = self.recursive_unlearn(node.children[i], data[indices])
                # Update the vectors of minimum and maximum values seen so far by the current node
                node.min_values: ndarray = vstack([node.children[i].min_values for i, _ in enumerate(node.children)]).min(axis=0)
                node.max_values: ndarray = vstack([node.children[i].max_values for i, _ in enumerate(node.children)]).max(axis=0)
                return node

    def recursive_unbuild(self, node: OnlineINode) -> OnlineINode:
        # If the current node is a leaf, return it
        if node.children is None:
            return node
        # If the current node is not a leaf, unbuild it
        else:
            # Recursively unbuild children
            for i, _ in enumerate(node.children):
                node.children[i]: OnlineINode = self.recursive_unbuild(node.children[i])
            # Update the vectors of minimum and maximum values seen so far by the current node
            node.min_values: ndarray = vstack([node.children[i].min_values for i, _ in enumerate(node.children)]).min(axis=0)
            node.max_values: ndarray = vstack([node.children[i].max_values for i, _ in enumerate(node.children)]).max(axis=0)
            # Delete children nodes, projection vector and split values
            node.children: ndarray[OnlineINode] = None
            node.projection_vector: ndarray[float] = None
            node.split_values: ndarray[float] = None
            return node

    def predict(self, data: ndarray) -> ndarray[float]:
        # Compute depth of each sample
        return self.recursive_depth_search(self.root, data, empty(shape=(data.shape[0],), dtype=float))

    def recursive_depth_search(self, node: OnlineINode, data: ndarray, depths: ndarray[float]) -> ndarray[float]:
        # If the current node is a leaf, fill the depths vector with the current depth plus a normalization factor
        if node.children is None or data.shape[0] == 0:
            depths[:] = node.depth + OnlineITree.get_random_path_length(self.branching_factor, self.max_leaf_samples,
                                                                        node.data_size)
        else:
            # Partition data
            partition_indices: list[ndarray[int]] = self.split_data(data, node.projection_vector, node.split_values)
            # Fill the vector of depths
            for i, indices in enumerate(partition_indices):
                depths[indices]: ndarray[float] = self.recursive_depth_search(node.children[i], data[indices],
                                                                              depths[indices])
        return depths

    def split_data(self, data: ndarray, projection_vector: ndarray[float], split_values: ndarray[float]) -> list[ndarray[int]]:
        # Project data using projection vector
        projected_data: ndarray = data @ projection_vector
        # Sort projected data and keep sort indices
        sort_indices: ndarray = argsort(projected_data)
        # Split data according to their membership
        partition: list[ndarray[int]] = split(sort_indices, projected_data[sort_indices].searchsorted(split_values))
        return partition
