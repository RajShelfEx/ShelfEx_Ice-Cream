import torch


class similarBoundingBoxes:

    def __init__(self):
        pass

    # Function to filter out bounding boxes based on highest confidence
    def FilterBoxes(self, boxes, confidences, SimilarIndices):
        """
        Filters out similar bounding boxes based on confidence scores, retaining only the one with the highest confidence within each group of similar boxes.

        Parameters:
        - boxes (torch.Tensor): A tensor of bounding boxes with shape (N, 4), where N is the number of bounding boxes.
        - confidences (torch.Tensor): A tensor of confidence scores for each bounding box with shape (N,).
        - SimilarIndices (list[list[int]]): A list of lists, where each inner list contains indices of bounding boxes considered similar to each other.

        Returns:
        - KeepIndices (list[int]): A sorted list of indices of bounding boxes to keep based on the highest confidence within each similar group.

        Process:
        1. Initializes a set of indices to keep all bounding boxes initially.
        2. Iterates over groups of similar bounding boxes.
        3. For each group, extracts confidence scores and identifies the bounding box with the highest confidence.
        4. Discards all other bounding boxes in the group from the set of indices to keep.
        5. Returns a sorted list of the remaining indices.
        """
        KeepIndices = set(range(len(boxes)))  # All indices to keep initially

        for indices in SimilarIndices:
            if len(indices) > 1:
                # Extract confidence scores for these indices
                ConfScores = confidences[indices]
                # Find the index with the highest confidence score
                MaxConfIndex = indices[torch.argmax(ConfScores)]
                # Remove all other indices from KeepIndices
                for idx in indices:
                    if idx != MaxConfIndex:
                        KeepIndices.discard(idx)
        
        # Convert KeepIndices to sorted list
        KeepIndices = sorted(list(KeepIndices))
        return KeepIndices
        # return boxes[KeepIndices], confidences[KeepIndices]

    # Group similar rows
    def FindSimilarGroups(self, similar_matrix):
        """_summary_

        Filters out similar bounding boxes based on confidence scores, keeping only the one with the highest confidence within each group of similar boxes.

        Parameters:
        - boxes (torch.Tensor): A tensor of bounding boxes in (x1, y1, x2, y2) format, with shape (N, 4).
        - confidences (torch.Tensor): A tensor of confidence scores for each bounding box, with shape (N,).
        - SimilarIndices (list[list[int]]): A list of lists, where each inner list contains indices of bounding boxes that are considered similar.

        Returns:
        - KeepIndices (list[int]): A sorted list of indices of bounding boxes to keep based on highest confidence within similar groups.
        
        Process:
        1. Initializes a set of indices to keep.
        2. Iterates over groups of similar bounding boxes.
        3. For each group, determines the bounding box with the highest confidence score.
        4. Discards all other bounding boxes in the group from the set of indices to keep.
        5. Returns the final list of indices to keep.
        """
        visited = set()
        groups = []

        def dfs(node, group):
            stack = [node]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    group.append(current)
                    # Find neighbors by checking which rows are similar
                    neighbors = torch.nonzero(similar_matrix[current], as_tuple=True)[0].tolist()
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            stack.append(neighbor)

        for row in range(similar_matrix.size(0)):
            if row not in visited:
                group = []
                dfs(row, group)
                groups.append(group)
        
        return groups
       
    def IndexesOfSimilarCord(self, boxes, confidences, SkuIndexes):
        """
        Filters out similar bounding boxes based on a defined similarity threshold.

        Parameters:
        - boxes (torch.Tensor): A tensor of bounding boxes in (x1, y1, x2, y2) format, with shape (N, 4), where N is the number of bounding boxes.
        - confidences (torch.Tensor): A tensor of confidence scores for each bounding box, with shape (N,).
        - SkuIndexes (list[int]): A list of SKU indexes corresponding to each bounding box.

        Returns:
        - filtered_boxes (torch.Tensor): A tensor of filtered bounding boxes after removing similar ones.
        - filtered_confidences (torch.Tensor): A tensor of confidence scores corresponding to the filtered bounding boxes.
        - filtered_sku_indexes (list[int]): A list of SKU indexes corresponding to the filtered bounding boxes.

        Process:
        1. Calculates pairwise differences between bounding boxes.
        2. Determines if bounding boxes are similar based on a threshold.
        3. Groups similar bounding boxes.
        4. Filters out groups with fewer bounding boxes than a minimum required number.
        5. Keeps bounding boxes from the filtered groups and removes others.
        
        """
        # Define the similarity threshold
        threshold = 2
        # Calculate the differences
        differences = boxes.unsqueeze(0) - boxes.unsqueeze(1)
        # Calculate the absolute differences
        AbsDifferences = torch.abs(differences)

        # Check if all differences are less than the threshold
        SimilarRows = torch.all(AbsDifferences < threshold, dim=2)
        # Find groups of similar rows
        SimilarGroups = self.FindSimilarGroups(SimilarRows)

        # Filter groups by size
        n = 2  # Number of similar bounding boxes required
        FilteredGroups = [group for group in SimilarGroups if len(group) >= n]

        # Print filtered groups
        SimilarIndices = []
        for group in FilteredGroups:
            print(f"Group of {len(group)} similar rows: {group}")
            SimilarIndices.append(group)
            
        if len(SimilarIndices) == 0:
            # No duplicates bounding boxes
            return boxes, confidences, SkuIndexes
        else:
            KeepIndices = self.FilterBoxes(boxes, confidences, SimilarIndices)
            SkuIndexes = [SkuIndexes[i] for i in KeepIndices]
            return boxes[KeepIndices], confidences[KeepIndices], SkuIndexes
            
            

