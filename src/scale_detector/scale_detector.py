import numpy as np
from sklearn.cluster import DBSCAN

from src.number_detector.number_detector import NumberDetector
from src.ray_casting.intersection_seeker import IntersectionSeeker
from src.ray_casting.segment import Segment


class ScaleDetector:

    #grouping: 'median', 'cluster', 'IQR', 'z-score'

    @staticmethod
    def get_segments_len(segments):
        return np.array([len(segment) for segment in segments])

    @staticmethod
    def get_mean_segments_len(segments):
        if not len(segments):
            return

        return np.mean([segment.length() for segment in segments])

    @staticmethod
    def get_median_segments_len(segments):
        if not len(segments):
            return

        return np.median([segment.length() for segment in segments])

    @classmethod
    def modified_z_score(cls, data):
        median = np.median(data)
        deviation_from_median = data - median
        mad = np.median(np.abs(deviation_from_median))
        
        # Check if MAD is zero to avoid division by zero
        if mad == 0:
            # If MAD is zero, all data is identical, Z score is 0
            modified_z_scores = np.zeros_like(data)
        else:
            modified_z_scores = 0.6745 * deviation_from_median / mad
        return modified_z_scores

    @classmethod
    def cluster_segments_Z_score(cls, candidate_segments):
        # Compute the segment lengths
        segment_lengths = cls.get_segments_len(candidate_segments)
        # Compute the modified Z scores
        z_scores = cls.modified_z_score(segment_lengths)
        # Identify segments that fall within the threshold
        segments_within_z_score = [seg for seg, z_score in zip(candidate_segments, z_scores) 
                                if abs(z_score) < 3.5]
        return segments_within_z_score
    
    @classmethod
    def cluster_segments_DBSCAN(cls, candidate_segments, dbscan_args=None):
        if dbscan_args:
            eps = dbscan_args['eps']
            min_samples = dbscan_args['min_samples']
        else:
            eps = 20
            min_samples = 1

        # We need to reshape to be able to cluster
        segment_lengths = cls.get_segments_len(candidate_segments).reshape(-1, 1)
        
        # Fit the DBSCAN model to the data
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(segment_lengths)

        # Extract the cluster labels and find the label of the largest cluster
        labels = clustering.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Exclude noise (label=-1) from the counts
        non_noise_labels = unique_labels[unique_labels != -1]
        non_noise_counts = counts[unique_labels != -1]
        
        # If no non-noise clusters exist, return an empty list
        if len(non_noise_counts) == 0:
            return []

        largest_cluster_label = non_noise_labels[np.argmax(non_noise_counts)]

        # Filter segments that are in the largest cluster
        largest_cluster_segments = [seg for seg, label in zip(candidate_segments, labels) 
                                    if label == largest_cluster_label]

        return largest_cluster_segments

    @staticmethod
    def compute_iqr(segment_lengths):
        # Compute 1st and 3rd quartiles (Q1 and Q3)
        Q1 = np.percentile(segment_lengths, 25)
        Q3 = np.percentile(segment_lengths, 75)

        # Compute IQR
        IQR = Q3 - Q1

        # Compute "whiskers"
        lower_whisker = Q1 - 1.5 * IQR
        upper_whisker = Q3 + 1.5 * IQR

        return lower_whisker, upper_whisker

    @classmethod
    def get_candidate_segments(cls, numbers, grouping='cluster', dbscan_args=None):
        segments = []

        for current_number, next_number in zip(numbers, numbers[1:]):
            current_digit_name = int(current_number.name)
            next_digit_name = int(next_number.name)

            if next_digit_name == current_digit_name + 1:
                point = current_number.center()
                next_point = next_number.center()
                test_segment = Segment(point, next_point)
                segments.append(test_segment)

        if not segments:
            return []

        if grouping == 'cluster':
            return cls.cluster_segments_DBSCAN(segments, dbscan_args)
        elif grouping == 'IQR':
            return cls.cluster_segments_IQR(segments)
        elif grouping == 'z-score':
            return cls.cluster_segments_Z_score(segments)

        return segments

    @classmethod
    def get_segments_within_iqr(cls, candidate_segments):
        if len(candidate_segments) <= 1:
            return candidate_segments

        # We need to reshape to be able to compute IQR
        segment_lengths = cls.get_segments_len(candidate_segments).reshape(-1, 1)

        lower_whisker, upper_whisker = cls.compute_iqr(segment_lengths)

        # Filter segments whose length falls within the whiskers
        segments_within_iqr = [seg for seg, len_seg in zip(candidate_segments, segment_lengths) 
                            if lower_whisker <= len_seg <= upper_whisker]

        return segments_within_iqr

    @classmethod
    def cluster_segments_IQR(cls, candidate_segments):
        # Identify segments that fall within the IQR
        segments_within_iqr = cls.get_segments_within_iqr(candidate_segments)

        return segments_within_iqr

    @classmethod
    def get_ruler_scale(cls, candidate_segments, grouping='cluster'):
        if grouping == 'median':
            return round(cls.get_median_segments_len(candidate_segments))

        return round(cls.get_mean_segments_len(candidate_segments))

    @classmethod
    def detect_ruler_scale(cls, image_path, grouping, model_type, min_segments=3, dbscan_args=None):
        all_numbers = NumberDetector.get_numbers(image_path, model_type)

        # Find numbers, that lie on the one line
        numbers_on_one_line = IntersectionSeeker.get_candidate_numbers(all_numbers, min_segments=min_segments)

        # Get candidate segments based on the found numbers
        candidate_segments = cls.get_candidate_segments(numbers_on_one_line, grouping, dbscan_args)

        # Calculate the mean length of the ruler scale if any segments were found, otherwise None
        mean_len = cls.get_ruler_scale(candidate_segments, grouping) if candidate_segments else None

        return candidate_segments, mean_len, all_numbers, numbers_on_one_line
