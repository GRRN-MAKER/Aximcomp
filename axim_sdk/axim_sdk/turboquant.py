import math
import random
from typing import List, Tuple

class TurboQuant:
    """
    Implementation of the TurboQuant compression algorithm principles
    from Google Research (PolarQuant + QJL).
    """
    def __init__(self, original_dim: int, target_dim: int = 64):
        self.original_dim = original_dim
        self.target_dim = target_dim
        # Generate a random projection matrix for Johnson-Lindenstrauss Transform
        # Using normal distribution N(0, 1)
        self.projection_matrix = [
            [random.gauss(0, 1) for _ in range(original_dim)]
            for _ in range(target_dim)
        ]

    def _polar_quant(self, vector: List[float]) -> List[Tuple[float, float]]:
        """
        PolarQuant: Converts Cartesian coordinates to Polar representations
        for more efficient boundary distributions.
        Groups pairs of coordinates and converts them to (radius, angle).
        """
        polar_encoded = []
        for i in range(0, len(vector), 2):
            x = vector[i]
            y = vector[i+1] if i + 1 < len(vector) else 0.0
            
            radius = math.sqrt(x**2 + y**2)
            angle = math.atan2(y, x)
            polar_encoded.append((radius, angle))
            
        return polar_encoded

    def _qjl_transform(self, vector: List[float]) -> List[int]:
        """
        Quantized Johnson-Lindenstrauss (QJL):
        Projects the vector to a lower dimension and quantizes to 1-bit (sign bit).
        """
        projected = []
        for i in range(self.target_dim):
            # Dot product
            val = sum(self.projection_matrix[i][j] * vector[j] for j in range(self.original_dim))
            # 1-bit Quantization (Sign bit: +1 or -1)
            projected.append(1 if val >= 0 else -1)
        return projected

    def compress(self, vector: List[float]) -> dict:
        """
        Applies TurboQuant compression to a high-dimensional vector.
        """
        if len(vector) != self.original_dim:
            # Pad or truncate if dimensions don't match
            if len(vector) < self.original_dim:
                vector = vector + [0.0] * (self.original_dim - len(vector))
            else:
                vector = vector[:self.original_dim]

        polar_data = self._polar_quant(vector)
        qjl_bits = self._qjl_transform(vector)
        
        return {
            "polar_radii": [p[0] for p in polar_data],
            "polar_angles": [p[1] for p in polar_data],
            "qjl_bits": qjl_bits,
            "compressed_size_bits": len(qjl_bits) # 1 bit per projected dimension
        }

    def compute_similarity(self, qjl_bits1: List[int], qjl_bits2: List[int]) -> float:
        """
        Computes similarity between two compressed vectors using Hamming distance
        on the QJL 1-bit signatures.
        """
        matches = sum(1 for a, b in zip(qjl_bits1, qjl_bits2) if a == b)
        # Normalize to -1.0 to 1.0 range similar to cosine similarity
        similarity = (2.0 * matches / self.target_dim) - 1.0
        return similarity
