import unittest
import numpy as np


def crop(
    image: np.array,
    pos: tuple[int, int],
    cropping: tuple[int, int],
    strict: bool = True,
) -> np.array:
    """
    Crops 2D array at given position and with given cropping.

    Args:
        image (np.array):
            2D array to crop.
        pos (tuple[int, int]):
            Center position for cropping.
        cropping (tuple[int, int]):
            Size of the cropping along (y, x).
        strict (bool, optional (default=True)):
            If `False` allows for the cropping to be adapted
            wrt the array edges when they are exceeded.
    
    Returns:
        output (np.array): Cropped 2D array (shape twice the `cropping`).
    
    Raises:
        ValueError: If cropping is not a positive int tuple.
        IndexError: If cropping wrt indexes exceeds 2D array edges
                    (only if `strict` is `True`).
    
    Notes:
        - Negative indexes are allowed.
    """
    n, m = image.shape
    y, x = pos
    cy, cx = cropping

    if cy <= 0 or cx <= 0:
        raise ValueError("Cropping must be a tuple of positive integers.")

    flagx = (((0 <= x - cx) and (x + cx < m)) or ((cx - x <= m) and (x + cx < 0)))
    flagy = (((0 <= y - cy) and (y + cy < n)) or ((cy - y <= n) and (y + cy < 0)))
    
    if not (flagx and flagy):
        if not strict:
            # the crop extends up to the 2nd row/col from top/bottom/left/right
            if not flagx:
                cx = min(x - 2, m - x - 3) if x > 0 else min(x + m + 2, -x - 2)
            if not flagy:
                cy = min(y - 2, n - y - 3) if y > 0 else min(y + n + 2, -y - 2)
            print(f"Cropping {cropping} at pos {pos} exceeds array edges, new cropping: {cy, cx}")
        else:
            raise IndexError(f"Cropping {cropping} at pos {pos} exceeds array edges.")
    
    return image[y - cy : y + cy + 1, x - cx : x + cx + 1]



class TestCropping(unittest.TestCase):
    """Test for the `crop()` method in `show.py`."""

    def test_errors(self):
        """Test for input values."""
        n, m = 20, 20
        pos = (5, 5)
        cr = 10
        a = np.random.randint(0, 10, (n, m))

        with self.assertRaises(ValueError):
            crop(a, pos, (-cr, cr))
            crop(a, pos, (cr, -cr))
            crop(a, pos, (-cr, -cr))
        
        with self.assertRaises(IndexError):
            crop(a, (n // 4, m // 4), (cr, cr))
            crop(a, (n // 4, -m // 4), (cr, cr))
            crop(a, (-n // 4, m // 4), (cr, cr))
            crop(a, (-n // 4, -m // 4), (cr, cr))
    
    def test_strict_cropping(self):
        """Test for strict cropping."""
        n, m = 20, 20
        cr = 2
        a = np.zeros((n, m))
        target_shape = (2 * cr + 1, 2 * cr + 1)

        self.assertEqual(
            crop(a, (n // 4, m // 4), (cr, cr), strict=False).shape,
            target_shape,
        )
        self.assertEqual(
            crop(a, (n // 4, -m // 4), (cr, cr), strict=False).shape,
            target_shape,
        )
        self.assertEqual(
            crop(a, (-n // 4, m // 4), (cr, cr), strict=False).shape,
            target_shape,
        )
        self.assertEqual(
            crop(a, (-n // 4, -m // 4), (cr, cr), strict=False).shape,
            target_shape,
        )
    
    def test_adaptable_cropping(self):
        """Test for adaptable cropping."""
        n, m = 20, 20
        cr = 10
        a = np.zeros((n, m))
        target_shape = (2 * (n // 4 - 2) + 1, 2 * (m // 4 - 2) + 1)

        print(
            target_shape,
            crop(a, (n // 4, m // 4), (cr, cr), strict=False).shape,
            crop(a, (n // 4, -m // 4), (cr, cr), strict=False).shape,
            crop(a, (-n // 4, m // 4), (cr, cr), strict=False).shape,
            crop(a, (-n // 4, -m // 4), (cr, cr), strict=False).shape,
        )

        #self.assertEqual(
        #    crop(a, (n // 4, m // 4), (cr, cr), strict=False).shape,
        #    target_shape,
        #)
        #self.assertEqual(
        #    crop(a, (n // 4, -m // 4), (cr, cr), strict=False).shape,
        #    target_shape,
        #)
        #self.assertEqual(
        #    crop(a, (-n // 4, m // 4), (cr, cr), strict=False).shape,
        #    target_shape,
        #)
        #self.assertEqual(
        #    crop(a, (-n // 4, -m // 4), (cr, cr), strict=False).shape,
        #    target_shape,
        #)




if __name__ == "__main__":
    unittest.main()