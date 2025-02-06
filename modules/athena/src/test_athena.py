import unittest
from model import create_athena
from utils import ChessDataGenerator
from config import TEST_DATA

class TestAthena(unittest.TestCase):
    def test_model_output_shape(self):
        model = create_athena()
        self.assertEqual(model.output[0].shape, (None, 64, 64))  # Move probs
        self.assertEqual(model.output[1].shape, (None, 1))      # Criticality
    
    def test_data_generator(self):
        gen = ChessDataGenerator(TEST_DATA, batch_size=32)
        batch = gen[0]
        self.assertEqual(len(batch), 2)  # Inputs and targets

if __name__ == "__main__":
    unittest.main()