from __future__ import annotations
import unittest
import keras, numpy as np
from examples.rps import (
   RockPaperScissors, ValidationAccuracyCallback, data_generator, train_generator, validation_generator, get_random_image, BATCH_SIZE
)

class TestRockPaperScissors(unittest.TestCase):
    def setUp(self):
        self.train_generator = data_generator('training')
        self.validation_generator = data_generator('validation')
        self.class_names = self.train_generator.class_names
        self.rps_model = RockPaperScissors(classes=len(self.class_names))
        
    def test_model_compile(self):
        self.rps_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.assertIsNotNone(self.rps_model.optimizer)
        
    def test_data_generator(self):
        self.assertIsNotNone(self.train_generator)
        self.assertIsNotNone(self.validation_generator)
        
    def test_training(self):
        history = self.rps_model.fit(train_generator, epochs=1, validation_data=validation_generator, callbacks=[ValidationAccuracyCallback()])
        self.assertIsInstance(history.history, dict)
        
    def test_prediction(self):
        img = get_random_image(self.class_names)
        pred_img = keras.preprocessing.image.load_img(img, target_size=(150, 150))
        x = keras.preprocessing.image.img_to_array(pred_img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = self.rps_model.predict(images, batch_size=BATCH_SIZE)
        max_index = np.argmax(classes)
        self.assertTrue(max_index >= 0 and max_index < len(self.class_names))

if __name__ == '__main__':
    unittest.main()