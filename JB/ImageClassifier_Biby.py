from collections import defaultdict
import os
import numpy as np
import pickle
from PIL import Image
from typing import Callable


class ImageClassifier:
    def __init__(self, database=None) -> None:
        self._database = database

    def __str__(self) -> str:
        if self._database is not None:
            return f"ImageRegognizer with database of {len(self._database)} different Classes, each containing {len(self._database[0])} images."
        else:
            return "Empty database!"

    @classmethod
    def create_database(cls, imageFolder: str) -> None:
        """
        Create new ImageClassifier, populating the database out of images contained in imageFolder
        Images in folder have to follow defined scheme: letter_number.jpg, where letter describes what the image displays (letter 0, letter 1 etc) and number is an incrementing number starting from 0
        """
        database = defaultdict(list)
        for number in range(10):
            for index in range(
                16
            ):  # Change 16 to the number of images you have per category
                image = Image.open(os.path.join(imageFolder, f"{number}_{index}.jpg"))
                database[number].append(np.array(image))
        return cls(database)

    def save_database(self, database_name: str) -> None:
        """
        Save data base to file
        """
        with open(f"{database_name}.pkl", "wb") as db:
            pickle.dump(
                self._database, db
            )  # Probably not working as I don't know if pickle knows how to handle numpy arrays

    def load_database(self, database_name: str) -> None:
        """
        Load database from file
        """
        with open(f"{database_name}.pkl", "rb") as db:
            self._database = pickle.load(db)

    @staticmethod
    def normalize_binary(image: np.array) -> np.array:
        """
        Simple normalize method that normalizes an image to binary according to static pixel value <> 255/2
        """
        return np.apply_along_axis(lambda x: round(np.sum(x) / (3 * 255)), 2, image)

    @staticmethod
    def normalize_not(image: np.array) -> np.array:
        """
        Use this method if you do not want to normalize an image or database and use raw fotage
        """
        return image

    def normalize_database(self, norm_function: Callable[[np.array], np.array]) -> None:
        """
        applies normFunction to every image in Database
        """
        for number, images in self._database.items():
            self._database[number] = [norm_function(image) for image in images]

    def classify_image(
        self, image_path: str, norm_function: Callable[[np.array], np.array]
    ) -> None:
        """
        Classifies image according to database images, apply same normalize function as you applies previously on the whole database
        returns tuple of found image class and certainty as a value between 50 and 100, 50 meaning totaly uncertantiy between two possible classes, 100 meaning total certainty
        """
        test_image = Image.open(image_path)
        test_image = np.array(test_image)
        test_image = norm_function(test_image)
        confidence_dict = defaultdict(lambda: 0)
        for digit, digit_images in self._database.items():
            for image in digit_images:
                value, count = np.unique(
                    np.add(digit_images, test_image).flatten(), return_counts=True
                )
                value_count = dict(zip(value, count))
                confidence_dict[digit] = value_count[2] - value_count[1]
        max_confidence = sorted(confidence_dict.values(), reverse=True)
        max_key = max(confidence_dict, key=confidence_dict.get)
        return max_key, min(
            round(((1.0 / max_confidence[0]) * max_confidence[1]) * 100, 2), 100.00
        )


# Use for testing ImageRecognizer Class
if __name__ == "__main__":
    image_classifier = ImageClassifier.create_database(imageFolder="images")
    print(image_classifier)
    image_classifier.normalize_database(ImageClassifier.normalize_binary)
    for number in range(10):
        print(
            image_classifier.classify_image(
                f"test{number}.jpg", ImageClassifier.normalize_binary
            )
        )
