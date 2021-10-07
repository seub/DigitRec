from PIL import Image
import numpy as np
import pickle
from collections import defaultdict


class ImageClassifier:
    def __init__(self, databaseName):
        self._database = None
        self._databaseName = databaseName

    def __str__(self):
        if not self._database: # Never test object == None
            return ("ImageRegognizer with database of" + str(len(self._database)) + " different Classes, each containing " + str(len(self._database[0])) + " images.")
        
    @staticmethod
    def createDatabase(imageFolder, databaseName):
        number_db = defaultdict(list)
        for digit in range(10):
            for index in range(16): # Change 16 to the number of images you have per category
                image = Image.open(imageFolder + "/" + str(digit) + "_" + str(index) + ".jpg")
                number_db[digit].append(np.array(image).tolist())
        with open(databaseName + ".pkl", "wb") as db:
            pickle.dump(number_db,db)
        
    def openDatabase(self):
        with open(self._databaseName+".pkl", "rb") as db:
            self._database = pickle.load(db)
       
    @staticmethod
    def normalizeBinary(image):
        for column in image:
            for pixel in range(len(column)):
                total_color = 0
                for color in column[pixel]:
                    total_color += color
                if total_color/3 < 255/2:
                    column[pixel] = 0
                else:
                    column[pixel] = 1
        return image
        

    def normalizeDatabase(self, normFunction):
        for number in range(len(self._database.keys())):
            for image in self._database[number]:
                image = normFunction(image)

        
    def classifyImage(self, img, normFunction):
        test_image = Image.open(img)
        test_image = np.array(test_image).tolist()
        test_image = normFunction(test_image)
        confidence_dict = {}
        for i in range(10):
            confidence_dict[i] = 0
        for number in range(len(self._database.keys())):
            for image in self._database[number]:
                for column in range(len(image)):
                    for pixel in range(len(image[column])):
                        if image[column][pixel] == test_image[column][pixel] == 1:
                            confidence_dict[number] += 2
                        if image[column][pixel] == test_image[column][pixel] == 0:
                            continue
                        confidence_dict[number] -= 1
        max_confidence = sorted(confidence_dict.values(), reverse=True)
        for key, value in confidence_dict.items():
            if value == max_confidence[0]:
                return key, min(round(((1.0/max_confidence[0])*max_confidence[1])*100,2), 100.00)

if __name__ == "__main__":
    ImageClassifier.createDatabase("images", "number_db")
    imageClassifier = ImageClassifier("number_db")
    imageClassifier.openDatabase()
    print(imageClassifier)
    imageClassifier.normalizeDatabase(ImageClassifier.normalizeBinary)
    for number in range(10):
        print(imageClassifier.classifyImage("test" + str(number) +".jpg", ImageClassifier.normalizeBinary))