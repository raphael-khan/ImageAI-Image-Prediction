from imageai.Prediction import ImagePrediction
import os

execution_path = os.getcwd()

prediction = ImagePredcition()
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath(os.path.join(execution_path, 'MobileNetV2'))
prediction.loadModel()

predictions, probabilities = prediction.predictImage(os.path.join(execution_path, 'godzilla.jpg'), result_count=5)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)
