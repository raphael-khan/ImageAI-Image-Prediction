from imageai.Classification import ImageClassification
import os

execution_path = os.getcwd()

prediction = ImageClassification()
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath(os.path.join(execution_path, 'mobilenet_v2.h5'))
prediction.loadModel()

predictions, probabilities = prediction.predictImage(os.path.join(execution_path, 'godzilla.jpg'), result_count=5)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)
