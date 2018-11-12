from flask import Flask, request, jsonify
from flask_restful import Resource, Api

from Detect import spamDetector
from SuccessPrediction import classifierForComments
from SuccessPrediction import classifierForReactions

app = Flask(__name__)
api = Api(app)

class Spam_Detection(Resource):
    def get(self):
        return spamDetector.detector("Hey, how was your dinner?")
    def post(self):
        incoming_json = request.json['data']
        response = spamDetector.detector(incoming_json)
        data = {}
        data['comments'] = response
        return jsonify(data)

class Reactions_Success_Prediction(Resource):
    def post(self):
        incoming_json = request.json['data']
        response = classifierForReactions.classifier(incoming_json)
        print int(response)
        success = {}
        success['success'] = response
        return jsonify(success)

class Emotions_Success_Prediction(Resource):
    def post(self):
        incoming_json = request.json['data']
        response = classifierForComments.detector(incoming_json)
        success = {}
        success['success'] = response
        return jsonify(success)

api.add_resource(Spam_Detection, '/spam-detection')
api.add_resource(Reactions_Success_Prediction, '/reactions-success-prediction')
api.add_resource(Emotions_Success_Prediction, '/emotions-success-prediction')

if __name__ == '__main__':
     app.run()