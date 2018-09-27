from flask import Flask, request
from flask_restful import Resource, Api
import spamDetector
# import json
from flask import jsonify
# from SuccessPrediction import modelTrainer

app = Flask(__name__)
api = Api(app)

class Spam_Detection(Resource):
    def get(self):
        return spamDetector.detector("Hey how was your dinner?")
    def post(self):
        print jsonify(request)
        # d = json.load(data)
        # print d
        # incoming_json = request.json['data']
        # print incoming_json[0]['message']
        # response = spamDetector.detector(incoming_json)
        return "success"

api.add_resource(Spam_Detection, '/spam-detection')

if __name__ == '__main__':
     app.run()