#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
from model_deployment import predict

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Movie Genres Prediction',
    description='Deep Learning model that predicts the genres of a movie')

ns = api.namespace('predict', 
     description='Genres Prediction')
   
parser = api.parser()

parser.add_argument(
    'plot', 
    type=str, 
    required=True, 
    help='Movie Plot', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class GenresApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
    
        return {
         "result": predict([args.plot])
        }, 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000, threaded=False)
