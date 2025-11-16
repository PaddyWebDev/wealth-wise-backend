from sklearn.metrics.pairwise import cosine_similarity
import joblib
from flask import Flask, request, make_response
from flask_cors import CORS
from operator import itemgetter
from openpyxl import load_workbook
from flask import Flask, request, make_response
from model.cosine_similarity import recommend_funds
from operator import itemgetter


app = Flask(__name__)
CORS(app)


@app.route('/', methods=["GET"])
def home():
    return make_response("Backend for Recipe Recommendation System", 200)


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        if (not request):
            return make_response({
                "Error": {
                    "code": 400,
                    "message": "Bad request"
                }
            }, 400)


        results = recommend_funds(data.get("expected_return"), data.get("risk_level"), data.get("category"), data.get("min_sip"))
        print(results)
        # print(results)
        return "hi"

        # return jsonify(results)

    except Exception as e:
        print(e)
        return make_response({"Error": {"code": 500, "message": str(e)}}, 500)


if __name__ == '__main__':
    app.run(debug=True)
