from sklearn.metrics.pairwise import cosine_similarity
import joblib
from flask import Flask, request, make_response, jsonify
from flask_cors import CORS
from flask import Flask, request, make_response


app = Flask(__name__)
CORS(app)

cv = joblib.load("model/vectorizer.pkl")
vectorMatrix = joblib.load("model/vector_matrix.pkl")
df = joblib.load("model/dataFrame.pkl")


@app.route('/', methods=["GET"])
def home():
    return make_response("Backend for Recipe Recommendation System", 200)


def recommend_by_preferences(min_sip: str, risk_level: str, category: str, expected_return: str):
    user_input = ' '.join([
        min_sip,
        risk_level,
        category.lower(),
        expected_return,
        expected_return,
        expected_return
    ])

    user_vector = cv.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, vectorMatrix).flatten()

    # Get top 5 similar recipe indices
    top_indices = similarity_scores.argsort()[-5:][::-1]

    recommendations = []
    for idx in top_indices:
        recipe = {
            "idx": int(idx),  # Cast to int for safe JSON serialization
            "title": df.iloc[idx].to_dict() 
        }
        recommendations.append(recipe)

    return recommendations


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        # force=True ensures JSON is parsed
        data = request.get_json(force=True)

        if not data:
            return make_response({
                "error": {
                    "code": 400,
                    "message": "Bad request – No JSON received"
                }
            }, 400)



        results = recommend_by_preferences(
            min_sip=str(data.get("min_sip")),
            risk_level=str(data.get("risk_level")),
            category=str(data.get("category")),
            expected_return=str(data.get("expected_return")),
        )

        return jsonify(results), 200

    except Exception as e:
        print("Error:", e)
        return make_response({"error": str(e)}, 500)


if __name__ == '__main__':
    app.run(debug=True)
