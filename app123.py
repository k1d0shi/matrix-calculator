from flask import Flask, render_template, request, jsonify
import numpy as np
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        data = request.get_json()
        A = np.array(data.get("A"))
        B = np.array(data.get("B"))
        op = data.get("operation")

        if op == "add":
            result = (A + B).tolist()
        elif op == "sub":
            result = (A - B).tolist()
        elif op == "mul":
            result = np.dot(A, B).tolist()
        elif op == "detA":
            detA = np.linalg.det(A)
            result = [[int(detA) if abs(detA - round(detA)) < 1e-9 else round(detA, 5)]]
        elif op == "detB":
            detB = np.linalg.det(B)
            result = [[int(detB) if abs(detB - round(detB)) < 1e-9 else round(detB, 5)]]
        elif op == "transposeA":
            result = A.T.tolist()
        elif op == "transposeB":
            result = B.T.tolist()
        elif op == "rankA":
            result = [[np.linalg.matrix_rank(A)]]
        elif op == "rankB":
            result = [[np.linalg.matrix_rank(B)]]
        elif op == "invA":
            if np.linalg.det(A) == 0:
                return jsonify({"Ошибка": "Матрица A вырождена, обратной не существует"})
            result = np.linalg.inv(A).tolist()
        elif op == "invB":
            if np.linalg.det(B) == 0:
                return jsonify({"Ошибка": "Матрица B вырождена, обратной не существует"})
            result = np.linalg.inv(B).tolist()

        else:
            return jsonify({"error": "Неизвестная операция"})

        for i in range(len(result)):
            if isinstance(result[i], list):
                result[i] = [int(x) if abs(x - round(x)) < 1e-9 else round(x, 3) for x in result[i]]
            else:
                result[i] = int(result[i]) if abs(result[i] - round(result[i])) < 1e-9 else round(result[i], 3)

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

