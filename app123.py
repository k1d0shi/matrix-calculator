from flask import Flask, render_template, request, jsonify
import numpy as np
import os

app = Flask(__name__)


def fmt_num(x, ndigits=6):
    """Формат числа: если целое — как int, иначе округлить."""
    try:
        if abs(x - round(x)) < 1e-9:
            return int(round(x))
        else:
            return round(float(x), ndigits)
    except Exception:
        return x


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        data = request.get_json()
        A = np.array(data.get("A", []), dtype=float)
        B = np.array(data.get("B", []), dtype=float)
        op = data.get("operation", "")
        steps = []
        result = None

        # --- Проверки размеров и формы ---
        if A.size == 0:
            raise ValueError("Матрица A пуста или некорректна.")
        if B.size == 0:
            raise ValueError("Матрица B пуста или некорректна.")

        # --- Сложение ---
        if op == "add":
            steps.append("Операция: сложение матриц A + B.")
            if A.shape != B.shape:
                raise ValueError("Матрицы должны быть одинакового размера для сложения.")
            rows, cols = A.shape
            res = np.zeros((rows, cols))
            for i in range(rows):
                for j in range(cols):
                    aij = A[i, j]
                    bij = B[i, j]
                    sij = aij + bij
                    steps.append(f"Шаг [{i},{j}]: {fmt_num(aij)} + {fmt_num(bij)} = {fmt_num(sij)}")
                    res[i, j] = sij
            result = res.tolist()

        # --- Вычитание ---
        elif op == "sub":
            steps.append("Операция: вычитание матриц A - B.")
            if A.shape != B.shape:
                raise ValueError("Матрицы должны быть одинакового размера для вычитания.")
            rows, cols = A.shape
            res = np.zeros((rows, cols))
            for i in range(rows):
                for j in range(cols):
                    aij = A[i, j]
                    bij = B[i, j]
                    dij = aij - bij
                    steps.append(f"Шаг [{i},{j}]: {fmt_num(aij)} - {fmt_num(bij)} = {fmt_num(dij)}")
                    res[i, j] = dij
            result = res.tolist()

        # --- Умножение ---
        elif op == "mul":
            steps.append("Операция: умножение матриц A × B.")
            if A.shape[1] != B.shape[0]:
                raise ValueError("Число столбцов A должно равняться числу строк B для умножения.")
            rows = A.shape[0]
            cols = B.shape[1]
            inner = A.shape[1]
            res = np.zeros((rows, cols))
            for i in range(rows):
                for j in range(cols):
                    terms = []
                    s = 0.0
                    for k in range(inner):
                        ak = A[i, k]
                        bk = B[k, j]
                        terms.append(f"{fmt_num(ak)}×{fmt_num(bk)}")
                        s += ak * bk
                    steps.append(f"Шаг [{i},{j}]: " + " + ".join(terms) + f" = {fmt_num(s)}")
                    res[i, j] = s
            result = res.tolist()

        # --- Транспонирование A ---
        elif op == "transposeA":
            steps.append("Операция: транспонирование матрицы A (Aᵀ).")
            steps.append(f"A =\n{A.tolist()}")
            T = A.T
            steps.append(f"Aᵀ =\n{T.tolist()}")
            result = T.tolist()

        # --- Транспонирование B ---
        elif op == "transposeB":
            steps.append("Операция: транспонирование матрицы B (Bᵀ).")
            steps.append(f"B =\n{B.tolist()}")
            T = B.T
            steps.append(f"Bᵀ =\n{T.tolist()}")
            result = T.tolist()

        # --- Ранг A ---
        elif op == "rankA":
            steps.append("Операция: вычисление ранга матрицы A.")
            r = int(np.linalg.matrix_rank(A))
            steps.append(f"Ранг(A) = {r}")
            result = [[r]]

        # --- Ранг B ---
        elif op == "rankB":
            steps.append("Операция: вычисление ранга матрицы B.")
            r = int(np.linalg.matrix_rank(B))
            steps.append(f"Ранг(B) = {r}")
            result = [[r]]

        # --- Определитель A ---
        elif op == "detA":
            steps.append("Операция: вычисление определителя det(A).")
            if A.shape[0] != A.shape[1]:
                raise ValueError("Матрица A должна быть квадратной для вычисления определителя.")
            detA = float(np.linalg.det(A))
            steps.append(f"det(A) = {fmt_num(detA, ndigits=6)}")
            result = [[fmt_num(detA, ndigits=6)]]

        # --- Определитель B ---
        elif op == "detB":
            steps.append("Операция: вычисление определителя det(B).")
            if B.shape[0] != B.shape[1]:
                raise ValueError("Матрица B должна быть квадратной для вычисления определителя.")
            detB = float(np.linalg.det(B))
            steps.append(f"det(B) = {fmt_num(detB, ndigits=6)}")
            result = [[fmt_num(detB, ndigits=6)]]

        # --- Обратная матрица A ---
        elif op == "invA":
            steps.append("Операция: нахождение обратной матрицы A⁻¹.")
            if A.shape[0] != A.shape[1]:
                raise ValueError("Матрица A должна быть квадратной для обратной матрицы.")
            detA = float(np.linalg.det(A))
            steps.append(f"Сначала вычисляем det(A) = {fmt_num(detA, ndigits=6)}")
            if abs(detA) < 1e-12:
                raise ValueError("Матрица A вырождена — обратная матрица не существует.")
            invA = np.linalg.inv(A)
            steps.append("Далее вычисляем A⁻¹ (с использованием метода обратной матрицы / библиотечной функции).")
            steps.append(f"A⁻¹ =\n{invA.tolist()}")
            result = invA.tolist()

        # --- Обратная матрица B ---
        elif op == "invB":
            steps.append("Операция: нахождение обратной матрицы B⁻¹.")
            if B.shape[0] != B.shape[1]:
                raise ValueError("Матрица B должна быть квадратной для обратной матрицы.")
            detB = float(np.linalg.det(B))
            steps.append(f"Сначала вычисляем det(B) = {fmt_num(detB, ndigits=6)}")
            if abs(detB) < 1e-12:
                raise ValueError("Матрица B вырождена — обратная матрица не существует.")
            invB = np.linalg.inv(B)
            steps.append("Далее вычисляем B⁻¹ (с использованием метода обратной матрицы / библиотечной функции).")
            steps.append(f"B⁻¹ =\n{invB.tolist()}")
            result = invB.tolist()

        else:
            return jsonify({"error": "Неизвестная операция"}), 400

        # --- Форматирование чисел результата: целые как int, дробные — округлить ---
        # Если результат — скаляр или 1x1 — привести к [[value]] чтобы фронтенд работал единообразно
        if isinstance(result, (int, float)):
            result = [[fmt_num(result)]]
        # Убедимся, что результат — двумерный список (матрица) или список списков
        if isinstance(result, list):
            # Если это список чисел (1D), преобразуем в [[...]]
            if len(result) > 0 and not isinstance(result[0], list):
                result = [result]
            # Округляем элементы матрицы
            for i in range(len(result)):
                for j in range(len(result[i])):
                    try:
                        result[i][j] = fmt_num(result[i][j], ndigits=6)
                    except Exception:
                        pass

        return jsonify({"result": result, "steps": steps})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
