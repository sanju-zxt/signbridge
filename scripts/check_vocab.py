import json

with open('ml/reports/asl_v4_44sign_evaluation.json') as f:
    d = json.load(f)

labels = d['confusion_matrix_44x44']['labels']
matrix = d['confusion_matrix_44x44']['matrix']

for w in ['BAD', 'MAYBE', 'LIKE', 'AGAIN', 'FINISHED']:
    row_idx = labels.index(w)
    row = matrix[row_idx]
    preds = [(labels[j], count) for j, count in enumerate(row) if count > 0]
    print(f"{w} predicted within 44 as: {preds}")





