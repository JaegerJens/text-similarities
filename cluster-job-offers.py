from sentence_transformers import SentenceTransformer, util
import json


with open('job-offers.json', encoding="UTF-8") as job_offer_file:
    offers = json.load(job_offer_file)
descriptions = [job['description'] for job in offers]


model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(descriptions, convert_to_tensor=True)
cosine_scores = util.cos_sim(embeddings, embeddings)

# Find the pairs with the highest cosine similarity scores
pairs = []
for i in range(len(cosine_scores)-1):
    for j in range(i+1, len(cosine_scores)):
        pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

# Sort scores in decreasing order
pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

for pair in pairs[0:20]:
    i, j = pair['index']
    print("Score: {:.4f}:\t{} | {}".format(
        pair['score'],
        offers[i]['detailsLink'],
        offers[j]['detailsLink']
    ))
