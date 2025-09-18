import json

test_data = json.load(open("data/test_data.json", "r"))

results = json.load(open("results.json", "r"))

evaluation_data = []

for item in test_data:
    query = item["query"]
    answer = item["answer"]
    page = item["page"]

    for result in results:
        if result["query"] == query:
            result_answer = result["response"]
            evaluation_data.append({
                "query": query,
                "answer": answer,
                "page": page,
                "result": result_answer,
            })
            break

json.dump(evaluation_data, open("evaluation_data.json", "w"), indent="  ", ensure_ascii=False)
