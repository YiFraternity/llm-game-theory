import re
import json

file_path = r"C:\miaosiyu\桌面\py\merged_ranks.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    item['resps'] = [re.sub(r'\D', '', resp) for resp in item['resps']]

print(json.dumps(data, indent=4, ensure_ascii=False))