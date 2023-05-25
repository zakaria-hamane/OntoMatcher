import pandas as pd
from fuzzywuzzy import fuzz
from concurrent.futures import ThreadPoolExecutor
import os
import json

# Define the function to calculate fuzzy match score for all pairs
def calculate_scores(cui_strs):
    results = []
    cui, strs = cui_strs
    for i in range(len(strs)):
        for j in range(i + 1, len(strs)):
            # Ensure strings are not identical and come from different sources
            if strs[i][0] != strs[j][0] and strs[i][1] != strs[j][1]:
                score = fuzz.ratio(strs[i][0], strs[j][0])
                if score >= 90:
                    results.append({"CUI": cui, "str1": strs[i][0], "str2": strs[j][0], "score": score, "SAB1": strs[i][1], "SAB2": strs[j][1]})
    return results

# Read MRCONSO.RRF and create a dictionary of CUI to STR and its source
cui_to_str = {}
with open("extracted_data/2022AB/META/MRCONSO.RRF") as f:
    for line in f:
        fields = line.strip().split("|")
        cui, str, sab = fields[0], fields[14], fields[11]
        if cui in cui_to_str:
            cui_to_str[cui].append((str, sab))
        else:
            cui_to_str[cui] = [(str, sab)]

# Use ThreadPoolExecutor to calculate scores
results = []
with ThreadPoolExecutor(max_workers=os.cpu_count() - 4) as executor:
    for cui, strs in cui_to_str.items():
        results.extend(list(executor.submit(calculate_scores, (cui, strs)).result()))

# Convert the results list to a DataFrame
df = pd.DataFrame(results)

# Save the DataFrame to a JSON file
df.to_json("extracted_data/exact_match_pairs.json", orient="records")
