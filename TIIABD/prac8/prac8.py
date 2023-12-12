import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Specify the backend
import time

df = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = [df.iloc[i].dropna().tolist() for i in range(df.shape[0])]

# Визуализация данных для 20 наиболее популярных товаров
top_20_products = df[0].value_counts().head(20)
plt.figure(figsize=(10, 6))
top_20_products.plot(kind='bar', title='Топ 20')
plt.show()

transactions = [df.iloc[i].dropna().tolist() for i in range(df.shape[0])]

from apriori_python import apriori

# Load the dataset
df = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = [df.iloc[i].dropna().tolist() for i in range(df.shape[0])]

start_py = time.perf_counter()
result_py = apriori(transactions, minSup=0.04, minConf=0.15)[1]
end_py = time.perf_counter()
time_py = end_py - start_py
print("Total rules:", len(result_py), "\n")
output_py = []
for rule in result_py:
    output_py.append([f"{', '.join(rule[0])} & {', '.join(rule[1])}", rule[2]])
pd.DataFrame(output_py, columns=["Rule", "Support"]).sort_values(by="Support", ascending=False)

from apyori import apriori

# import time


# Load the dataset
df = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

transactions = [df.iloc[i].dropna().tolist() for i in range(df.shape[0])]

start_apyori = time.perf_counter()
result_apyori = list(apriori(transactions, min_support=0.035,
                             min_confidence=0.22))
end_apyori = time.perf_counter()
time_apyori = end_apyori - start_apyori
print("Total rules:", len(result_apyori), "\n")
output_apyori = []
for rule in result_apyori:
    if len(rule.items) > 1:
        output_apyori.append([" & ".join(rule.items), rule.support])

pd.DataFrame(output_apyori, columns=["Rule", "Support"]).sort_values(by="Support", ascending=False)

from efficient_apriori import apriori

# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib
# import time


# Load the dataset
df = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

transactions = [df.iloc[i].dropna().tolist() for i in range(df.shape[0])]

start_efficient = time.perf_counter()
itemsets_efficient, rules_efficient = apriori(transactions, min_support=0.035,
                                              min_confidence=0.23)
end_efficient = time.perf_counter()
time_efficient = end_efficient - start_efficient
print("Total rules:", len(rules_efficient), "\n")
output_efficient = []
for rule in rules_efficient:
    output_efficient.append(
        [
            f"{rule.lhs[0]} & {rule.rhs[0]}",
            rule.support,
            rule.confidence,
            rule.conviction,

        ]
    )

pd.DataFrame(
    output_efficient, columns=["Rule", "Support", "Confidence", "Conviction"]
).sort_values(by="Support", ascending=False)

from fpgrowth_py import fpgrowth

# Load the dataset
df = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

transactions = [df.iloc[i].dropna().tolist() for i in range(df.shape[0])]

start_fpgrowth = time.perf_counter()
itemsets_fpgrowth, rules_fpgrowth = fpgrowth(
    transactions, minSupRatio=0.035, minConf=0.23
)

end_fpgrowth = time.perf_counter()
time_fpgrowth = end_fpgrowth - start_fpgrowth
print("Total rules:", len(rules_fpgrowth), "\n")
output_fpgrowth = []
for rule in rules_fpgrowth:
    output_fpgrowth.append(
        [
            f"{''.join(rule[0])} & {''.join(rule[1])}", rule[2],
        ]
    )

pd.DataFrame(output_fpgrowth, columns=["Rule", "Support"]).sort_values(
    by="Support", ascending=False)

import matplotlib.pyplot as plt

print("Execution time of APRIORI_PYTHON:", time_py)
print("Execution time of APYORI:", time_apyori)
print("Execution time of EFFICIENT:", time_efficient)
print("Execution time of FPGROWTH:", time_fpgrowth)
plt.bar(
    ["APRIORI_PYTHON", "APYORI", "EFFICIENT", "FPGROWTH"],
    [time_py, time_apyori, time_efficient, time_fpgrowth],
)
plt.show()

data = pd.read_csv('data.csv', header=None)

# Визуализация данных для 20 наиболее популярных товаров
top_20_products = data[0].value_counts().head(20)
plt.figure(figsize=(10, 6))
top_20_products.plot(kind='bar', title='Топ 20')
plt.show()

transactions = [data.iloc[i].dropna().tolist() for i in range(data.shape[0])]

data = pd.read_csv('data.csv', header=None)
transactions = [data.iloc[i].dropna().tolist() for i in range(data.shape[0])]

start_py = time.perf_counter()
result_py = apriori(transactions, minSup=0.04, minConf=0.15)[1]
end_py = time.perf_counter()
time_py = end_py - start_py
print("Total rules:", len(result_py), "\n")
output_py = []
for rule in result_py:
    output_py.append([f"{', '.join(rule[0])} & {', '.join(rule[1])}", rule[2]])
pd.DataFrame(output_py, columns=["Rule", "Support"]).sort_values(by="Support", ascending=False)

df = pd.read_csv('data.csv', header=None)

transactions = [df.iloc[i].dropna().tolist() for i in range(df.shape[0])]

start_apyori = time.perf_counter()
result_apyori = list(apriori(transactions, min_support=0.035,
                             min_confidence=0.22))
end_apyori = time.perf_counter()
time_apyori = end_apyori - start_apyori
print("Total rules:", len(result_apyori), "\n")
output_apyori = []
for rule in result_apyori:
    if len(rule.items) > 1:
        output_apyori.append([" & ".join(rule.items), rule.support])

pd.DataFrame(output_apyori, columns=["Rule", "Support"]).sort_values(by="Support", ascending=False)

df = pd.read_csv('data.csv', header=None)

transactions = [df.iloc[i].dropna().tolist() for i in range(df.shape[0])]

start_efficient = time.perf_counter()
itemsets_efficient, rules_efficient = apriori(transactions, min_support=0.035,
                                              min_confidence=0.23)
end_efficient = time.perf_counter()
time_efficient = end_efficient - start_efficient
print("Total rules:", len(rules_efficient), "\n")
output_efficient = []
for rule in rules_efficient:
    output_efficient.append(
        [
            f"{rule.lhs[0]} & {rule.rhs[0]}",
            rule.support,
            rule.confidence,
            rule.conviction,

        ]
    )

pd.DataFrame(
    output_efficient, columns=["Rule", "Support", "Confidence", "Conviction"]
).sort_values(by="Support", ascending=False)

import matplotlib.pyplot as plt

print("Execution time of APRIORI_PYTHON:", time_py)
print("Execution time of APYORI:", time_apyori)
print("Execution time of EFFICIENT:", time_efficient)
print("Execution time of FPGROWTH:", time_fpgrowth)
plt.bar(
    ["APRIORI_PYTHON", "APYORI", "EFFICIENT", "FPGROWTH"],
    [time_py, time_apyori, time_efficient, time_fpgrowth],
)
plt.show()
