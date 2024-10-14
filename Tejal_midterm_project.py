#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import random

# List of items for Amazon, Best Buy, and K-Mart
amazon_items = [
    "A Beginner’s Guide", "Java: The Complete Reference", "Java For Dummies",
    "Android Programming: The Big Nerd Ranch", "Head First Java 2nd Edition",
    "Beginning Programming with Java", "Java 8 Pocket Guide",
    "C++ Programming in Easy Steps", "Effective Java (2nd Edition)",
    "HTML and CSS: Design and Build Websites"
]

bestbuy_items = [
    "Digital Camera", "Laptop", "Desktop", "Printer",
    "Flash Drive", "Microsoft Office", "Speakers",
    "Laptop Case", "Anti-Virus", "External Hard Drive"
]

kmart_items = [
    "Quilts", "Bedspreads", "Decorative Pillows",
    "Bed Skirts", "Sheets", "Shams",
    "Bedding Collections", "Kids Bedding",
    "Embroidered Bedspread", "Towels"
]

# Create transactions for Amazon, Best Buy, and K-Mart
def create_transactions(store_items, store_name):
    transactions = []
    for i in range(1, 21):
        num_items = random.randint(2, 6)  # Each transaction will have 2 to 6 items
        transaction_items = random.sample(store_items, num_items)
        transactions.append({
            "Transaction ID": f"{store_name}Trans{i}",
            "Items": ", ".join(transaction_items)
        })
    return transactions

# Compile all transactions
all_transactions = []
stores = {
    "Amazon": amazon_items,
    "Best Buy": bestbuy_items,
    "K-Mart": kmart_items
}

for store, items in stores.items():
    all_transactions.extend(create_transactions(items, store))

# Convert to DataFrame
df_transactions = pd.DataFrame(all_transactions)

# Save to CSV
df_transactions.to_csv("store_transactions.csv", index=False)

print("Transactions saved to 'store_transactions.csv'.")


# In[2]:


import pandas as pd

# Define the Amazon items
amazon_items = [
    "A Beginner’s Guide",
    "Java: The Complete Reference",
    "Java For Dummies",
    "Android Programming: The Big Nerd Ranch",
    "Head First Java 2nd Edition",
    "Beginning Programming with Java",
    "Java 8 Pocket Guide",
    "C++ Programming in Easy Steps",
    "Effective Java (2nd Edition)",
    "HTML and CSS: Design and Build Websites"
]

# Define the Best Buy items
bestbuy_items = [
    "Digital Camera",
    "Lab Top",
    "Desk Top",
    "Printer",
    "Flash Drive",
    "Microsoft Office",
    "Speakers",
    "Lab Top Case",
    "Anti-Virus",
    "External Hard-Drive"
]

# Define the K-Mart items
kmart_items = [
    "Quilts",
    "Bedspreads",
    "Decorative Pillows",
    "Bed Skirts",
    "Sheets",
    "Shams",
    "Bedding Collections",
    "Kids Bedding",
    "Embroidered Bedspread",
    "Towels"
]

# Define transactions for Amazon
transactions_amazon = [
    ["A Beginner’s Guide", "Java: The Complete Reference", "Java For Dummies", "Android Programming: The Big Nerd Ranch"],
    ["A Beginner’s Guide", "Java: The Complete Reference", "Java For Dummies"],
    ["A Beginner’s Guide", "Java: The Complete Reference", "Java For Dummies", "Android Programming: The Big Nerd Ranch", "Head First Java 2nd Edition"],
    ["Android Programming: The Big Nerd Ranch", "Head First Java 2nd Edition", "Beginning Programming with Java"],
    ["Android Programming: The Big Nerd Ranch", "Beginning Programming with Java", "Java 8 Pocket Guide"],
    ["A Beginner’s Guide", "Android Programming: The Big Nerd Ranch", "Head First Java 2nd Edition"],
    ["A Beginner’s Guide", "Head First Java 2nd Edition", "Beginning Programming with Java"],
    ["Java: The Complete Reference", "Java For Dummies", "Android Programming: The Big Nerd Ranch"],
    ["Java For Dummies", "Android Programming: The Big Nerd Ranch", "Head First Java 2nd Edition", "Beginning Programming with Java"],
    ["Beginning Programming with Java", "Java 8 Pocket Guide", "C++ Programming in Easy Steps"],
    ["A Beginner’s Guide", "Java: The Complete Reference", "Java For Dummies", "Android Programming: The Big Nerd Ranch"],
    ["A Beginner’s Guide", "Java: The Complete Reference", "Java For Dummies", "HTML and CSS: Design and Build Websites"],
    ["A Beginner’s Guide", "Java: The Complete Reference", "Java For Dummies", "Java 8 Pocket Guide", "HTML and CSS: Design and Build Websites"],
    ["Java For Dummies", "Android Programming: The Big Nerd Ranch", "Head First Java 2nd Edition"],
    ["Java For Dummies", "Android Programming: The Big Nerd Ranch"],
    ["A Beginner’s Guide", "Java: The Complete Reference", "Java For Dummies", "Android Programming: The Big Nerd Ranch"],
    ["A Beginner’s Guide", "Java: The Complete Reference", "Java For Dummies", "Android Programming: The Big Nerd Ranch"],
    ["Head First Java 2nd Edition", "Beginning Programming with Java", "Java 8 Pocket Guide"],
    ["Android Programming: The Big Nerd Ranch", "Head First Java 2nd Edition"],
    ["A Beginner’s Guide", "Java: The Complete Reference", "Java For Dummies"]
]

# Define transactions for Best Buy
transactions_bestbuy = [
    ["Desk Top", "Printer", "Flash Drive", "Microsoft Office", "Speakers", "Anti-Virus"],
    ["Lab Top", "Flash Drive", "Microsoft Office", "Lab Top Case", "Anti-Virus"],
    ["Lab Top", "Printer", "Flash Drive", "Microsoft Office", "Anti-Virus", "Lab Top Case", "External Hard-Drive"],
    ["Lab Top", "Printer", "Flash Drive", "Anti-Virus", "External Hard-Drive", "Lab Top Case"],
    ["Lab Top", "Flash Drive", "Lab Top Case", "Anti-Virus"],
    ["Lab Top", "Printer", "Flash Drive", "Microsoft Office"],
    ["Desk Top", "Printer", "Flash Drive", "Microsoft Office"],
    ["Lab Top", "External Hard-Drive", "Anti-Virus"],
    ["Desk Top", "Printer", "Flash Drive", "Microsoft Office", "Lab Top Case", "Anti-Virus", "Speakers", "External Hard-Drive"],
    ["Digital Camera", "Lab Top", "Desk Top", "Printer", "Flash Drive", "Microsoft Office", "Lab Top Case", "Anti-Virus", "External Hard-Drive", "Speakers"],
    ["Lab Top", "Desk Top", "Lab Top Case", "External Hard-Drive", "Speakers", "Anti-Virus"],
    ["Digital Camera", "Lab Top", "Lab Top Case", "External Hard-Drive", "Anti-Virus", "Speakers"],
    ["Digital Camera", "Speakers"],
    ["Digital Camera", "Desk Top", "Printer", "Flash Drive", "Microsoft Office"],
    ["Printer", "Flash Drive", "Microsoft Office", "Anti-Virus", "Lab Top Case", "Speakers", "External Hard-Drive"],
    ["Digital Camera", "Flash Drive", "Microsoft Office", "Anti-Virus", "Lab Top Case", "External Hard-Drive", "Speakers"],
    ["Digital Camera", "Lab Top", "Lab Top Case"],
    ["Digital Camera", "Lab Top Case", "Speakers"],
    ["Digital Camera", "Lab Top", "Printer", "Flash Drive", "Microsoft Office", "Speakers", "Lab Top Case", "Anti-Virus"],
    ["Digital Camera", "Lab Top", "Speakers", "Anti-Virus", "Lab Top Case"]
]

# Define transactions for K-Mart
transactions_kmart = [
    ["Decorative Pillows", "Quilts", "Embroidered Bedspread"],
    ["Embroidered Bedspread", "Shams", "Kids Bedding", "Bedding Collections", "Bed Skirts", "Bedspreads", "Sheets"],
    ["Decorative Pillows", "Quilts", "Embroidered Bedspread", "Shams", "Kids Bedding", "Bedding Collections"],
    ["Kids Bedding", "Bedding Collections", "Sheets", "Bedspreads", "Bed Skirts"],
    ["Decorative Pillows", "Kids Bedding", "Bedding Collections", "Sheets", "Bed Skirts", "Bedspreads"],
    ["Bedding Collections", "Bedspreads", "Bed Skirts", "Sheets", "Shams", "Kids Bedding"],
    ["Decorative Pillows", "Quilts"],
    ["Decorative Pillows", "Quilts", "Embroidered Bedspread"],
    ["Bedspreads", "Bed Skirts", "Shams", "Kids Bedding", "Sheets"],
    ["Quilts", "Embroidered Bedspread", "Bedding Collections"],
    ["Bedding Collections", "Bedspreads", "Bed Skirts", "Kids Bedding", "Shams", "Sheets"],
    ["Decorative Pillows", "Quilts"],
    ["Embroidered Bedspread", "Shams"],
    ["Sheets", "Shams", "Bed Skirts", "Kids Bedding"],
    ["Decorative Pillows", "Quilts"],
    ["Decorative Pillows", "Kids Bedding", "Bed Skirts", "Shams"],
    ["Decorative Pillows", "Shams", "Bed Skirts"],
    ["Quilts", "Sheets", "Kids Bedding"],
    ["Shams", "Bed Skirts", "Kids Bedding", "Sheets"],
    ["Decorative Pillows", "Bedspreads", "Shams", "Sheets", "Bed Skirts", "Kids Bedding"]
]

# Function to save transactions to CSV for each store
def save_transactions(store_name, transactions, db_number):
    df_transactions = pd.DataFrame({
        "Transaction ID": [f"{store_name}Trans{j+1} DB{db_number}" for j in range(len(transactions))],
        "Items": [", ".join(transaction) for transaction in transactions]
    })
    df_transactions.to_csv(f"{store_name.lower().replace(' ', '_')}_transactions_db{db_number}.csv", index=False)

# Save Amazon transactions
save_transactions("Amazon", transactions_amazon, 1)

# Save Best Buy transactions
save_transactions("Best Buy", transactions_bestbuy, 2)

# Save K-Mart transactions
save_transactions("K-Mart", transactions_kmart, 3)

print("All store databases saved successfully!")


# In[3]:


import pandas as pd
from itertools import combinations
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth
import time

# Define items for Amazon, Best Buy, and K-Mart
amazon_items = [
    "A Beginner’s Guide", "Java: The Complete Reference", "Java For Dummies",
    "Android Programming: The Big Nerd Ranch", "Head First Java 2nd Edition",
    "Beginning Programming with Java", "Java 8 Pocket Guide",
    "C++ Programming in Easy Steps", "Effective Java (2nd Edition)",
    "HTML and CSS: Design and Build Websites"
]

bestbuy_items = [
    "Digital Camera", "Lab Top", "Desk Top", "Printer",
    "Flash Drive", "Microsoft Office", "Speakers",
    "Lab Top Case", "Anti-Virus", "External Hard Drive"
]

kmart_items = [
    "Quilts", "Bedspreads", "Decorative Pillows", "Bed Skirts",
    "Sheets", "Shams", "Bedding Collections",
    "Kids Bedding", "Embroidered Bedspread", "Towels"
]

# Transactions for Amazon, Best Buy, and K-Mart
transactions_db = {
    'Amazon': [
        ["A Beginner’s Guide", "Java: The Complete Reference", "Java For Dummies", "Android Programming: The Big Nerd Ranch"],
        ["A Beginner’s Guide", "Java: The Complete Reference", "Java For Dummies"],
        ["A Beginner’s Guide", "Java: The Complete Reference", "Java For Dummies", "Android Programming: The Big Nerd Ranch", "Head First Java 2nd Edition"],
        ["Android Programming: The Big Nerd Ranch", "Head First Java 2nd Edition", "Beginning Programming with Java"],
        ["Android Programming: The Big Nerd Ranch", "Beginning Programming with Java", "Java 8 Pocket Guide"],
        ["A Beginner’s Guide", "Android Programming: The Big Nerd Ranch", "Head First Java 2nd Edition"],
        ["A Beginner’s Guide", "Head First Java 2nd Edition", "Beginning Programming with Java"],
        ["Java: The Complete Reference", "Java For Dummies", "Android Programming: The Big Nerd Ranch"],
        ["Java For Dummies", "Android Programming: The Big Nerd Ranch", "Head First Java 2nd Edition", "Beginning Programming with Java"],
        ["Beginning Programming with Java", "Java 8 Pocket Guide", "C++ Programming in Easy Steps"],
        ["A Beginner’s Guide", "Java: The Complete Reference", "Java For Dummies", "Android Programming: The Big Nerd Ranch"],
        ["A Beginner’s Guide", "Java: The Complete Reference", "Java For Dummies", "HTML and CSS: Design and Build Websites"],
        ["A Beginner’s Guide", "Java: The Complete Reference", "Java For Dummies", "Java 8 Pocket Guide", "HTML and CSS: Design and Build Websites"],
        ["Java For Dummies", "Android Programming: The Big Nerd Ranch", "Head First Java 2nd Edition"],
        ["Java For Dummies", "Android Programming: The Big Nerd Ranch"],
        ["A Beginner’s Guide", "Java: The Complete Reference", "Java For Dummies", "Android Programming: The Big Nerd Ranch"],
        ["A Beginner’s Guide", "Java: The Complete Reference", "Java For Dummies", "Android Programming: The Big Nerd Ranch"],
        ["Head First Java 2nd Edition", "Beginning Programming with Java", "Java 8 Pocket Guide"],
        ["Android Programming: The Big Nerd Ranch", "Head First Java 2nd Edition"],
        ["A Beginner’s Guide", "Java: The Complete Reference", "Java For Dummies"]
    ],
    'Best Buy': [
        ["Desk Top", "Printer", "Flash Drive", "Microsoft Office", "Speakers", "Anti-Virus"],
        ["Lab Top", "Flash Drive", "Microsoft Office", "Lab Top Case", "Anti-Virus"],
        ["Lab Top", "Printer", "Flash Drive", "Microsoft Office", "Anti-Virus", "Lab Top Case", "External Hard Drive"],
        ["Lab Top", "Printer", "Flash Drive", "Anti-Virus", "External Hard Drive", "Lab Top Case"],
        ["Lab Top", "Flash Drive", "Lab Top Case", "Anti-Virus"],
        ["Lab Top", "Printer", "Flash Drive", "Microsoft Office"],
        ["Desk Top", "Printer", "Flash Drive", "Microsoft Office"],
        ["Lab Top", "External Hard Drive", "Anti-Virus"],
        ["Desk Top", "Printer", "Flash Drive", "Microsoft Office", "Lab Top Case", "Anti-Virus", "Speakers", "External Hard Drive"],
        ["Digital Camera", "Lab Top", "Desk Top", "Printer", "Flash Drive", "Microsoft Office", "Lab Top Case", "Anti-Virus", "External Hard Drive", "Speakers"],
        ["Lab Top", "Desk Top", "Lab Top Case", "External Hard Drive", "Speakers", "Anti-Virus"],
        ["Digital Camera", "Lab Top", "Lab Top Case", "External Hard Drive", "Anti-Virus", "Speakers"],
        ["Digital Camera", "Speakers"],
        ["Digital Camera", "Desk Top", "Printer", "Flash Drive", "Microsoft Office"],
        ["Printer", "Flash Drive", "Microsoft Office", "Anti-Virus", "Lab Top Case", "Speakers", "External Hard Drive"],
        ["Digital Camera", "Flash Drive", "Microsoft Office", "Anti-Virus", "Lab Top Case", "External Hard Drive", "Speakers"],
        ["Digital Camera", "Lab Top", "Lab Top Case"],
        ["Digital Camera", "Lab Top Case", "Speakers"],
        ["Digital Camera", "Lab Top", "Printer", "Flash Drive", "Microsoft Office", "Speakers", "Lab Top Case", "Anti-Virus"],
        ["Digital Camera", "Lab Top", "Speakers", "Anti-Virus", "Lab Top Case"]
    ],
    'K-Mart': [
        ["Decorative Pillows", "Quilts", "Embroidered Bedspread"],
        ["Embroidered Bedspread", "Shams", "Kids Bedding", "Bedding Collections", "Bed Skirts", "Bedspreads", "Sheets"],
        ["Decorative Pillows", "Quilts", "Embroidered Bedspread", "Shams", "Kids Bedding", "Bedding Collections"],
        ["Kids Bedding", "Bedding Collections", "Sheets", "Bedspreads", "Bed Skirts"],
        ["Decorative Pillows", "Kids Bedding", "Bedding Collections", "Sheets", "Bed Skirts", "Bedspreads"],
        ["Bedding Collections", "Bedspreads", "Bed Skirts", "Sheets", "Shams", "Kids Bedding"],
        ["Decorative Pillows", "Quilts"],
        ["Decorative Pillows", "Quilts", "Embroidered Bedspread"],
        ["Bedspreads", "Bed Skirts", "Shams", "Kids Bedding", "Sheets"],
        ["Quilts", "Embroidered Bedspread", "Bedding Collections"],
        ["Bedding Collections", "Bedspreads", "Bed Skirts", "Kids Bedding", "Shams", "Sheets"],
        ["Decorative Pillows", "Quilts"],
        ["Embroidered Bedspread", "Shams"],
        ["Sheets", "Shams", "Bed Skirts", "Kids Bedding"],
        ["Decorative Pillows", "Quilts"],
        ["Decorative Pillows", "Kids Bedding", "Bed Skirts", "Shams"],
        ["Decorative Pillows", "Shams", "Bed Skirts"],
        ["Quilts", "Sheets", "Kids Bedding"],
        ["Shams", "Bed Skirts", "Kids Bedding", "Sheets"],
        ["Decorative Pillows", "Bedspreads", "Shams", "Sheets", "Bed Skirts", "Kids Bedding"]
    ]
}

# Prompt user for input
min_support = float(input("Enter minimum support (e.g., 0.2): "))
min_confidence = float(input("Enter minimum confidence (e.g., 0.7): "))

# Convert transactions to DataFrame with binary encoding for apriori and fp-growth
def encode_transactions(transactions, items):
    encoded_vals = []
    for transaction in transactions:
        encoded_transaction = {item: (item in transaction) for item in items}
        encoded_vals.append(encoded_transaction)
    return pd.DataFrame(encoded_vals)

# Brute Force Implementation
def brute_force(transactions, items, min_support):
    start_time = time.time()

    def calculate_support(itemset, transactions):
        count = 0
        for transaction in transactions:
            if itemset.issubset(transaction):
                count += 1
        return count / len(transactions)

    def find_frequent_itemsets(items, transactions, size, min_support):
        itemsets = [set(combo) for combo in combinations(items, size)]
        frequent_itemsets = []
        for itemset in itemsets:
            support = calculate_support(itemset, transactions)
            if support >= min_support:
                frequent_itemsets.append((itemset, support))
        return frequent_itemsets

    k = 1
    frequent_itemsets = []
    while True:
        current_frequent_itemsets = find_frequent_itemsets


# In[4]:


get_ipython().system('pip install mlxtend')
import pandas as pd
from itertools import combinations
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth
import time

# Define items for Amazon, Kmart, and Best Buy
amazon_items = [
    "Wireless Earbuds", "Smartphone", "Laptop", "Smartwatch",
    "Bluetooth Speaker", "Kindle", "HDMI Cable", "Smart Home Assistant",
    "External Hard Drive", "Gaming Headset"
]

kmart_items = [
    "Bedding Set", "Kitchen Utensils", "Cookware Set",
    "Bathroom Towels", "Home Decor", "Laundry Basket",
    "Storage Containers", "Garden Tools", "Electric Grill",
    "Coffee Maker"
]

bestbuy_items = [
    "4K TV", "Gaming Console", "Laptop", "Wireless Router",
    "Bluetooth Headphones", "Smartwatch", "Portable Speaker",
    "External Hard Drive", "Home Theater System", "Smart Doorbell"
]

# Transactions for each store
transactions_db = {
    "Amazon": [
        ["Wireless Earbuds", "Smartwatch", "Bluetooth Speaker"],
        ["Smartphone", "Laptop"],
        ["Smart Home Assistant", "External Hard Drive"],
        ["Gaming Headset", "Wireless Earbuds", "Smartwatch"],
        ["Bluetooth Speaker", "Smartphone"],
        ["Laptop", "HDMI Cable"],
        ["Wireless Earbuds", "Kindle"],
        ["Smartwatch", "Gaming Headset"],
        ["Smart Home Assistant", "Bluetooth Speaker"],
        ["External Hard Drive", "Laptop"]
    ],
    "Kmart": [
        ["Bedding Set", "Kitchen Utensils", "Bathroom Towels"],
        ["Cookware Set", "Electric Grill"],
        ["Home Decor", "Laundry Basket"],
        ["Storage Containers", "Garden Tools", "Coffee Maker"],
        ["Bedding Set", "Laundry Basket"],
        ["Kitchen Utensils", "Bathroom Towels"],
        ["Electric Grill", "Storage Containers"],
        ["Coffee Maker", "Home Decor"],
        ["Garden Tools", "Bedding Set"],
        ["Bathroom Towels", "Kitchen Utensils"]
    ],
    "Best Buy": [
        ["4K TV", "Gaming Console", "Wireless Router"],
        ["Laptop", "Bluetooth Headphones"],
        ["Smartwatch", "Portable Speaker"],
        ["External Hard Drive", "Home Theater System", "Smart Doorbell"],
        ["4K TV", "Laptop"],
        ["Wireless Router", "Smartwatch"],
        ["Bluetooth Headphones", "Gaming Console"],
        ["Portable Speaker", "Smart Doorbell"],
        ["External Hard Drive", "Home Theater System"],
        ["Smartwatch", "Gaming Console"]
    ]
}

# Prompt user for input
min_support = float(input("Enter minimum support (e.g., 0.2): "))
min_confidence = float(input("Enter minimum confidence (e.g., 0.7): "))

# Convert transactions to DataFrame with binary encoding for apriori and fp-growth
def encode_transactions(transactions, items):
    encoded_vals = []
    for transaction in transactions:
        encoded_transaction = {item: (item in transaction) for item in items}
        encoded_vals.append(encoded_transaction)
    return pd.DataFrame(encoded_vals)

# Brute Force Implementation
def brute_force(transactions, items, min_support):
    start_time = time.time()

    def calculate_support(itemset, transactions):
        count = 0
        for transaction in transactions:
            if itemset.issubset(transaction):
                count += 1
        return count / len(transactions)

    def find_frequent_itemsets(items, transactions, size, min_support):
        itemsets = [set(combo) for combo in combinations(items, size)]
        frequent_itemsets = []
        for itemset in itemsets:
            support = calculate_support(itemset, transactions)
            if support >= min_support:
                frequent_itemsets.append((itemset, support))
        return frequent_itemsets

    k = 1
    frequent_itemsets = []
    while True:
        current_frequent_itemsets = find_frequent_itemsets(items, transactions, k, min_support)
        if len(current_frequent_itemsets) == 0:
            break
        frequent_itemsets.extend(current_frequent_itemsets)
        k += 1

    # Generate association rules
    rules = []
    for itemset, support in frequent_itemsets:
        if len(itemset) > 1:
            for consequent in itemset:
                antecedent = itemset - {consequent}
                if len(antecedent) > 0:
                    confidence = calculate_support(itemset, transactions) / calculate_support(antecedent, transactions)
                    if confidence >= min_confidence:
                        rules.append((list(antecedent), list(consequent), confidence, support))

    brute_force_time = time.time() - start_time
    return frequent_itemsets, rules, brute_force_time

# Apriori Implementation
def apriori_algorithm(df, min_support, min_confidence):
    start_time = time.time()
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
    apriori_time = time.time() - start_time
    return frequent_itemsets, rules, apriori_time

# FP-Growth Implementation
def fpgrowth_algorithm(df, min_support, min_confidence):
    start_time = time.time()
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
    fpgrowth_time = time.time() - start_time
    return frequent_itemsets, rules, fpgrowth_time

# Main comparison function
def compare_algorithms(transactions_db, items_db, min_support, min_confidence):
    for store, transactions in transactions_db.items():
        print(f"\n--- {store} ---")
        items = items_db[store]
        transactions = [set(t) for t in transactions]

        # Brute Force
        frequent_itemsets_bf, rules_bf, bf_time = brute_force(transactions, items, min_support)
        print(f"Brute Force Time: {bf_time:.4f} seconds")
        print(f"Frequent Itemsets (Brute Force): {frequent_itemsets_bf}")

        if rules_bf:
            print("\nFinal Association Rules (Brute Force):")
            for idx, (antecedent, consequent, confidence, support) in enumerate(rules_bf, start=1):
                print(f"Rule {idx}: {antecedent} -> {consequent}")
                print(f"Confidence: {confidence * 100:.2f}%")
                print(f"Support: {support * 100:.2f}%\n")

        # Apriori
        df_encoded = encode_transactions(transactions, items)
        frequent_itemsets_ap, rules_ap, ap_time = apriori_algorithm(df_encoded, min_support, min_confidence)
        print(f"Apriori Time: {ap_time:.4f} seconds")
        print(f"Frequent Itemsets (Apriori):\n{frequent_itemsets_ap}")

        if not rules_ap.empty:
            print("\nFinal Association Rules (Apriori):")
            for idx, row in rules_ap.iterrows():
                print(f"Rule {idx + 1}: {row['antecedents']} -> {row['consequents']}")
                print(f"Confidence: {row['confidence'] * 100:.2f}%")
                print(f"Support: {row['support'] * 100:.2f}%\n")

        # FP-Growth
        frequent_itemsets_fp, rules_fp, fp_time = fpgrowth_algorithm(df_encoded, min_support, min_confidence)
        print(f"FP-Growth Time: {fp_time:.4f} seconds")
        print(f"Frequent Itemsets (FP-Growth):\n{frequent_itemsets_fp}")

        if not rules_fp.empty:
            print("\nFinal Association Rules (FP-Growth):")
            for idx, row in rules_fp.iterrows():
                print(f"Rule {idx + 1}: {row['antecedents']} -> {row['consequents']}")
                print(f"Confidence: {row['confidence'] * 100:.2f}%")
                print(f"Support: {row['support'] * 100:.2f}%\n")

# Dictionary of item sets for each store
items_db = {
    "Amazon": amazon_items,
    "Kmart": kmart_items,
    "Best Buy": bestbuy_items
}

# Run the comparison for Amazon, Kmart, and Best Buy
compare_algorithms(transactions_db, items_db, min_support, min_confidence)


# In[ ]:




