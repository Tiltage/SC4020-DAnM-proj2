import pandas as pd
from itertools import combinations
from collections import defaultdict
import time
from typing import List, Set, Dict, Tuple
import numpy as np

class Apriori:
    """
    Apriori Algorithm Implementation for Frequent Itemset Mining
    """
    def __init__(self, min_support = 0.1, min_confidence = 0.5):
        """
        Initialize Apriori algorithm with parameters
        
        Args:
            min_support: Minimum support threshold (0.0 to 1.0)
            min_confidence: Minimum confidence threshold (0.0 to 1.0)
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.transactions = []
        self.frequent_itemsets = {}
        self.association_rules = []
        self.total_itemsets = 0

        #Binary mapping
        self.item_to_idx = {}
        self.idx_to_item = {} 
        
    def load_data(self, df: pd.DataFrame) -> None:
        """
        Load a pandas DataFrame and encode symptoms as binary vectors.
        Assumes first column is Disease,
        rest are symptoms (may contain NaN)
        """
        # Get all unique items (excluding NaN)
        unique_items = sorted(df.iloc[:, :].stack().dropna().unique())
        self.item_to_idx = {item: i for i, item in enumerate(unique_items)}
        self.idx_to_item = {i: item for item, i in self.item_to_idx.items()}

        # Initialize transaction matrix
        num_transactions = len(df)
        num_items = len(unique_items)
        self.transactions = np.zeros((num_transactions, num_items), dtype=bool)

        for row_idx, row in enumerate(df.iloc[:, :].values):
            for item in row:
                if pd.notna(item):
                    self.transactions[row_idx, self.item_to_idx[item]] = True

        self.total_itemsets = num_transactions
        print(f"Loaded {self.total_itemsets} transactions with {num_items} unique items.")
    
    def calculate_support(self, itemset_indices: Set[int]) -> float:
        """
        Calculate support for an itemset (given as indices)
        """
        # Vectorized: check rows where all itemset columns are True
        count = np.sum(np.all(self.transactions[:, list(itemset_indices)], axis=1))
        return count / self.total_itemsets
    
    def generate_candidates(self, prev_frequent: List[Set[int]], k: int) -> List[Set[int]]:
        """
        Generate candidate itemsets of size k from previous frequent itemsets
        """
        candidates = set()
        for i in range(len(prev_frequent)):
            for j in range(i + 1, len(prev_frequent)):
                union_set = prev_frequent[i] | prev_frequent[j]
                if len(union_set) == k:
                    candidates.add(frozenset(union_set))  # frozenset ensures uniqueness
        # Convert back to list of sets for the rest of the code
        pruned_candidates = [set(x) for x in candidates]

        # Prune: remove candidates with any infrequent (k-1)-subset
        prev_frequent_set = set(frozenset(x) for x in prev_frequent)
        for candidate in candidates:
            valid = True
            for subset in combinations(candidate, k - 1):
                if frozenset(subset) not in prev_frequent_set:
                    valid = False
                    break
            if valid:
                pruned_candidates.append(candidate)

        return pruned_candidates

    def find_frequent_itemsets(self) -> Dict[int, List[Set[int]]]:
        """
        Find all frequent itemsets using Apriori algorithm
        """
        print("Finding frequent itemsets...")
        start_time = time.time()

        self.frequent_itemsets = {}

        # 1-itemsets
        num_items = self.transactions.shape[1]
        frequent_1 = []
        for idx in range(num_items):
            support = self.calculate_support({idx})
            if support >= self.min_support:
                frequent_1.append({idx})

        self.frequent_itemsets[1] = frequent_1
        print(f"Found {len(frequent_1)} frequent 1-itemsets")

        # >2-itemsets
        k = 2
        while self.frequent_itemsets.get(k - 1):
            candidates = self.generate_candidates(self.frequent_itemsets[k - 1], k)
            frequent_k = []
            for candidate in candidates:
                support = self.calculate_support(candidate)
                if support >= self.min_support:
                    frequent_k.append(candidate)
            self.frequent_itemsets[k] = frequent_k
            print(f"Found {len(frequent_k)} frequent {k}-itemsets")
            k += 1

        end_time = time.time()
        print(f"Frequent itemset mining completed in {end_time - start_time:.2f} seconds")
        return self.frequent_itemsets
    
    def generate_association_rules(self) -> List[Dict]:
        """
        Generate association rules from frequent itemsets
        """
        print("Generating association rules...")
        start_time = time.time()
        self.association_rules = []

        for size, itemsets in self.frequent_itemsets.items():
            if size < 2:
                continue
            for itemset in itemsets:
                itemset_list = list(itemset)
                itemset_support = self.calculate_support(itemset)
                for ante_size in range(1, size):
                    for antecedent in combinations(itemset_list, ante_size):
                        antecedent_set = set(antecedent)
                        consequent_set = itemset - antecedent_set
                        antecedent_support = self.calculate_support(antecedent_set)
                        if antecedent_support > 0:
                            confidence = itemset_support / antecedent_support
                            rule = {
                                'antecedent': {self.idx_to_item[i] for i in antecedent_set},
                                'consequent': {self.idx_to_item[i] for i in consequent_set},
                                'support': itemset_support,
                                'confidence': confidence
                            }
                            self.association_rules.append(rule)

        end_time = time.time()
        print(f"Association rules completed in {end_time - start_time:.2f} seconds")
        print(f"Generated {len(self.association_rules)} rules")
        return self.association_rules
    
    def print_frequent_itemsets(self) -> None:
        """Print all frequent itemsets with their support"""
        print("\n" + "="*50)
        print("FREQUENT ITEMSETS")
        print("="*50)
        
        for size, itemsets in self.frequent_itemsets.items():
            print(f"\n{size}-Itemsets (Support >= {self.min_support}):")
            for itemset in itemsets:
                support = self.calculate_support(itemset)
                items_str = {self.idx_to_item[i] for i in itemset}
                print(f"{items_str} - Support: {support:.3f}")
    
    def print_association_rules(self, top_n: int = 10) -> None:
        """
        Print top association rules
        
        Args:
            top_n: Number of top rules to display
        """
        print("\n" + "="*50)
        print("ASSOCIATION RULES")
        print("="*50)
        
        # Sort rules by confidence (you can change to support or lift)
        sorted_rules = sorted(self.association_rules, 
                            key=lambda x: x['confidence'], 
                            reverse=True)
        
        print(f"\nTop {min(top_n, len(sorted_rules))} Rules (by confidence):")
        for i, rule in enumerate(sorted_rules[:top_n]):
            print(f"\nRule {i + 1}:")
            print(f"  IF {rule['antecedent']} THEN {rule['consequent']}")
            print(f"  Support: {rule['support']:.3f}")
            print(f"  Confidence: {rule['confidence']:.3f}")
    
    def get_summary(self) -> Dict:
        """Get summary statistics of the analysis"""
        total_itemsets = sum(len(itemsets) for itemsets in self.frequent_itemsets.values())
        
        return {
            'total_transactions': len(self.transactions),
            'min_support': self.min_support,
            'min_confidence': self.min_confidence,
            'total_frequent_itemsets': total_itemsets,
            'max_itemset_size': max(self.frequent_itemsets.keys()) if self.frequent_itemsets else 0,
            'total_association_rules': len(self.association_rules)
        }


# Example usage and test function
def example_usage():
    """Example of how to use the Apriori algorithm"""
    
    # Sample transaction data
    sample_data = [
        ['milk', 'bread', 'butter'],
        ['beer', 'diapers'],
        ['milk', 'diapers', 'beer', 'cola'],
        ['bread', 'butter', 'milk'],
        ['beer', 'bread'],
        ['milk', 'bread', 'butter', 'beer'],
        ['diapers', 'cola'],
        ['bread', 'milk']
    ]
    sample_data_df = pd.DataFrame.from_records(sample_data)
    print(sample_data_df)

    # Initialize Apriori
    apriori = Apriori(min_support=0.3, min_confidence=0.5)
    
    # Load data
    apriori.load_data(sample_data_df)
    
    # Find frequent itemsets
    frequent_itemsets = apriori.find_frequent_itemsets()
    
    # Generate association rules
    association_rules = apriori.generate_association_rules()
    
    # Display results
    apriori.print_frequent_itemsets()
    apriori.print_association_rules(top_n=5)
    
    # Print summary
    summary = apriori.get_summary()
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    example_usage()