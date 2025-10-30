import pandas as pd
from itertools import combinations
from collections import defaultdict
import time
from typing import List, Set, Dict, Tuple

class Apriori:
    """
    Apriori Algorithm Implementation for Frequent Itemset Mining
    """
    
    def __init__(self, min_support: float = 0.1, min_confidence: float = 0.5):
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
        
    def load_data(self, file_path: str, delimiter: str = ',') -> None:
        """
        Load transaction data from CSV file
        
        Args:
            file_path: Path to CSV file
            delimiter: Column delimiter in CSV
        """
        try:
            df = pd.read_csv(file_path, delimiter=delimiter)
            self.transactions = [set(row.dropna().astype(str)) for _, row in df.iterrows()]
            print(f"Loaded {len(self.transactions)} transactions")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def create_transactions(self, data: List[List[str]]) -> None:
        """
        Create transactions from list of lists
        
        Args:
            data: List of transactions where each transaction is a list of items
        """
        self.transactions = [set(transaction) for transaction in data]
        print(f"Created {len(self.transactions)} transactions")
    
    def get_unique_items(self) -> Set[str]:
        """Get all unique items from all transactions"""
        unique_items = set()
        for transaction in self.transactions:
            unique_items.update(transaction)
        return unique_items
    
    def calculate_support(self, itemset: Set[str]) -> float:
        """
        Calculate support for an itemset
        
        Args:
            itemset: Set of items to calculate support for
            
        Returns:
            Support value (0.0 to 1.0)
        """
        if not self.transactions:
            return 0.0
        
        count = 0
        for transaction in self.transactions:
            if itemset.issubset(transaction):
                count += 1
                
        return count / len(self.transactions)
    
    def generate_candidates(self, prev_frequent: List[Set[str]], k: int) -> List[Set[str]]:
        """
        Generate candidate itemsets of size k
        
        Args:
            prev_frequent: Frequent itemsets of size k-1
            k: Size of candidate itemsets to generate
            
        Returns:
            List of candidate itemsets
        """
        candidates = []
        
        # Join step: Combine itemsets
        for i in range(len(prev_frequent)):
            for j in range(i + 1, len(prev_frequent)):
                itemset1 = prev_frequent[i]
                itemset2 = prev_frequent[j]
                
                # Join if first k-2 items are the same
                if len(itemset1.union(itemset2)) == k:
                    candidate = itemset1.union(itemset2)
                    candidates.append(candidate)
        
        # Prune step: Remove candidates with infrequent subsets
        pruned_candidates = []
        for candidate in candidates:
            valid = True
            # Check all subsets of size k-1
            for subset in combinations(candidate, k - 1):
                if set(subset) not in prev_frequent:
                    valid = False
                    break
            if valid:
                pruned_candidates.append(candidate)
                
        return pruned_candidates
    
    def find_frequent_itemsets(self) -> Dict[int, List[Set[str]]]:
        """
        Find all frequent itemsets using Apriori algorithm
        
        Returns:
            Dictionary with itemset size as key and list of frequent itemsets as value
        """
        print("Finding frequent itemsets...")
        start_time = time.time()
        
        self.frequent_itemsets = {}
        
        # Step 1: Find frequent 1-itemsets
        unique_items = self.get_unique_items()
        frequent_1 = []
        
        for item in unique_items:
            support = self.calculate_support({item})
            if support >= self.min_support:
                frequent_1.append({item})
        
        self.frequent_itemsets[1] = frequent_1
        print(f"Found {len(frequent_1)} frequent 1-itemsets")
        
        # Step 2: Find frequent k-itemsets
        k = 2
        while self.frequent_itemsets[k - 1]:
            # Generate candidates
            candidates = self.generate_candidates(self.frequent_itemsets[k - 1], k)
            
            # Calculate support and filter
            frequent_k = []
            for candidate in candidates:
                support = self.calculate_support(candidate)
                if support >= self.min_support:
                    frequent_k.append(candidate)
            
            self.frequent_itemsets[k] = frequent_k
            print(f"Found {len(frequent_k)} frequent {k}-itemsets")
            
            k += 1
        
        # Remove empty levels
        self.frequent_itemsets = {k: v for k, v in self.frequent_itemsets.items() if v}
        
        end_time = time.time()
        print(f"Frequent itemset mining completed in {end_time - start_time:.2f} seconds")
        
        return self.frequent_itemsets
    
    def generate_association_rules(self) -> List[Dict]:
        """
        Generate association rules from frequent itemsets
        
        Returns:
            List of association rules with support, confidence, and lift
        """
        print("Generating association rules...")
        self.association_rules = []
        
        for itemset_size, itemsets in self.frequent_itemsets.items():
            if itemset_size < 2:  # Need at least 2 items for rules
                continue
                
            for itemset in itemsets:
                itemset_list = list(itemset)
                itemset_support = self.calculate_support(itemset)
                
                # Generate all possible antecedents
                for antecedent_size in range(1, itemset_size):
                    for antecedent in combinations(itemset_list, antecedent_size):
                        antecedent_set = set(antecedent)
                        consequent_set = itemset - antecedent_set
                        
                        # Calculate confidence
                        antecedent_support = self.calculate_support(antecedent_set)
                        if antecedent_support > 0:
                            confidence = itemset_support / antecedent_support
                            
                            if confidence >= self.min_confidence:
                                # Calculate lift
                                consequent_support = self.calculate_support(consequent_set)
                                lift = itemset_support / (antecedent_support * consequent_support) if consequent_support > 0 else 0
                                
                                rule = {
                                    'antecedent': antecedent_set,
                                    'consequent': consequent_set,
                                    'support': itemset_support,
                                    'confidence': confidence,
                                    'lift': lift
                                }
                                self.association_rules.append(rule)
        
        print(f"Generated {len(self.association_rules)} association rules")
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
                print(f"  {set(itemset)} - Support: {support:.3f}")
    
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
            print(f"  Lift: {rule['lift']:.3f}")
    
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
    
    # Initialize Apriori
    apriori = Apriori(min_support=0.3, min_confidence=0.5)
    
    # Load data
    apriori.create_transactions(sample_data)
    
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
    # Run example
    example_usage()
    
    # For using with CSV file:
    # apriori = Apriori(min_support=0.1, min_confidence=0.5)
    # apriori.load_data('transactions.csv')
    # apriori.find_frequent_itemsets()
    # apriori.generate_association_rules()
    # apriori.print_frequent_itemsets()
    # apriori.print_association_rules()