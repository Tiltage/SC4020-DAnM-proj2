"""
Cancer Feature Sequential Pattern Mining
Transforms numerical features into categorical sequences and mines patterns
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from scipy import stats
import json
import itertools


class SequenceTransformer:
    """Transform numerical features into categorical sequences based on z-scores"""
    
    def __init__(self, top_k: int = 5, max_length: int = 3, max_gap: int = 1):
        """
        Initialize transformer
        
        Args:
            top_k: Number of top features to select per patient
            max_length: Maximum sequence length
            max_gap: Maximum gap for grouping features into itemsets
        """
        self.top_k = top_k
        self.max_length = max_length
        self.max_gap = max_gap
        self.feature_cols = None
        self.z_score_matrix = None
        
    def fit_transform(self, df: pd.DataFrame) -> List[Dict]:
        """
        Transform dataframe into sequences
        
        Args:
            df: DataFrame with 'id', 'diagnosis', and feature columns
            
        Returns:
            List of dictionaries containing patient sequences
        """
        # Get feature columns (exclude id and diagnosis)
        self.feature_cols = [col for col in df.columns 
                            if col not in ['id', 'diagnosis']]
        
        # Calculate z-scores for each feature
        self.z_score_matrix = {}
        for col in self.feature_cols:
            values = df[col].values
            self.z_score_matrix[col] = stats.zscore(values)
        
        # Transform each patient to sequence
        sequences = []
        for idx, row in df.iterrows():
            patient_id = row['id']
            diagnosis = row['diagnosis']
            
            # Get z-scores for this patient
            feature_scores = [(col, abs(self.z_score_matrix[col][idx])) 
                            for col in self.feature_cols]
            
            # Sort by z-score magnitude (descending)
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Create sequence from top-k features
            sequence = self._create_sequence(feature_scores)
            
            sequences.append({
                'id': patient_id,
                'diagnosis': diagnosis,
                'sequence': sequence,
                'top_features': [f[0] for f in feature_scores[:self.top_k]],
                'z_scores': [f[1] for f in feature_scores[:self.top_k]]
            })
        
        return sequences
    
    def _create_sequence(self, feature_scores: List[Tuple[str, float]]) -> List[List[str]]:
        """
        Create sequence with itemsets from ranked features
        
        Args:
            feature_scores: List of (feature_name, z_score) tuples sorted by z-score
            
        Returns:
            List of itemsets (each itemset is a list of feature names)
        """
        if not feature_scores:
            return []
        
        # Take top-k features
        top_features = feature_scores[:self.top_k]
        
        # Group features into itemsets based on rank proximity (maxgap)
        sequence = []
        current_itemset = [top_features[0][0]]
        
        for i in range(1, min(len(top_features), self.max_length * 2)):
            # Check if this feature should be in the same itemset
            # (within maxgap positions)
            if i - len(current_itemset) <= self.max_gap:
                current_itemset.append(top_features[i][0])
            else:
                # Start new itemset
                sequence.append(current_itemset)
                current_itemset = [top_features[i][0]]
                
                # Stop if we've reached max_length itemsets
                if len(sequence) >= self.max_length:
                    break
        
        # Add last itemset if not empty and within length limit
        if current_itemset and len(sequence) < self.max_length:
            sequence.append(current_itemset)
        
        return sequence[:self.max_length]


class SequencePatternMiner:
    """Mine sequential patterns from categorized sequences"""
    
    def __init__(self, min_support: float = 0.3, max_pattern_length: int = 3):
        """
        Initialize pattern miner
        
        Args:
            min_support: Minimum support threshold (0-1)
            max_pattern_length: Maximum length of patterns to mine
        """
        self.min_support = min_support
        self.max_pattern_length = max_pattern_length
        self.patterns = None
        
    def mine_patterns(self, sequences: List[Dict]) -> pd.DataFrame:
        """
        Mine patterns from sequences
        
        Args:
            sequences: List of sequence dictionaries with 'diagnosis' and 'sequence'
            
        Returns:
            DataFrame of patterns with support and discriminative metrics
        """
        try:
            # Separate by diagnosis
            malignant_seqs = [s for s in sequences if s['diagnosis'] == 'M']
            benign_seqs = [s for s in sequences if s['diagnosis'] == 'B']
            
            print(f"Processing {len(malignant_seqs)} malignant and {len(benign_seqs)} benign sequences")
            
            # Extract patterns from each group
            malignant_patterns = self._extract_patterns(malignant_seqs)
            benign_patterns = self._extract_patterns(benign_seqs)
            
            print(f"Found {len(malignant_patterns)} unique patterns in malignant, {len(benign_patterns)} in benign")
            
            # Calculate support and create pattern dataframe
            all_pattern_keys = set(malignant_patterns.keys()) | set(benign_patterns.keys())
            
            pattern_list = []
            for pattern_key in all_pattern_keys:
                m_count = malignant_patterns.get(pattern_key, 0)
                b_count = benign_patterns.get(pattern_key, 0)
                
                m_support = m_count / len(malignant_seqs) if malignant_seqs else 0
                b_support = b_count / len(benign_seqs) if benign_seqs else 0
                
                # Filter by minimum support
                if m_support >= self.min_support or b_support >= self.min_support:
                    # Convert pattern key back from string representation
                    pattern = self._pattern_from_key(pattern_key)
                    
                    # Calculate lift (discriminative power)
                    pseudocount = 0.5 / len(malignant_seqs)  # Small value relative to dataset size
                    lift = (m_support + pseudocount) / (b_support + pseudocount)

                    
                    # Determine which class this pattern discriminates
                    if lift > 1.5:
                        discriminates = 'Malignant'
                    elif lift < 0.67:
                        discriminates = 'Benign'
                    else:
                        discriminates = 'Neutral'
                    
                    pattern_list.append({
                        'pattern': pattern,
                        'pattern_str': self._pattern_to_string(pattern),
                        'length': len(pattern),
                        'malignant_support': m_support,
                        'benign_support': b_support,
                        'malignant_count': m_count,
                        'benign_count': b_count,
                        'lift': lift,
                        'log_lift': np.log(lift) if lift > 0 else -10,  # handle log(0)
                        'discriminates': discriminates
                    })
            
            # Create DataFrame and sort by discriminative power
            if pattern_list:
                df_patterns = pd.DataFrame(pattern_list)
                df_patterns = df_patterns.sort_values('log_lift', 
                                                      key=lambda x: abs(x), 
                                                      ascending=False)
                df_patterns = df_patterns.reset_index(drop=True)
                print(f"Successfully mined {len(df_patterns)} patterns meeting support threshold")
            else:
                print("No patterns found meeting the minimum support threshold")
                # Return empty dataframe with correct columns
                df_patterns = pd.DataFrame(columns=[
                    'pattern', 'pattern_str', 'length', 'malignant_support', 
                    'benign_support', 'malignant_count', 'benign_count', 
                    'lift', 'log_lift', 'discriminates'
                ])
            
            self.patterns = df_patterns
            return df_patterns
            
        except Exception as e:
            print(f"Error in mine_patterns: {e}")
            import traceback
            traceback.print_exc()
            # Return empty dataframe on error
            return pd.DataFrame(columns=[
                'pattern', 'pattern_str', 'length', 'malignant_support', 
                'benign_support', 'malignant_count', 'benign_count', 
                'lift', 'log_lift', 'discriminates'
            ])
    
    def _extract_patterns(self, sequences: List[Dict]) -> Dict[str, int]:
        """
        Extract all subset patterns from itemsets
        
        Args:
            sequences: List of sequence dictionaries where each sequence contains one itemset
            
        Returns:
            Dictionary mapping pattern keys to counts
        """
        pattern_counts = defaultdict(int)
        
        for seq_dict in sequences:
            sequence = seq_dict['sequence']
            
            # Each sequence contains one itemset with multiple features
            if len(sequence) == 1 and isinstance(sequence[0], list):
                itemset = sequence[0]  # Extract the actual itemset
                
                # Generate all subsets of the itemset (patterns of different lengths)
                for length in range(1, min(self.max_pattern_length + 1, len(itemset) + 1)):
                    # Generate all combinations of the given length
                    for subset in itertools.combinations(itemset, length):
                        # Sort for consistency and convert to pattern key
                        pattern_key = json.dumps(sorted(subset))
                        pattern_counts[pattern_key] += 1
            else:
                # Handle case where sequence format is different
                print(f"Warning: Unexpected sequence format: {sequence}")
        
        return pattern_counts
    
    def _pattern_from_key(self, pattern_key: str) -> List[str]:
        """Convert pattern key back to pattern list"""
        try:
            return json.loads(pattern_key)
        except json.JSONDecodeError:
            return [pattern_key]
    
    def _pattern_to_string(self, pattern: List[str]) -> str:
        """Convert pattern to readable string format"""
        if isinstance(pattern, list):
            return '{' + ', '.join(pattern) + '}'
        else:
            return '{' + str(pattern) + '}'
    
    def get_top_patterns(self, n: int = 50, discriminates: str = None) -> pd.DataFrame:
        """
        Get top N patterns
        
        Args:
            n: Number of patterns to return
            discriminates: Filter by discrimination class ('Malignant', 'Benign', or None for all)
            
        Returns:
            DataFrame of top patterns
        """
        if self.patterns is None:
            raise ValueError("No patterns mined yet. Call mine_patterns() first.")
        
        df = self.patterns
        
        if discriminates is not None:
            df = df[df['discriminates'] == discriminates]
        
        return df.head(n)
    
    def analyze_patterns(self, top_n: int = 20):
        """Print analysis of mined patterns"""
        if self.patterns is None or len(self.patterns) == 0:
            print("No patterns to analyze")
            return
        
        print(f"\n=== PATTERN ANALYSIS (Top {top_n}) ===")
        print(f"Total patterns mined: {len(self.patterns)}")
        
        # Count by discrimination type
        disc_counts = self.patterns['discriminates'].value_counts()
        print(f"\nPatterns by discrimination type:")
        for disc_type, count in disc_counts.items():
            print(f"  {disc_type}: {count}")
        
        # Count by pattern length
        length_counts = self.patterns['length'].value_counts().sort_index()
        print(f"\nPatterns by length:")
        for length, count in length_counts.items():
            print(f"  Length {length}: {count} patterns")
        
        # Show top patterns
        print(f"\nTop {top_n} discriminative patterns:")
        top_patterns = self.get_top_patterns(top_n)
        
        for i, row in top_patterns.iterrows():
            print(f"{i+1:2d}. [{row['discriminates']:8}] lift:{row['lift']:6.2f} "
                  f"(M:{row['malignant_support']:.3f}, B:{row['benign_support']:.3f}) "
                  f"- {row['pattern_str']}")



def analyze_sequences(sequences: List[Dict]) -> Dict:
    """
    Analyze basic statistics of sequences
    
    Args:
        sequences: List of sequence dictionaries
        
    Returns:
        Dictionary of statistics
    """
    malignant = [s for s in sequences if s['diagnosis'] == 'M']
    benign = [s for s in sequences if s['diagnosis'] == 'B']
    
    stats = {
        'total_sequences': len(sequences),
        'malignant_count': len(malignant),
        'benign_count': len(benign),
        'avg_sequence_length': np.mean([len(s['sequence']) for s in sequences]),
        'avg_itemset_size': np.mean([len(item) for s in sequences 
                                     for item in s['sequence']]),
    }
    
    return stats


def export_results(patterns: pd.DataFrame, sequences: List[Dict], 
                   output_prefix: str = 'cancer_mining'):
    """
    Export results to CSV files
    
    Args:
        patterns: DataFrame of mined patterns
        sequences: List of sequences
        output_prefix: Prefix for output files
    """
    # Export patterns
    patterns_export = patterns[['pattern_str', 'length', 'malignant_support', 
                                'benign_support', 'lift', 'discriminates']].copy()
    patterns_export.to_csv(f'{output_prefix}_patterns.csv', index=False)
    
    # Export sequences
    seq_export = []
    for s in sequences:
        seq_export.append({
            'id': s['id'],
            'diagnosis': s['diagnosis'],
            'sequence': ' → '.join(['{' + ','.join(itemset) + '}' 
                                   for itemset in s['sequence']]),
            'top_features': ','.join(s['top_features'])
        })
    pd.DataFrame(seq_export).to_csv(f'{output_prefix}_sequences.csv', index=False)
    
    print(f"Results exported to {output_prefix}_patterns.csv and {output_prefix}_sequences.csv")


if __name__ == "__main__":
    # Example usage
    print("Cancer Feature Sequential Pattern Mining")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv('../RawData/cancer_data.csv')
    print(f"\nLoaded {len(df)} records")
    
    # Transform to sequences
    print("\nTransforming to sequences...")
    transformer = SequenceTransformer(top_k=5, max_length=3, max_gap=1)
    sequences = transformer.fit_transform(df)
    
    # Analyze sequences
    stats = analyze_sequences(sequences)
    print("\nSequence Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("Sequence format check:")
    for i, seq in enumerate(sequences[:3]):
        print(f"Sequence {i}: diagnosis={seq['diagnosis']}, sequence_type={type(seq['sequence'])}, length={len(seq['sequence'])}")
        print(f"  Content: {seq['sequence']}")

    # Mine patterns
    print("\nMining patterns...")
    miner = SequencePatternMiner(min_support=0.3)
    patterns = miner.mine_patterns(sequences)
    
    print(f"\nFound {len(patterns)} frequent patterns")
    print("\nTop 10 patterns:")
    print(patterns[['pattern_str', 'malignant_support', 'benign_support', 
                    'lift', 'discriminates']].head(10))
    
    # Export results
    export_results(patterns, sequences)