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
from itertools import combinations

class SequenceTransformer:
    """Transform numerical features into categorical sequences based on z-scores"""
    
    def __init__(self, top_k: int = 5, max_length: int = 3, max_gap: int = 1):
        """
        Initialize transformer
        
        Args:
            top_k: Number of top features to select per patient
            max_length: Maximum sequence length
            max_gap: Maximum gap allowed between consecutive items in subsequence
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
            List of dictionaries containing patient sequences and all valid subsequences
        """
        # Get feature columns (exclude id and diagnosis)
        self.feature_cols = [col for col in df.columns 
                            if col not in ['id', 'diagnosis']]
        
        # Calculate z-scores for each feature
        self.z_score_matrix = {}
        for col in self.feature_cols:
            values = df[col].values
            self.z_score_matrix[col] = np.round(stats.zscore(values), 4)
        
        # Transform each patient to sequences
        sequences = []
        for idx, row in df.iterrows():
            patient_id = row['id']
            diagnosis = row['diagnosis']
            
            # Get z-scores for this patient
            feature_scores = [(col, abs(self.z_score_matrix[col][idx])) 
                            for col in self.feature_cols]
            
            # Sort by z-score magnitude (descending)
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get top-k features
            top_features = [f[0] for f in feature_scores[:self.top_k]]
            top_z_scores = [float(f[1]) for f in feature_scores[:self.top_k]]
            
            # Generate all valid subsequences
            subsequences = self._generate_all_subsequences(top_features)
            
            sequences.append({
                'id': patient_id,
                'diagnosis': diagnosis,
                'top_features': top_features,
                'z_scores': top_z_scores,
                'subsequences': subsequences  # All valid subsequences for mining
            })
        
        return sequences
    
    def _generate_all_subsequences(self, features: List[str]) -> List[List[str]]:
        """
        Generate all valid subsequences from ranked features respecting max_gap and max_length
        
        Args:
            features: List of feature names in ranked order (top-k)
            
        Returns:
            List of all valid subsequences
        """
        if not features:
            return []
        
        all_subsequences = []
        n = len(features)
        
        # Generate subsequences of length 1 to max_length
        for length in range(1, min(self.max_length, n) + 1):
            # Generate all combinations of indices of this length
            for indices in combinations(range(n), length):
                # Check if this combination satisfies max_gap constraint
                if self._check_max_gap(indices):
                    # Extract features at these indices
                    subsequence = [features[i] for i in indices]
                    all_subsequences.append(subsequence)
        
        return all_subsequences
    
    def _check_max_gap(self, indices: tuple) -> bool:
        """
        Check if a sequence of indices satisfies the max_gap constraint
        
        Args:
            indices: Tuple of indices in ascending order
            
        Returns:
            True if all consecutive gaps are <= max_gap
        """
        if len(indices) <= 1:
            return True
        
        # Check gap between each consecutive pair
        for i in range(len(indices) - 1):
            gap = indices[i + 1] - indices[i] - 1
            if gap > self.max_gap:
                return False
        
        return True
    
class SequencePatternMiner:
    """Mine sequential patterns from categorized sequences"""
    
    def __init__(self, min_support: float = 0.05, max_pattern_length: int = 3):
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
            sequences: List of sequence dictionaries with 'diagnosis' and 'subsequences'
            
        Returns:
            DataFrame of patterns with support and discriminative metrics
        """
        try:
            # Separate by diagnosis
            malignant_seqs = [s for s in sequences if s['diagnosis'] == 'M']
            benign_seqs = [s for s in sequences if s['diagnosis'] == 'B']
            
            print(f"Processing {len(malignant_seqs)} malignant and {len(benign_seqs)} benign patient records")
            
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
                    
                    # Filter by max pattern length
                    if len(pattern) <= self.max_pattern_length:
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
        Extract all subsequence patterns from patient records
        
        Args:
            sequences: List of sequence dictionaries with 'subsequences' field
            
        Returns:
            Dictionary mapping pattern keys to counts
        """
        pattern_counts = defaultdict(int)
        
        for seq_dict in sequences:
            # Get all subsequences for this patient
            subsequences = seq_dict.get('subsequences', [])
            
            # Each subsequence is already a valid ordered pattern
            for subseq in subsequences:
                # Convert to pattern key (preserve order - sequential patterns!)
                # Don't sort because order matters: [A, B, C] ≠ [C, B, A]
                pattern_key = json.dumps(subseq)
                pattern_counts[pattern_key] += 1
        
        return pattern_counts
    
    def _pattern_from_key(self, pattern_key: str) -> List[str]:
        """Convert pattern key back to pattern list"""
        try:
            return json.loads(pattern_key)
        except json.JSONDecodeError:
            return [pattern_key]
    
    def _pattern_to_string(self, pattern: List[str]) -> str:
        """Convert pattern to readable string format (preserving order)"""
        if isinstance(pattern, list):
            if len(pattern) == 1:
                # Single feature
                return '{' + pattern[0] + '}'
            else:
                # Sequential pattern
                return ' → '.join(['{' + feat + '}' for feat in pattern])
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
        
        print(f"\n{'='*80}")
        print(f"SEQUENTIAL PATTERN MINING ANALYSIS (Top {top_n})")
        print(f"{'='*80}")
        print(f"Total patterns mined: {len(self.patterns)}")
        
        # Count by discrimination type
        disc_counts = self.patterns['discriminates'].value_counts()
        print(f"\nPatterns by discrimination type:")
        for disc_type, count in disc_counts.items():
            print(f"  {disc_type:.<20} {count:>6} ({count/len(self.patterns)*100:>5.1f}%)")
        
        # Count by pattern length
        length_counts = self.patterns['length'].value_counts().sort_index()
        print(f"\nPatterns by length:")
        for length, count in length_counts.items():
            print(f"  Length {length}:.............. {count:>6} patterns ({count/len(self.patterns)*100:>5.1f}%)")
        
        # Statistics
        print(f"\nLift statistics:")
        print(f"  Max lift (Malignant):...... {self.patterns[self.patterns['discriminates']=='Malignant']['lift'].max():.2f}")
        print(f"  Min lift (Benign):......... {self.patterns[self.patterns['discriminates']=='Benign']['lift'].min():.2f}")
        print(f"  Mean |log(lift)|:.......... {abs(self.patterns['log_lift']).mean():.2f}")
        
        # Show top patterns
        print(f"\n{'─'*80}")
        print(f"Top {top_n} Most Discriminative Patterns:")
        print(f"{'─'*80}")
        top_patterns = self.get_top_patterns(top_n)
        
        for i, row in top_patterns.iterrows():
            print(f"\n{i+1:2d}. {row['pattern_str']}")
            print(f"    Class: {row['discriminates']:9} | Lift: {row['lift']:8.2f} | Length: {row['length']}")
            print(f"    Malignant Support: {row['malignant_support']:6.2%} ({row['malignant_count']:3d} patients)")
            print(f"    Benign Support:    {row['benign_support']:6.2%} ({row['benign_count']:3d} patients)")
        




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
        sequences: List of sequences from Part 1
        output_prefix: Prefix for output files
    """
    # Export patterns
    patterns_export = patterns[['pattern_str', 'length', 'malignant_support', 
                                'benign_support', 'malignant_count', 'benign_count',
                                'lift', 'discriminates']].copy()
    patterns_export.to_csv(f'{output_prefix}_patterns.csv', index=False)
    print(f"Exported {len(patterns_export)} patterns to {output_prefix}_patterns.csv")
    
    # Export patient-level features (top-k ranked by z-score)
    seq_export = []
    for s in sequences:
        # Format top features with their z-scores
        features_with_scores = []
        for feat, zscore in zip(s['top_features'], s['z_scores']):
            features_with_scores.append(f"{feat}({zscore:.2f})")
        
        seq_export.append({
            'id': s['id'],
            'diagnosis': s['diagnosis'],
            'top_features': ', '.join(s['top_features']),
            'top_features_with_zscores': ', '.join(features_with_scores),
            'num_subsequences_generated': len(s['subsequences']),
            'mean_zscore': np.mean(s['z_scores']),
            'max_zscore': max(s['z_scores'])
        })
    
    seq_df = pd.DataFrame(seq_export)
    seq_df.to_csv(f'{output_prefix}_patient_features.csv', index=False)
    print(f"Exported {len(seq_df)} patient records to {output_prefix}_patient_features.csv")
    
    
    print(f"Files created:")
    print(f"  1. {output_prefix}_patterns.csv - {len(patterns_export)} mined patterns")
    print(f"  2. {output_prefix}_patient_features.csv - {len(seq_df)} patient records")