"""
Part 2: Sensitivity Analysis Across Discretization Strategies
==============================================================
Version without list comprehensions - uses explicit for loops only

This script takes the Part 1 output (feature sequences ranked by z-score)
and applies three different discretization strategies (uniform, quantile, kmeans)
to compare their effectiveness in pattern mining.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns


class Discretization_Analyzer:
    """
    Analyzes sequences using different discretization strategies
    """
    
    def __init__(self, df, sequences, n_bins=3):
        """
        Parameters:
        -----------
        df : DataFrame
            Original cancer dataset with continuous features
        sequences : list
            Output from Part 1 - list of dictionaries with 'sequence', 'diagnosis', etc.
        n_bins : int
            Number of bins for discretization (default: 3 for low/medium/high)
        """
        self.df = df
        self.sequences = sequences
        self.n_bins = n_bins
        
        # Get feature columns (exclude non-feature columns)
        self.feature_cols = []
        for col in df.columns:
            if col not in ['id', 'diagnosis', 'Unnamed: 32']:
                self.feature_cols.append(col)
        
        self.results = {}
        
    def discretize(self, strategy='quantile'):
        """
        Apply discretization strategy to all features
        
        Parameters:
        -----------
        strategy : str
            One of 'uniform', 'quantile', or 'kmeans'
            
        Returns:
        --------
        discretizer : fitted KBinsDiscretizer
        discretized_df : DataFrame with discretized values
        """
        print(f"\nApplying {strategy} Discretations\n")
        
        # Prepare data
        X = self.df[self.feature_cols].values
        
        # Apply discretization
        discretizer = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode='ordinal',
            strategy=strategy
        )
        X_binned = discretizer.fit_transform(X)
        
        # Create discretized dataframe
        discretized_df = pd.DataFrame(
            X_binned,
            columns=self.feature_cols,
            index=self.df.index
        )
        
        # Report bin edges for first 5 features
        print(f"\nBin edges for first 5 features:")
        num_features_to_show = min(5, len(self.feature_cols))
        for i in range(num_features_to_show):
            feature = self.feature_cols[i]
            edges = discretizer.bin_edges_[i]
            print(f"  {feature}:")
            print(f"    {edges}")
            
        # Check bin distribution for first feature
        print(f"\nBin distribution for '{self.feature_cols[0]}':")
        first_column = X_binned[:, 0]
        bin_counts = pd.Series(first_column).value_counts().sort_index()
        
        for bin_idx, count in bin_counts.items():
            pct = count / len(X_binned) * 100
            label_map = ['low', 'medium', 'high']
            label = label_map[int(bin_idx)]
            print(f"  {label} (bin {int(bin_idx)}): {count} samples ({pct:.1f}%)")
        
        return discretizer, discretized_df
    
    def create_discretized_sequences(self, discretized_df, strategy_name):
        """
        Convert Part 1 sequences to discretized sequences
        
        Parameters:
        -----------
        discretized_df : DataFrame
            Discretized feature values
        strategy_name : str
            Name of discretization strategy
            
        Returns:
        --------
        discretized_sequences : list
            List of dictionaries with discretized sequences
        """
        print(f"Number of input sequences: {len(self.sequences)}")
        
        discretized_sequences = []
        label_map = {0: 'low', 1: 'medium', 2: 'high'}
        
        for i in range(len(self.sequences)):
            seq_data = self.sequences[i]
            
            # Get top-k features (ranked by z-score)
            top_features = seq_data['top_features']  # List of 5 feature names

            # Get discretized labels for these features in order
            discretized_items = []
            for feature_name in top_features:
                # Get bin value
                bin_value = int(discretized_df.iloc[i][feature_name])
                label = label_map[bin_value]
                
                discretized_item = f"{label}_{feature_name}"
                discretized_items.append(discretized_item)

            discretized_sequence = []
            for item in discretized_items:
                discretized_sequence.append([item])
            
            discretized_seq_dict = {
                'id': seq_data['id'],
                'diagnosis': seq_data['diagnosis'],
                'original_sequence': seq_data['top_features'],
                'discretized_sequence': discretized_sequence,
                'z_scores': seq_data['z_scores']
            }
            
            discretized_sequences.append(discretized_seq_dict)
        
        print(f"Created {len(discretized_sequences)} discretized sequences")
        
        # Show examples
        print(f"\nExample discretized sequences (first 3 patients):")
        num_examples = min(3, len(discretized_sequences))
        for i in range(num_examples):
            seq = discretized_sequences[i]
            print(f"\n  Patient {seq['id']} ({seq['diagnosis']}):")
            
            # Show first 3 features from original
            orig_preview = []
            for j in range(min(3, len(seq['original_sequence']))):
                orig_preview.append(seq['original_sequence'][j])
            print(f"    Original: {orig_preview}")
            
            # Show first 3 features from discretized
            disc_preview = []
            for j in range(min(3, len(seq['discretized_sequence']))):
                disc_preview.append(seq['discretized_sequence'][j][0])
            print(f"    Discretized: {disc_preview}")
        
        return discretized_sequences
    
    def find_frequent_patterns(self, discretized_sequences, min_support=0.01, max_gap=1):
        '''
        Find frequent sequential patterns using GSP with max_gap 1
        '''
        print(f"\nMining patterns with min_support={min_support}, max_gap={max_gap}")
        
        # Extract sequences
        sequences = [seq_data['discretized_sequence'] for seq_data in discretized_sequences]
        
        n_sequences = len(sequences)
        min_count = int(min_support * n_sequences)
        
        print(f"Minimum count needed: {min_count}")
        
        # length 1
        item_counts = defaultdict(int)
        
        for seq in sequences:
            seen_items = set()
            for itemset in seq:
                for item in itemset:
                    seen_items.add(item)
            for item in seen_items:
                item_counts[item] += 1
        
        frequent_1 = {}
        for item, count in item_counts.items():
            if count >= min_count:
                frequent_1[item] = count / n_sequences
        
        # length 2
        pattern_counts_2 = defaultdict(int)
        
        for seq in sequences:
            for i in range(len(seq)):
                # j can be i+1 (gap=0) or i+2 (gap=1) when max_gap=1
                for j in range(i + 1, min(i + max_gap + 2, len(seq))):
                    for item1 in seq[i]:
                        for item2 in seq[j]:
                            pattern = (item1, item2)
                            pattern_counts_2[pattern] += 1
        
        frequent_2 = {}
        for pattern, count in pattern_counts_2.items():
            if count >= min_count:
                frequent_2[pattern] = count / n_sequences
        
        # length 3
        pattern_counts_3 = defaultdict(int)
        
        for seq in sequences:
            for i in range(len(seq)):
                for j in range(i + 1, min(i + max_gap + 2, len(seq))):
                    for k in range(j + 1, min(j + max_gap + 2, len(seq))):
                        for item1 in seq[i]:
                            for item2 in seq[j]:
                                for item3 in seq[k]:
                                    pattern = (item1, item2, item3)
                                    pattern_counts_3[pattern] += 1
        
        frequent_3 = {}
        for pattern, count in pattern_counts_3.items():
            if count >= min_count:
                frequent_3[pattern] = count / n_sequences
        
        print(f"Found {len(frequent_1)} frequent 1-itemsets")
        print(f"Found {len(frequent_2)} frequent 2-sequences")
        print(f"Found {len(frequent_3)} frequent 3-sequences")
            
        # Show examples
        if len(frequent_1) > 0:
            examples = list(frequent_1.items())[:3]
            print(f"\n  Example 1-itemsets:")
            for item, support in examples:
                print(f"    {item}: {support:.2%}")
        
        if len(frequent_2) > 0:
            examples = list(frequent_2.items())[:3]
            print(f"\n  Example 2-sequences:")
            for pattern, support in examples:
                print(f"{pattern}: {support:.2%}")
        
        return {
            'length_1': frequent_1,
            'length_2': frequent_2,
            'length_3': frequent_3
        }
    
    def calculate_lift_by_class(self, discretized_sequences, patterns):
        """
        Calculate lift for patterns in malignant vs benign classes
        
        Parameters:
        -----------
        discretized_sequences : list
            Discretized sequences with diagnosis
        patterns : dict
            Frequent patterns
            
        Returns:
        --------
        pattern_analysis : list
            List of patterns with lift calculations
        """
        print(f"\nCalculating lift by diagnosis class")
        
        # Separate by diagnosis
        malignant_seqs = []
        benign_seqs = []
        
        for seq_data in discretized_sequences:
            if seq_data['diagnosis'] == 'M':
                malignant_seqs.append(seq_data['discretized_sequence'])
            elif seq_data['diagnosis'] == 'B':
                benign_seqs.append(seq_data['discretized_sequence'])
        
        n_malignant = len(malignant_seqs)
        n_benign = len(benign_seqs)
        n_total = n_malignant + n_benign
        
        print(f"  Malignant patients: {n_malignant}")
        print(f"  Benign patients: {n_benign}")
        
        pattern_analysis = []
        
        # Analyze each pattern length
        for length, pattern_dict in patterns.items():
            for pattern, overall_support in pattern_dict.items():
                # Count in malignant
                mal_count = self._count_pattern_in_sequences(pattern, malignant_seqs)
                if n_malignant > 0:
                    mal_support = mal_count / n_malignant
                else:
                    mal_support = 0
                
                # Count in benign
                ben_count = self._count_pattern_in_sequences(pattern, benign_seqs)
                if n_benign > 0:
                    ben_support = ben_count / n_benign
                else:
                    ben_support = 0
                
                # Calculate lift
                expected_mal = overall_support * (n_malignant / n_total)
                if expected_mal > 0:
                    lift_mal = mal_support / expected_mal
                else:
                    lift_mal = 0
                
                expected_ben = overall_support * (n_benign / n_total)
                if expected_ben > 0:
                    lift_ben = ben_support / expected_ben
                else:
                    lift_ben = 0
                
                # Classify pattern
                if lift_mal > 1.5 and lift_ben < 0.7:
                    discriminates = 'Malignant'
                elif lift_ben > 1.5 and lift_mal < 0.7:
                    discriminates = 'Benign'
                else:
                    discriminates = 'Neutral'
                
                pattern_dict = {
                    'pattern': pattern,
                    'length': length,
                    'overall_support': overall_support,
                    'malignant_support': mal_support,
                    'benign_support': ben_support,
                    'lift_malignant': lift_mal,
                    'lift_benign': lift_ben,
                    'discriminates': discriminates
                }
                
                pattern_analysis.append(pattern_dict)
        
        return pattern_analysis
    
    def _count_pattern_in_sequences(self, pattern, sequences):
        """Helper to count how many sequences contain a pattern"""
        if isinstance(pattern, tuple):
            # Multi-item pattern
            count = 0
            for seq in sequences:
                if self._sequence_contains_pattern(seq, pattern):
                    count += 1
            return count
        else:
            # Single item
            count = 0
            for seq in sequences:
                found = False
                for itemset in seq:
                    if pattern in itemset:
                        found = True
                        break
                if found:
                    count += 1
            return count
    
    def _sequence_contains_pattern(self, sequence, pattern):
        """Check if sequence contains pattern in order"""
        pattern_idx = 0
        
        for itemset in sequence:
            if pattern_idx >= len(pattern):
                return True
            
            if pattern[pattern_idx] in itemset:
                pattern_idx += 1
        
        return pattern_idx >= len(pattern)
    
    def summarize_pattern_analysis(self, pattern_analysis):
        """
        Create summary statistics by pattern length and class discrimination
        
        Parameters:
        -----------
        pattern_analysis : list
            Pattern analysis results
            
        Returns:
        --------
        summary_df : DataFrame
            Summary statistics
        """
        # Check if pattern_analysis is empty
        if len(pattern_analysis) == 0:
            print("No patterns found to analyze!")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['length', 'discriminates', 'pattern_count', 
                                        'lift_mean', 'lift_std', 
                                        'malignant_support_mean', 'benign_support_mean'])
        
        df = pd.DataFrame(pattern_analysis)
        
        # Debug: Show what columns we have
        print(f"\nPattern analysis DataFrame columns: {df.columns.tolist()}")
        print(f"Number of patterns: {len(df)}")
        
        # Check if required columns exist
        required_cols = ['length', 'discriminates', 'pattern', 'lift_malignant', 
                        'lift_benign', 'malignant_support', 'benign_support']
        missing_cols = []
        for col in required_cols:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            print(f"Available columns: {df.columns.tolist()}")
            print(f"First row: {df.iloc[0].to_dict() if len(df) > 0 else 'No data'}")
            raise KeyError(f"Missing required columns: {missing_cols}")
        
                # Group by length and discrimination
        summary = df.groupby(['length', 'discriminates']).agg({
            'pattern': 'count',
            'lift_malignant': ['mean', 'std'],
            'lift_benign': ['mean', 'std'],
            'malignant_support': 'mean',
            'benign_support': 'mean'
        }).round(3)
        
        # Flatten the multi-level column names
        new_columns = []
        for col in summary.columns.values:
            if isinstance(col, tuple):
                # Join tuple elements with underscore, remove trailing underscores
                col_name = '_'.join(col).strip('_')
                new_columns.append(col_name)
            else:
                new_columns.append(col)

        summary.columns = new_columns

        # Rename to simpler names
        summary = summary.rename(columns={
            'pattern_count': 'pattern_count',
            'lift_malignant_mean': 'lift_mean',
            'lift_malignant_std': 'lift_std',
            'lift_benign_mean': 'lift_benign_mean',
            'lift_benign_std': 'lift_benign_std',
            'malignant_support_mean': 'malignant_support_mean',
            'benign_support_mean': 'benign_support_mean'
        })
        
        # Reset index for better display
        summary = summary.reset_index()
        
        return summary
    
    def run_complete_analysis(self, strategies=None, min_support=0.3):
        """
        Run complete analysis for all strategies
        
        Parameters:
        -----------
        strategies : list
            List of discretization strategies to test
        min_support : float
            Minimum support threshold
            
        Returns:
        --------
        results : dict
            Results for each strategy
        """
        if strategies is None:
            strategies = ['uniform', 'quantile', 'kmeans']
        
        all_results = {}
        
        for strategy in strategies:
            print(f"\n{'#'*70}")
            print(f"Strategy: {strategy}")
            print(f"{'#'*70}")
            
            # Step 1: Discretize
            discretizer, discretized_df = self.discretize(strategy)
            
            # Step 2: Create discretized sequences
            discretized_sequences = self.create_discretized_sequences(
                discretized_df, strategy
            )
            
            # Step 3: Find frequent patterns
            patterns = self.find_frequent_patterns(
                discretized_sequences, min_support
            )
            
            # Step 4: Analyze by class
            pattern_analysis = self.calculate_lift_by_class(
                discretized_sequences, patterns
            )
            
            # Step 5: Summarize
            summary = self.summarize_pattern_analysis(pattern_analysis)
            
            print(f"\nPattern Analysis Summary:")
            print(summary.to_string(index=False))
            
            # Store results
            result_dict = {
                'discretizer': discretizer,
                'discretized_df': discretized_df,
                'discretized_sequences': discretized_sequences,
                'patterns': patterns,
                'pattern_analysis': pattern_analysis,
                'summary': summary
            }
            
            all_results[strategy] = result_dict
        
        return all_results
    
    def create_comparison_table(self, all_results):
        """
        Create comparison table across all strategies
        
        Parameters:
        -----------
        all_results : dict
            Results from all strategies
            
        Returns:
        --------
        comparison_df : DataFrame
            Comparison table
        """
        comparison = []
        
        for strategy, results in all_results.items():
            patterns = results['patterns']
            summary = results['summary']
            
            # Get counts by discrimination type
            mal_patterns = summary[summary['discriminates'] == 'Malignant']
            ben_patterns = summary[summary['discriminates'] == 'Benign']
            
            # Calculate totals
            mal_pattern_count = 0
            if len(mal_patterns) > 0:
                mal_pattern_count = mal_patterns['pattern_count'].sum()
            
            ben_pattern_count = 0
            if len(ben_patterns) > 0:
                ben_pattern_count = ben_patterns['pattern_count'].sum()
            
            avg_mal_lift = 0
            if len(mal_patterns) > 0:
                avg_mal_lift = mal_patterns['lift_mean'].mean()
            
            avg_ben_lift = 0
            if len(ben_patterns) > 0:
                avg_ben_lift = ben_patterns['lift_mean'].mean()
            
            comparison_dict = {
                'Strategy': strategy,
                'Total Patterns (L=1)': len(patterns['length_1']),
                'Total Patterns (L=2)': len(patterns['length_2']),
                'Total Patterns (L=3)': len(patterns['length_3']),
                'Malignant Patterns': mal_pattern_count,
                'Benign Patterns': ben_pattern_count,
                'Avg Malignant Lift': avg_mal_lift,
                'Avg Benign Lift': avg_ben_lift
            }
            
            comparison.append(comparison_dict)
        
        comparison_df = pd.DataFrame(comparison)
        
        print("Comparison across strategies")
        print(comparison_df.to_string(index=False))
        
        return comparison_df


    def visualize_results(all_results, comparison_df):
        """
        Create visualizations comparing strategies
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Discretization Strategy Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Total patterns by strategy
        ax1 = axes[0, 0]
        strategies = comparison_df['Strategy']
        x = np.arange(len(strategies))
        width = 0.25
        
        ax1.bar(x - width, comparison_df['Total Patterns (L=1)'], width, label='Length 1', alpha=0.8)
        ax1.bar(x, comparison_df['Total Patterns (L=2)'], width, label='Length 2', alpha=0.8)
        ax1.bar(x + width, comparison_df['Total Patterns (L=3)'], width, label='Length 3', alpha=0.8)
        
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Number of Patterns')
        ax1.set_title('Frequent Patterns by Strategy')
        ax1.set_xticks(x)
        ax1.set_xticklabels(strategies)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Discriminative patterns
        ax2 = axes[0, 1]
        ax2.bar(x - width/2, comparison_df['Malignant Patterns'], width, 
                label='Malignant', alpha=0.8, color='red')
        ax2.bar(x + width/2, comparison_df['Benign Patterns'], width, 
                label='Benign', alpha=0.8, color='blue')
        
        ax2.set_xlabel('Strategy')
        ax2.set_ylabel('Number of Patterns')
        ax2.set_title('Discriminative Patterns by Class')
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategies)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Average lift values
        ax3 = axes[1, 0]
        ax3.bar(x - width/2, comparison_df['Avg Malignant Lift'], width, 
                label='Malignant', alpha=0.8, color='red')
        ax3.bar(x + width/2, comparison_df['Avg Benign Lift'], width, 
                label='Benign', alpha=0.8, color='blue')
        
        ax3.set_xlabel('Strategy')
        ax3.set_ylabel('Average Lift')
        ax3.set_title('Average Lift by Class')
        ax3.set_xticks(x)
        ax3.set_xticklabels(strategies)
        ax3.legend()
        ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline')
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Bin distribution for first strategy
        ax4 = axes[1, 1]
        strategy_names = []
        for name in all_results.keys():
            strategy_names.append(name)
        
        first_strategy = strategy_names[0]
        first_result = all_results[first_strategy]
        discretized_df = first_result['discretized_df']
        
        # Get bin distribution for first feature
        first_feature = discretized_df.columns[0]
        bin_counts = discretized_df[first_feature].value_counts().sort_index()
        
        ax4.bar(['Low', 'Medium', 'High'], bin_counts.values, alpha=0.8)
        ax4.set_xlabel('Bin')
        ax4.set_ylabel('Count')
        ax4.set_title(f'Bin Distribution\n({first_strategy}, {first_feature})')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig

    def count_high_features(self, pattern):
        """
        Helper method: Count how many features in pattern start with 'high_'
        
        """
        if isinstance(pattern, tuple):
            return sum(1 for item in pattern if item.startswith('high_'))
        elif isinstance(pattern, str):
            return 1 if pattern.startswith('high_') else 0
        else:
            return 0
    
    def analyze_high_feature_prevalence(self, all_results, strategy):
        """
        Analyze how many 'high' features appear in malignant vs benign patterns
        
        Parameters:
        -----------
        all_results : dict
            Results from run_complete_analysis()
        strategy : str
            Which discretization strategy to analyze
        """
        
        # Get pattern analysis for chosen strategy
        pattern_analysis = all_results[strategy]['pattern_analysis']
        
        # Separate patterns by class
        malignant_patterns = [p for p in pattern_analysis if p['discriminates'] == 'Malignant']
        benign_patterns = [p for p in pattern_analysis if p['discriminates'] == 'Benign']
        
        # Analyze by length
        for length_num in [2, 3]:
            length_key = f'length_{length_num}'
            
            print(f"\n{'─'*80}")
            print(f"LENGTH-{length_num} PATTERNS")
            print(f"{'─'*80}")
            
            # Filter by length
            mal_length = [p for p in malignant_patterns if p['length'] == length_key]
            ben_length = [p for p in benign_patterns if p['length'] == length_key]
            
            if len(mal_length) == 0 and len(ben_length) == 0:
                print(f"No length-{length_num} patterns found for either class")
                continue
            
            # Count 'high' features using helper method
            mal_high_counts = [self.count_high_features(p['pattern']) for p in mal_length]
            ben_high_counts = [self.count_high_features(p['pattern']) for p in ben_length]
            
            # Statistics - Malignant
            print(f"\nMalignant patterns (n={len(mal_length)}):")
            if mal_high_counts:
                print(f"  Average 'high' features per pattern: {np.mean(mal_high_counts):.2f}")
                for i in range(length_num + 1):
                    count = mal_high_counts.count(i)
                    pct = count / len(mal_high_counts) * 100 if mal_high_counts else 0
                    print(f"  Patterns with {i} 'high' feature{'s' if i != 1 else ''}: {count} ({pct:.1f}%)")
            else:
                print("  No malignant patterns of this length")
            
            # Statistics - Benign
            print(f"\nBenign patterns (n={len(ben_length)}):")
            if ben_high_counts:
                print(f"  Average 'high' features per pattern: {np.mean(ben_high_counts):.2f}")
                for i in range(length_num + 1):
                    count = ben_high_counts.count(i)
                    pct = count / len(ben_high_counts) * 100 if ben_high_counts else 0
                    print(f"  Patterns with {i} 'high' feature{'s' if i != 1 else ''}: {count} ({pct:.1f}%)")
            else:
                print("  No benign patterns of this length")
            
            # Comparison
            if mal_high_counts and ben_high_counts:
                mal_avg = np.mean(mal_high_counts)
                ben_avg = np.mean(ben_high_counts)
                
                print(f"Malignant avg 'high' features: {mal_avg:.2f} out of {length_num}")
                print(f"Benign avg 'high' features:    {ben_avg:.2f} out of {length_num}")
                
                
                # Percentage of max possible
                mal_pct = (mal_avg / length_num) * 100
                ben_pct = (ben_avg / length_num) * 100
                print(f"\n  Malignant: {mal_pct:.1f}% of features are 'high'")
                print(f"  Benign:    {ben_pct:.1f}% of features are 'high'")
            
            # Show example patterns
            if mal_length:
                print(f"\nTop-5 Malignant patterns (by lift):")
                top_mal = sorted(mal_length, key=lambda x: x['lift_malignant'], reverse=True)[:5]
                for i, p in enumerate(top_mal, 1):
                    if isinstance(p['pattern'], tuple):
                        pattern_str = ' → '.join([f'{{{item}}}' for item in p['pattern']])
                    else:
                        pattern_str = f"{{{p['pattern']}}}"
                    
                    num_high = self.count_high_features(p['pattern'])
                    high_ratio = f"{num_high}/{length_num}"
                    
                    print(f"  {i}. {pattern_str}")
                    print(f"     'high' features: {high_ratio}, "
                          f"Lift: {p['lift_malignant']:.2f}, "
                          f"M_support: {p['malignant_support']:.1%}")
            
            if ben_length:
                print(f"\nTop-5 Benign patterns (by lift):")
                top_ben = sorted(ben_length, key=lambda x: x['lift_benign'], reverse=True)[:5]
                for i, p in enumerate(top_ben, 1):
                    if isinstance(p['pattern'], tuple):
                        pattern_str = ' → '.join([f'{{{item}}}' for item in p['pattern']])
                    else:
                        pattern_str = f"{{{p['pattern']}}}"
                    
                    num_high = self.count_high_features(p['pattern'])
                    high_ratio = f"{num_high}/{length_num}"
                    
                    print(f"  {i}. {pattern_str}")
                    print(f"     'high' features: {high_ratio}, "
                          f"Lift: {p['lift_benign']:.2f}, "
                          f"B_support: {p['benign_support']:.1%}")
    
    def visualize_high_feature_distribution(self, all_results, strategy):
        """
        Create visualizations comparing 'high' feature prevalence
        """
        pattern_analysis = all_results[strategy]['pattern_analysis']
        
        # Separate patterns
        malignant_patterns = [p for p in pattern_analysis if p['discriminates'] == 'Malignant']
        benign_patterns = [p for p in pattern_analysis if p['discriminates'] == 'Benign']
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for idx, length_num in enumerate([2, 3]):
            ax = axes[idx]
            length_key = f'length_{length_num}'
            
            # Filter by length
            mal_length = [p for p in malignant_patterns if p['length'] == length_key]
            ben_length = [p for p in benign_patterns if p['length'] == length_key]
            
            if not mal_length and not ben_length:
                ax.text(0.5, 0.5, f'No length-{length_num} patterns found', 
                       ha='center', va='center', fontsize=12, transform=ax.transAxes)
                ax.set_title(f'Length-{length_num} Patterns', fontsize=14, fontweight='bold')
                ax.set_xlim(-0.5, length_num + 0.5)
                ax.set_ylim(0, 100)
                continue
            
            # Use helper method
            mal_high_counts = [self.count_high_features(p['pattern']) for p in mal_length]
            ben_high_counts = [self.count_high_features(p['pattern']) for p in ben_length]
            
            # Create distribution (percentage)
            x_positions = np.arange(length_num + 1)
            
            mal_dist = [(mal_high_counts.count(i) / len(mal_high_counts) * 100) if mal_high_counts else 0 
                        for i in range(length_num + 1)]
            ben_dist = [(ben_high_counts.count(i) / len(ben_high_counts) * 100) if ben_high_counts else 0 
                        for i in range(length_num + 1)]
            
            # Plot bars
            width = 0.35
            bars1 = ax.bar(x_positions - width/2, mal_dist, width, 
                          label=f'Malignant (n={len(mal_length)})', 
                          color='#e74c3c', alpha=0.7, edgecolor='black')
            bars2 = ax.bar(x_positions + width/2, ben_dist, width, 
                          label=f'Benign (n={len(ben_length)})', 
                          color='#2ecc71', alpha=0.7, edgecolor='black')
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.0f}%',
                               ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Formatting
            ax.set_xlabel('Number of "high" Features in Pattern', fontsize=12)
            ax.set_ylabel('Percentage of Patterns (%)', fontsize=12)
            ax.set_title(f'Length-{length_num} Patterns: "High" Feature Distribution\n({strategy.title()} Strategy)', 
                        fontsize=13, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(i) for i in range(length_num + 1)])
            ax.legend(loc='upper right')
            ax.grid(alpha=0.3, axis='y')
            ax.set_ylim(0, max(max(mal_dist + ben_dist) * 1.15, 10))
            
            # Add mean annotations
            if mal_high_counts:
                mal_mean = np.mean(mal_high_counts)
                ax.axvline(mal_mean, color='#c0392b', linestyle=':', linewidth=2, alpha=0.7)
                ax.text(mal_mean, ax.get_ylim()[1] * 0.95, f'Mal mean: {mal_mean:.2f}',
                       ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.3))
            
            if ben_high_counts:
                ben_mean = np.mean(ben_high_counts)
                ax.axvline(ben_mean, color='#27ae60', linestyle=':', linewidth=2, alpha=0.7)
                ax.text(ben_mean, ax.get_ylim()[1] * 0.85, f'Ben mean: {ben_mean:.2f}',
                       ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))
        
        plt.tight_layout()
        plt.show()