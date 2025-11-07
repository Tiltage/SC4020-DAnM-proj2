"""
Diagnostic script to check your cancer data
Run this first to see what's wrong
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("CANCER DATA DIAGNOSTIC TOOL")
print("=" * 70)

# 1. Try to load the file
print("\n1. LOADING FILE...")
try:
    df = pd.read_csv('../RawData/cancer_data.csv')
    print(f"✓ Successfully loaded file")
    print(f"  Shape: {df.shape}")
except Exception as e:
    print(f"✗ ERROR loading file: {e}")
    exit(1)

# 2. Check columns
print("\n2. CHECKING COLUMNS...")
print(f"  Columns found: {list(df.columns)}")

if 'id' not in df.columns:
    print("  ⚠ WARNING: No 'id' column found")
else:
    print(f"  ✓ 'id' column present")

if 'diagnosis' not in df.columns:
    print("  ⚠ WARNING: No 'diagnosis' column found")
    print("  Available columns:", list(df.columns))
else:
    print(f"  ✓ 'diagnosis' column present")
    print(f"  Diagnosis values: {df['diagnosis'].unique()}")
    print(f"  Diagnosis counts:\n{df['diagnosis'].value_counts()}")

# 3. Check data types
print("\n3. CHECKING DATA TYPES...")
print(df.dtypes)

# 4. Check for missing values
print("\n4. CHECKING FOR MISSING VALUES...")
missing = df.isnull().sum()
if missing.sum() > 0:
    print("  ⚠ WARNING: Found missing values:")
    print(missing[missing > 0])
else:
    print("  ✓ No missing values")

# 5. Check feature columns
print("\n5. CHECKING FEATURE COLUMNS...")
feature_cols = [col for col in df.columns if col not in ['id', 'diagnosis']]
print(f"  Number of features: {len(feature_cols)}")
print(f"  Feature columns: {feature_cols[:5]}... (showing first 5)")

# Check if features are numeric
non_numeric = []
for col in feature_cols:
    if not pd.api.types.is_numeric_dtype(df[col]):
        non_numeric.append(col)

if non_numeric:
    print(f"  ⚠ WARNING: Non-numeric feature columns found: {non_numeric}")
else:
    print(f"  ✓ All features are numeric")

# 6. Show sample data
print("\n6. SAMPLE DATA (first 5 rows):")
print(df.head())

# 7. Basic statistics
print("\n7. FEATURE STATISTICS:")
print(df[feature_cols[:5]].describe())

# 8. Test transformation
print("\n8. TESTING SEQUENCE TRANSFORMATION...")
try:
    from cancer_pattern_mining import SequenceTransformer
    
    transformer = SequenceTransformer(top_k=5, max_length=3, max_gap=1)
    sequences = transformer.fit_transform(df)
    
    print(f"  ✓ Created {len(sequences)} sequences")
    
    # Check if sequences have content
    empty_seqs = [s for s in sequences if len(s.get('sequence', [])) == 0]
    if empty_seqs:
        print(f"  ⚠ WARNING: {len(empty_seqs)} sequences are empty!")
    else:
        print(f"  ✓ All sequences have content")
    
    # Show example
    print("\n  Example sequences:")
    for i, seq in enumerate(sequences[:3]):
        seq_str = ' → '.join(['{' + ','.join(itemset) + '}' for itemset in seq['sequence']])
        print(f"    {i+1}. {seq['diagnosis']}: {seq_str}")
        
except Exception as e:
    print(f"  ✗ ERROR in transformation: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)