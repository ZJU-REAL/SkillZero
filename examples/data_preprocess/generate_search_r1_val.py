import pandas as pd
import os
import argparse

def assign_skill_type(row):
    data_source = str(row.get('data_source', ''))
    
    # Extract text content from prompt
    prompt = row.get('prompt', '')
    if isinstance(prompt, list):
        # prompt is in list of dict format, concatenate all content
        prompt_text = ' '.join(
            msg.get('content', '') if isinstance(msg, dict) else str(msg)
            for msg in prompt
        )
    else:
        prompt_text = str(prompt)
    prompt_lower = prompt_text.lower()

    # Rule-based classification
    if data_source == 'popqa':
        return 'entity_attribute_lookup'
    elif data_source in ('nq', 'triviaqa'):
        return 'direct_retrieval'
    elif 'which' in prompt_lower and 'or' in prompt_lower and 'for' not in prompt_lower:
        return 'compare'
    elif data_source == 'hotpotqa':
        return 'multi_hop_reasoning'
    else:
        return 'unknown'


def main():
    parser = argparse.ArgumentParser(description='Generate validation parquet file with skill_type column')
    parser.add_argument('--max_sample', type=int, default=1000,
                        help='Maximum number of samples to keep per skill_type (default: 1000)')
    parser.add_argument('--input', type=str,
                        default=None,
                        help='Input parquet file path (default: ~/data/searchR1_processed_direct/test.parquet)')
    parser.add_argument('--output_dir', type=str,
                        default=None,
                        help='Output directory (default: ~/data/searchR1_processed_direct)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    _default_search_data = os.path.join(os.path.expanduser('~'), 'data/searchR1_processed_direct')
    if args.input is None:
        args.input = os.path.join(_default_search_data, 'test.parquet')
    if args.output_dir is None:
        args.output_dir = _default_search_data

    # 1. Load data
    input_path = os.path.expanduser(args.input)
    print(f"Loading data: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Original data size: {len(df)} samples")

    # 2. Add skill_type column
    df['skill_type'] = df.apply(assign_skill_type, axis=1)

    # 3. Print skill_type distribution
    print("\n[Original] skill_type distribution:")
    print(df['skill_type'].value_counts().to_string())

    # 4. Keep at most max_sample per skill_type (random sampling)
    sampled_parts = []
    for skill, group in df.groupby('skill_type'):
        if skill == 'unknown':
            print(f"  [unknown] {len(group)} samples -> skipped")
            continue
        if len(group) > args.max_sample:
            sampled = group.sample(n=args.max_sample, random_state=args.seed)
            print(f"  [{skill}] {len(group)} -> sampled {args.max_sample} samples")
        else:
            sampled = group
            print(f"  [{skill}] {len(group)} samples (under limit, all kept)")
        sampled_parts.append(sampled)

    df_val = pd.concat(sampled_parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # 5. Print distribution after sampling
    print(f"\n[After sampling] Total data size: {len(df_val)} samples")
    print("skill_type distribution:")
    print(df_val['skill_type'].value_counts().to_string())

    # 6. Save
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'val_{args.max_sample}.parquet')
    df_val.to_parquet(output_path, index=False)
    print(f"\n✅ Saved to: {output_path}")


if __name__ == '__main__':
    main()