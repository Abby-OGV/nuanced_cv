import re
import os
import argparse
import pandas as pd

from convokit import Corpus, download

from utils import CATEGORY_FUNCTIONS, is_valid_claim

def strip_isitbullshit(pattern, text):
    return pattern.sub('', text).strip()

def extract_based_on_category(category_name, pattern_func, df):
    if "claim" not in df.columns:
        raise ValueError("DataFrame must have a claim column")
    
    os.makedirs("data", exist_ok=True)

    results = df['claim'].apply(lambda text: pattern_func(text))#.apply(pattern_func)
    
    mask = results.apply(lambda x: x[0])
    df['type'] = results.apply(lambda x: x[1])
    
    matching_claims = df[mask].copy()[["conv_id", "claim", "type"]]

    csv_filename = f"data/{category_name}.csv"
    matching_claims.to_csv(csv_filename, index=False)
    
    print(f"Saved {len(matching_claims)} claims to {csv_filename}")
    return matching_claims

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True,
                        choices=list(CATEGORY_FUNCTIONS.keys()) + ["all"],
                        help="Which category to extract, or all categories")
    args = parser.parse_args()

    isItBull=Corpus(filename=download("subreddit-IsItBullshit"))
    pattern = re.compile(
        r'^(?:\[Meta\]\s*|")?'
        r'(?:'
        r'(?:(is)?(\s)?it(\s)?bull+s*hit'
        r'|IIBS'
        r'|IIB'
        r'|IsItBS'
        r'|ItIsBullshit'
        r'|bullshit'
        r')[.:?!"\s]*)+',
        re.IGNORECASE
    )

    data_list = []
    for conv in isItBull.iter_conversations():
        claim = strip_isitbullshit(pattern, conv.meta['title'])
        if len(claim) > 0:
            data_list.append([conv.id, claim])
        # else:
        #     print(conv.meta['title'])

    df = pd.DataFrame(data=data_list, columns=['conv_id', 'claim']) 
    df['valid_claim'] = df['claim'].apply(is_valid_claim)
    filtered_df = df[df['valid_claim'] == True].reset_index(drop=True)

    print(f"After initial filtering, the dataset is {len(filtered_df)} out of {len(df)}")
    
    if args.category != "all":
        pattern_func = CATEGORY_FUNCTIONS[args.category]
        extract_based_on_category(args.category, pattern_func, filtered_df)
    else:
        for category, pattern_func in CATEGORY_FUNCTIONS.items():
            extract_based_on_category(category, pattern_func, filtered_df)

if __name__ == "__main__":
    main()