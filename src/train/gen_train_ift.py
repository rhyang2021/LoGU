import pandas as pd
import pdb
pd.options.mode.chained_assignment = None
from transformers import AutoTokenizer

instruction_pool={
    "bio": f"Write a biography for a specific entity. Your response should be as detailed as possible, and express uncertainty for any information you are not familiar with (e.g., 'I am not sure if/whether', 'It is uncertain that').",
    "wild": f"Write a paragraph for a specific entity. Your response should be as detailed as possible, and express uncertainty for any information you are not familiar with (e.g., 'I am not sure if/whether', 'It is uncertain that').",
    "longfact": f"Given a question about a specific object (e.g., a person, place, event, company, etc.), generate a comprehensive answer covering all relevant aspects of the question. Express uncertainty for any information you are not familiar with (e.g., 'I am not sure if/whether', 'It is uncertain that')."
}


def get_template(dataset, label):
    output = []
    for data in dataset:
        for prompt in data["prompt"]:
            for refine_answer in data['refine_answer']:
                output.append(
                    {"instruction": prompt,
                    "input": "",
                    "output": refine_answer
                    })

    
    return output


if __name__ == "__main__":
    import json
    from tqdm import tqdm
    import argparse
    
    parser = argparse.ArgumentParser(description="generate closed set fasts for each entity in bio dataset.")
    parser.add_argument("--model_id", type=str, default="mistral-7b", help="Model ID for generating")
    parser.add_argument("--method", type=str, default="sft", help="Model ID for generating")
    parser.add_argument("--input_dir", type=str, default="../long_uncertainty_express/sft_data", help="Model ID for generating")
    parser.add_argument("--output_dir", type=str, default="../long_uncertainty_express/sft_data", help="Model ID for generating")
    args = parser.parse_args()

    final_data = []
    for label in ['bio', 'wild', 'longfact']:
        with open(f'{args.input_dir}/{label}/{args.model_id}_train_{args.method}.json') as f:
            dataset = json.load(f)
        
        final_data += get_template(dataset, label)
            
        print(len(final_data))
    
    with open(f"{args.output_dir}/uncertain_{args.method}_{args.model_id}.json", 'w') as f:
        json.dump(final_data, f, indent=4)