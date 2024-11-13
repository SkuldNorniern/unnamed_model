import sys
from port_nanogpt.sample import sample
from port_nanogpt.train import train
from port_nanogpt.tuner import tuner
from port_nanogpt.inference import inference

def main() -> int:
    # get the subcommand for train or inference
    if len(sys.argv) < 2:
        print('Invalid subcommand')
        return 1
    
    subcommand = sys.argv[1]
    dataset_name = "1M-GPT4-Augmented"
    
    if subcommand == 'inference':
        model_path = "out/model"
        prompt = None
        
        # Parse additional arguments
        if len(sys.argv) >= 3:
            model_path = sys.argv[2]
        if len(sys.argv) >= 4:
            prompt = sys.argv[3]
            
        inference(model_path, prompt)
        return 0
        
    if len(sys.argv) >= 3:
        dataset_name = sys.argv[2]
    
    if subcommand == 'train':
        train(dataset_name)
    elif subcommand == 'sample':
        sample()
    elif subcommand == 'tuner':
        tuner(dataset_name)
    else:
        print('Invalid subcommand')
        print(f"Usage: python3 {sys.argv[0]} train|sample|tuner|inference [model_path] [prompt]")
        return 1

    return 0