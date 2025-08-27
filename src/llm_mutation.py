import re
import time
import glob
import yaml
import numpy as np
import transformers
from torch import bfloat16
import argparse
from cfg.constants import *
from utils.print_utils import box_print
from llm_utils import (split_file, submit_mixtral, submit_mixtral_hf, 
                       llm_code_qc, str2bool, generate_augmented_code, 
                       extract_note, clean_code_from_llm, retrieve_base_code)



"""
 █▀▀█  █  █  █▀▀█  █▄  █  █▀▀█  █▀▀▀  █▀▀▄ 
 █     █▀▀█  █▄▄█  █ █ █  █ ▄▄  █▀▀▀  █  █ 
 █▄▄█  █  █  █  █  █  ▀█  █▄▄█  █▄▄▄  █▄▄▀
"""
def augment_network(input_filename='network.py', configuration_filename='config.yaml', output_filename='network_x.py', template_txt=None,
                    top_p=0.15, temperature=0.1, apply_quality_control=False, inference_submission=False):
    
    print(f'Loading {input_filename} code')

    parts = split_file(input_filename)

    augment_idx = np.random.randint(1, len(parts))

    # select code to be augmented randomly 
    code2llm = parts[augment_idx]

    # prompt_templates = glob.glob(f'{ROOT_DIR}/templates/FixedPrompts/*/*.txt')
    # template_path = np.random.choice(prompt_templates)
    # template_path = f'{ROOT_DIR}/templates/{fname}'

    fname = template_txt

    with open(fname, 'r') as file:
        template_txt = file.read()

    configuration_info = yaml.dump(CONFIG_FILE, default_flow_style=False, indent=4, width=80)

    # add code to be augmented to the template_txt

    # Now we're giving the LLM the configuration info as well to fruther assit with the model evolution
    txt2llm = template_txt.format(code2llm.strip(), configuration_info) 


    code_from_llm = generate_augmented_code(txt2llm, augment_idx-1, apply_quality_control,
                                            top_p, temperature, inference_submission=inference_submission)
    
    note_txt = extract_note(code2llm)

    parts[augment_idx] = f"\n{note_txt}{code_from_llm}\n"


    # prompt_log = f'# Parent Prompt: {template_path} Root Code: {input_filename}\n'
    # python_network_txt = prompt_log + '# --OPTION--'.join(parts)
    python_network_txt = '# --OPTION--'.join(parts)

    # Write the text to the file
    with open(output_filename, 'w') as file:
        file.write(python_network_txt)
        
    box_print(f"Python code saved to {os.path.basename(output_filename)}", print_bbox_len=120, new_line_end=False)

    print('Job Done')




"""
 █▀▀█  █  █  █▀▀█  █▄  █  █▀▀█  █▀▀▀  █▀▀▄ 
 █     █▀▀█  █▄▄█  █ █ █  █ ▄▄  █▀▀▀  █  █ 
 █▄▄█  █  █  █  █  █  ▀█  █▄▄█  █▄▄▄  █▄▄▀
"""
if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description='Augment Python Network Script.')

    # Add arguments
    # I need to add an argument for model config file

    parser.add_argument('input_filename', type=str, help='Input file name')

    # Added a Configuration File Argument So users can input model file configuration changes 
    parser.add_argument('config_filename', type=str, help="Config file name")

    parser.add_argument('output_filename', type=str, help='Output file name')
    parser.add_argument('template_txt', type=str, help='Template txt')
    parser.add_argument('--top_p', type=float, default=0.15, help='Top P value for text generation')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature value for text generation')
    parser.add_argument('--apply_quality_control', type=str2bool, default=False, help='Use LLM QC')
    parser.add_argument('--inference_submission', type=str2bool, default=False, help='True to submit for inference remotely')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    augment_network(input_filename=args.input_filename,
                    configuration_filename=args.config_filename, # Added Configuration File Path For LLM to Make Changes to this
                    output_filename=args.output_filename,
                    template_txt=args.template_txt,
                    top_p=args.top_p, 
                    temperature=args.temperature,
                    apply_quality_control=args.apply_quality_control,
                    inference_submission=args.inference_submission,
                   )