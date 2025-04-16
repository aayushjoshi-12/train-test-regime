import streamlit as st
import pandas as pd
import itertools
import json
import os

st.set_page_config(page_title="Response Ranker", page_icon=":star:", layout="wide")

# TODO: remember to remove the cases from the output where both choices are same
# maybe remove the duplicates too
# remove 6th, 13th, 36th, 45th, 50th row data because all are wrong
# most cases all the responses are wrong compared to the ground truth selecting them two of them would be just confusing the reward model
# i think the same category of responses prompt should have similar steps. why in some case they suggest something else and in some case they suggest something else (read into it)
# our training dataset has an issue. why are the questions that are asked again (mainly of the same category have different response for the same question which should be same process)

INPUT_CSV = "./data/llama3-loan-mortgage-unranked-responses.csv"
OUTPUT_CSV = "./data/llama3-loan-mortgage-ranked-responses.csv"
CHECKPOINT_FILE = "./checkpoint.json"

# Load checkpoint if exists
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"current_row": 0, "completed_pairs": [], "output_data": []}

# Save checkpoint
def save_checkpoint(current_row, completed_pairs, output_data):
    checkpoint = {
        "current_row": current_row,
        "completed_pairs": completed_pairs,
        "output_data": output_data
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

def format_response(response: str) -> str:
    return response.split('assistant')[1].strip()

# Initialize session state
if 'checkpoint' not in st.session_state:
    st.session_state.checkpoint = load_checkpoint()
if 'df' not in st.session_state:
    st.session_state.df = pd.read_csv(INPUT_CSV)
if 'output_data' not in st.session_state:
    st.session_state.output_data = st.session_state.checkpoint["output_data"]
if 'current_row' not in st.session_state:
    st.session_state.current_row = st.session_state.checkpoint["current_row"]
if 'completed_pairs' not in st.session_state:
    st.session_state.completed_pairs = st.session_state.checkpoint["completed_pairs"]
if 'current_pairs' not in st.session_state:
    if st.session_state.current_row < len(st.session_state.df):
        row = st.session_state.df.iloc[st.session_state.current_row]
        responses = row.iloc[3:].tolist()
        st.session_state.current_pairs = list(itertools.combinations(range(len(responses)), 2))
        st.session_state.responses = responses
    else:
        st.session_state.current_pairs = []
        st.session_state.responses = []
    
def save_preference(choice, choice_1_idx, choice_2_idx):
    row = st.session_state.df.iloc[st.session_state.current_row]
    instruction = row['instruction']
    ground_truth = row['ground_truth']
    
    choice_1 = format_response(st.session_state.responses[choice_1_idx])
    choice_2 = format_response(st.session_state.responses[choice_2_idx])
    
    if choice == 1:
        choice_w = choice_1
        choice_l = choice_2
    else:
        choice_w = choice_2
        choice_l = choice_1
    
    st.session_state.output_data.append({
        'instruction': instruction,
        'ground_truth': ground_truth,
        'choice_1': choice_1,
        'choice_2': choice_2,
        'choice_w': choice_w,
        'choice_l': choice_l
    })
    
    pair_key = f"{choice_1_idx}-{choice_2_idx}"
    st.session_state.completed_pairs.append(pair_key)
    
    # If all pairs are completed for current row, move to next row
    if len(st.session_state.completed_pairs) >= len(st.session_state.current_pairs):
        st.session_state.current_row += 1
        if st.session_state.current_row < len(st.session_state.df):
            row = st.session_state.df.iloc[st.session_state.current_row]
            responses = row.iloc[3:].tolist()
            st.session_state.current_pairs = list(itertools.combinations(range(len(responses)), 2))
            st.session_state.responses = responses
            st.session_state.completed_pairs = []
    
    # Save checkpoint
    save_checkpoint(st.session_state.current_row, st.session_state.completed_pairs, st.session_state.output_data)
    
    # If all rows are processed, save to CSV
    if st.session_state.current_row >= len(st.session_state.df):
        output_df = pd.DataFrame(st.session_state.output_data)
        output_df.to_csv(OUTPUT_CSV, index=False)
        st.success(f"All responses have been ranked and saved to {OUTPUT_CSV}")

# Streamlit UI
st.title("Response Ranker")

st.write(f"Progress: Row {st.session_state.current_row + 1} of {len(st.session_state.df)}")
progress = st.progress(st.session_state.current_row / len(st.session_state.df))

if st.session_state.current_row < len(st.session_state.df):
    row = st.session_state.df.iloc[st.session_state.current_row]
    instruction = row['instruction']
    ground_truth = row['ground_truth']
    
    st.subheader("Instruction")
    st.write(instruction)

    st.subheader("Ground Truth")
    st.write(ground_truth)

    
    # Find the next pair that hasn't been completed
    for i, (choice_1_idx, choice_2_idx) in enumerate(st.session_state.current_pairs):
        pair_key = f"{choice_1_idx}-{choice_2_idx}"
        if pair_key not in st.session_state.completed_pairs:
            choice_1 = format_response(st.session_state.responses[choice_1_idx])
            choice_2 = format_response(st.session_state.responses[choice_2_idx])

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Response 1")
                st.write(choice_1)
                if st.button("Choose Response 1"):
                    save_preference(1, choice_1_idx, choice_2_idx)
                    st.rerun()
            with col2:
                st.subheader("Response 2")
                st.write(choice_2)
                if st.button("Choose Response 2"):
                    save_preference(2, choice_1_idx, choice_2_idx)
                    st.rerun()
            break
    else:
        st.session_state.current_row += 1
        save_checkpoint(st.session_state.current_row, [], st.session_state.output_data)
        st.rerun()
else:
    st.success(f"All responses have been ranked and saved to {OUTPUT_CSV}")

# Save button to manually save progress
if st.button("Save Progress"):
    save_checkpoint(st.session_state.current_row, st.session_state.completed_pairs, st.session_state.output_data)
    st.success("Progress saved!")
