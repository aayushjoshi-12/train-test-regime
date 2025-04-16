import json
import os

import pandas as pd
import streamlit as st

# Set page config
st.set_page_config(page_title="Response Ranking App", page_icon="🏆", layout="wide")

# Constants
DATA_PATH = (
    "./data/llama3-loan-mortgage-unranked-responses.csv"  # Path to the input CSV
)
PROGRESS_PATH = "ranking_progress.json"  # Path to store progress
OUTPUT_PATH = (
    "./data/llama3-loan-mortgage-ranked-responses.csv"  # Path to store final rankings
)


def load_data():
    """Load the dataset."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        st.error(f"Data file {DATA_PATH} not found!")
        return None


def load_progress():
    """Load progress from file."""
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r") as f:
            return json.load(f)
    return {"completed": [], "current_index": 0}


def save_progress(progress):
    """Save progress to file."""
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f)


def save_rankings(rankings):
    """Save rankings to output CSV."""
    if os.path.exists(OUTPUT_PATH):
        existing_df = pd.read_csv(OUTPUT_PATH)
        updated_df = pd.concat([existing_df, pd.DataFrame(rankings)], ignore_index=True)
        updated_df.to_csv(OUTPUT_PATH, index=False)
    else:
        pd.DataFrame(rankings).to_csv(OUTPUT_PATH, index=False)


def main():
    st.title("RLHF Response Ranking")
    st.write("Rank the responses from best (1) to worst (7) for each instruction.")

    # Initialize state
    if "data" not in st.session_state:
        st.session_state.data = load_data()

    if "progress" not in st.session_state:
        st.session_state.progress = load_progress()

    if st.session_state.data is None:
        return

    data = st.session_state.data
    progress = st.session_state.progress

    # Calculate how many examples are left
    total_examples = len(data)
    completed_examples = len(progress["completed"])
    remaining_examples = total_examples - completed_examples

    # Display progress
    st.progress(completed_examples / total_examples)
    st.write(f"Progress: {completed_examples}/{total_examples} examples ranked")

    # Get current example index
    current_index = progress["current_index"]

    if current_index >= len(data):
        st.success("All examples have been ranked! Check the output file.")
        return

    # Display current example
    current_example = data.iloc[current_index]

    with st.form(key="ranking_form"):
        st.subheader("Instruction:")
        st.write(current_example["instruction"])

        st.subheader("Ground Truth:")
        st.write(current_example["ground_truth"])

        st.subheader("Rank these responses (1=best, 7=worst):")

        # Extract response columns
        response_columns = [col for col in data.columns if col.startswith("prediction")]

        # Create placeholders for rankings
        rankings = {}
        # Create a 3-column grid layout
        cols = st.columns(3)
        
        for i, response_col in enumerate(response_columns, 1):
            # Cycle through columns (i-1)%3 gives 0,1,2,0,1,2,...
            with cols[(i-1) % 3]:
                st.write(f"**Response {i}:**")
                text = current_example[response_col].split("assistant", 1)[1].strip()
                st.markdown(text)
                rankings[response_col] = st.selectbox(
                    f"Rank for Response {i}", 
                    options=list(range(1, 8)), 
                    key=f"rank_{i}"
                )
                st.divider()

        submit = st.form_submit_button("Submit Rankings")

        if submit:
            # Validate that all rankings are unique
            if len(set(rankings.values())) != 7:
                st.error("Please assign unique ranks to each response (1-7).")
                return

            # Save the rankings
            output_row = {
                "instruction": current_example["instruction"],
            }

            # Add rankings to output
            for response_col, rank in rankings.items():
                response_idx = int(response_col.replace("response", ""))
                output_row[f"rank{rank}"] = current_example[response_col]

            # Update progress
            progress["completed"].append(current_index)
            progress["current_index"] = current_index + 1

            # Save progress
            save_progress(progress)
            save_rankings([output_row])

            # Update session state
            st.session_state.progress = progress

            # Rerun the app to show next example
            st.experimental_rerun()

    # Add a skip button
    if st.button("Skip this example"):
        progress["current_index"] = current_index + 1
        save_progress(progress)
        st.session_state.progress = progress
        st.experimental_rerun()


if __name__ == "__main__":
    main()
