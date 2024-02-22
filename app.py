import streamlit as st
import pandas as pd
import RAG_MTG_Cards as RMTG

raw_df = pd.read_csv("raw/filtered_oracle_database.csv")

def main():
    st.title("Card Generator App")

    option = st.radio("Select Input Option", ("User Input", "Random Selection"))

    if option == "User Input":
        card_name = st.text_input("Enter Card Name:")
        if st.button("Generate"):
            original_row = raw_df[raw_df['name'] == card_name]
            if not original_row.empty:
                original_card_col, rag_card_col, non_rag_card_col = st.columns((1, 2, 1))
                with original_card_col:
                    display_card_info(original_row.iloc[0], "Original Card")
                with rag_card_col:
                    mirrored_info_rag = get_mirrored_card_info(card_name, RAG=True)
                    display_card_info(mirrored_info_rag, "RAG Card")
                with non_rag_card_col:
                    mirrored_info_non_rag = get_mirrored_card_info(card_name, RAG=False)
                    display_card_info(mirrored_info_non_rag, "Non-RAG Card")
            else:
                st.warning("Card not found in the database.")
    else:
        if st.button("Generate Random Card"):
            random_card_row = raw_df.sample(n=1)
            original_card_col, rag_card_col, non_rag_card_col = st.columns((1, 1, 1))
            with original_card_col:
                display_card_info(random_card_row.iloc[0], "Original Card")
            with rag_card_col:
                mirrored_info_rag = get_mirrored_card_info(random_card_row.iloc[0]['name'], RAG=True)
                display_card_info(mirrored_info_rag, "RAG Model Card")
            with non_rag_card_col:
                mirrored_info_non_rag = get_mirrored_card_info(random_card_row.iloc[0]['name'], RAG=False)
                display_card_info(mirrored_info_non_rag, "Baseline Model Card")

def get_mirrored_card_info(card_name, RAG=True):
    query_response = RMTG.rag_query(card_name, RAG=RAG)
    mirrored_info = {}
    for line in query_response.split('\n'):
        if line:
            try:
                key, value = line.split(': ')
                mirrored_info[key.strip()] = value.strip()
            except ValueError:
                continue
    return mirrored_info

def display_card_info(info, display_name):
    st.subheader(display_name)
    for key, value in info.items():        
        st.write(f"{key}: {value}")

if __name__ == "__main__":
    main()