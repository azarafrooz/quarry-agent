import streamlit as st
import chromadb
from chromadb.config import Settings
import pandas as pd
import base64
import pickle
from dataclasses import dataclass
import plotly.express as px
import numpy as np
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap


@dataclass
class GeneratedTool:
    name: str
    description: str
    inputs: str
    output_type: str
    code: str
    dependencies: str


@st.cache_resource
def load_encoder():
    """Load and cache the sentence transformer model"""
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_embeddings(texts, encoder):
    """Generate embeddings for a list of texts"""
    return encoder.encode(texts)


def create_cluster_visualization(tools, items):
    """Create interactive cluster visualization of tools"""
    if not tools:
        return None

    # Get embeddings for tool descriptions
    encoder = load_encoder()
    descriptions = [f"{tool.name}: {tool.description}" for tool in tools]
    embeddings = get_embeddings(descriptions, encoder)

    # Reduce dimensionality with UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Create DataFrame for plotting with explicit columns for hover data
    plot_df = pd.DataFrame(
        {
            "x": reduced_embeddings[:, 0],
            "y": reduced_embeddings[:, 1],
            "Tool_Name": [tool.name for tool in tools],
            "Description": [tool.description for tool in tools],
            "ID": items["ids"],
        }
    )

    # Create interactive scatter plot with explicit hover data configuration
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        custom_data=[
            "Tool_Name",
            "Description",
            "ID",
        ],  # Include all data needed for hover
        title="Tool Clusters (Click a point to select tool for editing)",
    )

    # Customize hover template using custom_data indices
    fig.update_traces(
        marker=dict(size=10),
        hovertemplate=(
            "<b>Tool:</b> %{customdata[0]}<br>"
            + "<b>Description:</b> %{customdata[1]}<br>"
            + "<b>ID:</b> %{customdata[2]}<br>"
            + "<extra></extra>"
        ),
    )

    # Update layout
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        xaxis_showticklabels=False,
        yaxis_showticklabels=False,
        hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial"),
    )

    return fig


def initialize_chroma():
    """Initialize ChromaDB client"""
    chroma_client = chromadb.PersistentClient(
        path="chroma_dir", settings=Settings(anonymized_telemetry=False)
    )
    return chroma_client


def display_collections(client):
    """Display and let user select a collection"""
    collections = client.list_collections()
    if not collections:
        st.warning("No collections found in the database.")
        return None

    collection_names = [col.name for col in collections]
    selected_collection = st.selectbox("Select Collection", collection_names)
    return client.get_collection(selected_collection)


def decode_tools(encoded_tools):
    """Decode base64 encoded pickled tools"""
    return [pickle.loads(base64.b64decode(tool)) for tool in encoded_tools]


def encode_tool(tool):
    """Encode tool to base64 encoded pickled format"""
    return base64.b64encode(pickle.dumps(tool)).decode("utf-8")


def display_tool_editor(tool):
    """Display editor for a single tool"""
    edited_tool = GeneratedTool(
        name=st.text_input("Name", tool.name),
        description=st.text_area("Description", tool.description),
        inputs=st.text_area("Inputs", tool.inputs),
        output_type=st.text_input("Output Type", tool.output_type),
        code=st.text_area("Code", tool.code, height=300),
        dependencies=st.text_area("Dependencies", tool.dependencies),
    )
    return edited_tool


def display_collection_items(collection):
    """Display collection items with editing and deletion options"""
    if not collection:
        return

    # Get all items from collection
    items = collection.get()

    if not items["ids"]:
        st.warning("No items in this collection.")
        return

    # Decode tools
    tools = decode_tools(items["documents"])

    # Create initial DataFrame for tool selection
    df = pd.DataFrame(
        {
            "ID": items["ids"],
            "Name": [tool.name for tool in tools],
            "Description": [tool.description for tool in tools],
        }
    )

    # Create and display clustering visualization
    st.write("### Tool Clusters")
    fig = create_cluster_visualization(tools, items)

    if fig:
        selected_point = plotly_chart_with_selection(fig)
        if selected_point:
            st.session_state.selected_tool_id = selected_point["customdata"][0][0]

    # Display tools table
    st.write("### Available Tools:")
    selected_row = st.data_editor(
        df,
        hide_index=True,
        column_config={
            "ID": st.column_config.TextColumn("ID", width="medium"),
            "Name": st.column_config.TextColumn("Name", width="medium"),
            "Description": st.column_config.TextColumn("Description", width="large"),
        },
        key="tool_selector",
    )

    # Tool editing section
    st.write("### Edit Tool")

    # Initialize selected_tool_id if not set
    if (
        "selected_tool_id" not in st.session_state
        or st.session_state.selected_tool_id not in items["ids"]
    ):
        st.session_state.selected_tool_id = items["ids"][0]

    selected_id = st.selectbox(
        "Select tool to edit",
        items["ids"],
        key="tool_select_box",
        index=items["ids"].index(st.session_state.selected_tool_id),
    )

    if selected_id:
        tool_idx = items["ids"].index(selected_id)
        tool = tools[tool_idx]

        with st.form(key=f"edit_form_{selected_id}"):
            edited_tool = display_tool_editor(tool)

            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Update Tool"):
                    try:
                        # Encode the updated tool
                        encoded_tool = encode_tool(edited_tool)

                        # Update the tool in the collection
                        collection.update(
                            ids=[selected_id],
                            documents=[encoded_tool],
                            metadatas=[{"type": "tool"}],
                        )
                        st.success("Tool updated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error updating tool: {str(e)}")

            with col2:
                if st.form_submit_button("Delete Tool", type="primary"):
                    try:
                        collection.delete(ids=[selected_id])
                        st.success("Tool deleted successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting tool: {str(e)}")


def plotly_chart_with_selection(fig):
    """Display Plotly chart and handle click events"""
    # Add JavaScript callback for click events
    fig.update_layout(clickmode="event+select")

    # Create empty placeholder for click data
    click_data = st.empty()

    # Display the plot
    chart = st.plotly_chart(fig, use_container_width=True)

    # Return click data if available
    return st.session_state.get("plotly_click_data", None)


def main():
    st.title("Tool Collection Manager")

    # Initialize ChromaDB client
    client = initialize_chroma()

    # Get selected collection
    collection = display_collections(client)

    # Display tools with edit/delete functionality
    display_collection_items(collection)


if __name__ == "__main__":
    main()
