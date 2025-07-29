#!/usr/bin/env python3
"""
Generate a visual flowchart showing how the knowledge graph extraction works.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_system_diagram():
    """Create a comprehensive diagram of the knowledge graph extraction system."""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    input_color = '#E3F2FD'
    extract_color = '#FFF3E0'
    build_color = '#E8F5E8'
    output_color = '#FCE4EC'
    
    # Title
    ax.text(5, 9.5, 'Knowledge Graph Extraction System', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Input Phase
    input_box = FancyBboxPatch((0.5, 8), 2, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor=input_color, 
                              edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 8.4, 'Input Text\n(Image Metadata)', 
            ha='center', va='center', fontweight='bold')
    
    # Text Splitting
    split_box = FancyBboxPatch((3.5, 8), 2, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor=input_color, 
                              edgecolor='black', linewidth=1)
    ax.add_patch(split_box)
    ax.text(4.5, 8.4, 'Text Splitting\n(4096 char chunks)', 
            ha='center', va='center')
    
    # Phase 1: Extract
    extract_title = FancyBboxPatch((0.5, 6.5), 9, 0.3, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=extract_color, 
                                  edgecolor='black', linewidth=2)
    ax.add_patch(extract_title)
    ax.text(5, 6.65, 'PHASE 1: EXTRACT - Triplet Generation', 
            ha='center', va='center', fontweight='bold', fontsize=14)
    
    # LLM Extraction
    llm_extract_box = FancyBboxPatch((1, 5.5), 2.5, 0.8, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=extract_color, 
                                    edgecolor='black', linewidth=1)
    ax.add_patch(llm_extract_box)
    ax.text(2.25, 5.9, 'LLM Extractor\n(Gemini 2.0)', 
            ha='center', va='center', fontweight='bold')
    
    # Prompt box
    prompt_box = FancyBboxPatch((4, 5.5), 2, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#FFFDE7', 
                               edgecolor='black', linewidth=1)
    ax.add_patch(prompt_box)
    ax.text(5, 5.9, 'System Prompt +\nEntity Types', 
            ha='center', va='center')
    
    # Raw triplets
    raw_triplets_box = FancyBboxPatch((6.5, 5.5), 2.5, 0.8, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor=extract_color, 
                                     edgecolor='black', linewidth=1)
    ax.add_patch(raw_triplets_box)
    ax.text(7.75, 5.9, 'Raw Triplets\n(alex:PERSON, ...)', 
            ha='center', va='center')
    
    # Validation
    validation_box = FancyBboxPatch((3.5, 4.2), 3, 0.8, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor='#FFF8E1', 
                                   edgecolor='black', linewidth=1)
    ax.add_patch(validation_box)
    ax.text(5, 4.6, 'Validation & Parsing\n(Regex + Type Check)', 
            ha='center', va='center')
    
    # Phase 2: Build
    build_title = FancyBboxPatch((0.5, 3.2), 9, 0.3, 
                                boxstyle="round,pad=0.1", 
                                facecolor=build_color, 
                                edgecolor='black', linewidth=2)
    ax.add_patch(build_title)
    ax.text(5, 3.35, 'PHASE 2: BUILD - Graph Construction', 
            ha='center', va='center', fontweight='bold', fontsize=14)
    
    # Similarity search
    similarity_box = FancyBboxPatch((0.5, 2.2), 2, 0.8, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=build_color, 
                                   edgecolor='black', linewidth=1)
    ax.add_patch(similarity_box)
    ax.text(1.5, 2.6, 'Similarity Search\n(FAISS + Embeddings)', 
            ha='center', va='center')
    
    # Builder LLM
    builder_box = FancyBboxPatch((3, 2.2), 2, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor=build_color, 
                                edgecolor='black', linewidth=1)
    ax.add_patch(builder_box)
    ax.text(4, 2.6, 'Builder LLM\nValidation', 
            ha='center', va='center', fontweight='bold')
    
    # Decision
    decision_box = FancyBboxPatch((5.5, 2.2), 1.8, 0.8, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='#F3E5F5', 
                                 edgecolor='black', linewidth=1)
    ax.add_patch(decision_box)
    ax.text(6.4, 2.6, 'Decision:\nAccept/Modify/Reject', 
            ha='center', va='center')
    
    # Graph integration
    integration_box = FancyBboxPatch((7.5, 2.2), 2, 0.8, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=build_color, 
                                    edgecolor='black', linewidth=1)
    ax.add_patch(integration_box)
    ax.text(8.5, 2.6, 'Graph Integration\n+ Source Linking', 
            ha='center', va='center')
    
    # Output
    output_title = FancyBboxPatch((0.5, 1), 9, 0.3, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=output_color, 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(output_title)
    ax.text(5, 1.15, 'OUTPUT - Knowledge Graph', 
            ha='center', va='center', fontweight='bold', fontsize=14)
    
    # Output formats
    json_box = FancyBboxPatch((1, 0.1), 2, 0.6, 
                             boxstyle="round,pad=0.1", 
                             facecolor=output_color, 
                             edgecolor='black', linewidth=1)
    ax.add_patch(json_box)
    ax.text(2, 0.4, 'JSON Data\n24 triplets', 
            ha='center', va='center')
    
    graph_box = FancyBboxPatch((4, 0.1), 2, 0.6, 
                              boxstyle="round,pad=0.1", 
                              facecolor=output_color, 
                              edgecolor='black', linewidth=1)
    ax.add_patch(graph_box)
    ax.text(5, 0.4, 'NetworkX Graph\nVisualization', 
            ha='center', va='center')
    
    png_box = FancyBboxPatch((7, 0.1), 2, 0.6, 
                            boxstyle="round,pad=0.1", 
                            facecolor=output_color, 
                            edgecolor='black', linewidth=1)
    ax.add_patch(png_box)
    ax.text(8, 0.4, 'PNG Image\nGraph Plot', 
            ha='center', va='center')
    
    # Arrows
    arrows = [
        # Input flow
        ((2.5, 8.4), (3.5, 8.4)),
        ((4.5, 8.0), (4.5, 6.8)),
        
        # Extract phase
        ((2.25, 5.5), (2.25, 5.0)),
        ((4, 5.9), (3.5, 5.9)),
        ((6, 5.9), (6.5, 5.9)),
        ((7.75, 5.5), (7.75, 5.0)),
        ((5, 4.2), (5, 3.5)),
        
        # Build phase  
        ((1.5, 2.2), (1.5, 1.8)),
        ((2.5, 2.6), (3, 2.6)),
        ((5, 2.6), (5.5, 2.6)),
        ((7.3, 2.6), (7.5, 2.6)),
        ((8.5, 2.2), (8.5, 1.8)),
        
        # To outputs
        ((3, 1.0), (3, 0.7)),
        ((5, 1.0), (5, 0.7)),
        ((7, 1.0), (7, 0.7)),
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=0, shrinkB=0,
                              mutation_scale=20, fc="black", lw=2)
        ax.add_patch(arrow)
    
    # Add example triplet
    ax.text(0.5, 7.2, 'Example Input:\n"Alex graduated from\nColumbia University"', 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    ax.text(8.5, 7.2, 'Example Output:\n(alex:PERSON,\ngraduated from,\ncolumbia university:LOCATION)', 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('system_architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_system_diagram()
    print("✅ System architecture diagram saved as 'system_architecture_diagram.png'")
