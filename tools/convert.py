#!/usr/bin/env python3
import argparse
import json
import os
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert hypergraph text data to JSON format.")
    parser.add_argument(
        "input_file",
        nargs="?",
        default="original.txt",
        help="Path to the input text file (default: original.txt)"
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default="original.json",
        help="Path to the output JSON file (default: original.json)"
    )
    return parser.parse_args()

def clean_label(label_str):
    # Remove surrounding spaces, double quotes, and single quotes
    return label_str.strip().strip('"').strip("'").strip()

def parse_node_line(line, line_num):
    parts = [p.strip() for p in line.split(',')]
    if len(parts) < 4:
        raise ValueError(f"Line {line_num} has too few values: '{line}'. Expected id, label, x, y.")
        
    try:
        node_id = int(parts[0])
        x = float(parts[-2])
        y = float(parts[-1])
    except ValueError as e:
        raise ValueError(f"Line {line_num} has invalid numeric values: '{line}'. {e}")
        
    # Reconstruct label in case it contains commas
    label_raw = ",".join(parts[1:-2])
    label = clean_label(label_raw)
    
    return {
        "id": node_id,
        "label": label,
        "x": x,
        "y": y
    }

def parse_hyperedge_line(line, line_num):
    parts = [p.strip() for p in line.split(',')]
    if len(parts) < 5:
        raise ValueError(f"Line {line_num} has too few values: '{line}'. Expected id, r, g, b, label, ...")
        
    try:
        he_id = int(parts[0])
        r = int(parts[1])
        g = int(parts[2])
        b = int(parts[3])
    except ValueError as e:
        raise ValueError(f"Line {line_num} has invalid numeric values (id or RGB): '{line}'. {e}")
        
    remaining = parts[4:]
    
    # Scan from left to right to find the 'num_nodes' field (integer N)
    # such that the subsequent N elements are all integers.
    found = False
    num_nodes = 0
    node_ids = []
    label_parts = []
    
    for c_idx in range(len(remaining)):
        val_str = remaining[c_idx]
        try:
            N = int(val_str)
            if N < 0:
                continue
        except ValueError:
            continue
            
        # Check if we have at least N elements after c_idx
        if c_idx + 1 + N > len(remaining):
            continue
            
        # Verify the next N elements are all integers
        try:
            candidate_node_ids = [int(remaining[j]) for j in range(c_idx + 1, c_idx + 1 + N)]
            # If all are integers, this is our match!
            num_nodes = N
            node_ids = candidate_node_ids
            label_parts = remaining[:c_idx]
            found = True
            break
        except ValueError:
            continue
            
    if not found:
        raise ValueError(f"Line {line_num}: Could not parse node list structure matching 'num_nodes' count in '{line}'")
        
    label_raw = ",".join(label_parts)
    label = clean_label(label_raw)
    
    return {
        "id": he_id,
        "color": {
            "r": r,
            "g": g,
            "b": b
        },
        "label": label,
        "nodes": node_ids
    }

def convert_file(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Reading from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        # Ignore empty lines
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("Error: Input file is empty.", file=sys.stderr)
        sys.exit(1)

    # First line contains: num_nodes, num_hyperedges
    header = lines[0].split(',')
    if len(header) < 2:
        print(f"Error: Invalid header format in the first line: '{lines[0]}'", file=sys.stderr)
        sys.exit(1)
        
    try:
        num_nodes = int(header[0])
        num_hyperedges = int(header[1])
    except ValueError as e:
        print(f"Error: Header values are not integers: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Header: {num_nodes} nodes, {num_hyperedges} hyperedges")
    
    # Validate total expected lines
    expected_total_lines = 1 + num_nodes + num_hyperedges
    if len(lines) < expected_total_lines:
        print(f"Error: Expected at least {expected_total_lines} lines (1 header + {num_nodes} nodes + {num_hyperedges} hyperedges), but file has {len(lines)} lines.", file=sys.stderr)
        sys.exit(1)
        
    # Parse nodes
    nodes = []
    print(f"Parsing {num_nodes} nodes...")
    for i in range(1, num_nodes + 1):
        try:
            node = parse_node_line(lines[i], i + 1)
            nodes.append(node)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
            
    # Parse hyperedges
    hyperedges = []
    print(f"Parsing {num_hyperedges} hyperedges...")
    start_edge_idx = 1 + num_nodes
    for i in range(start_edge_idx, start_edge_idx + num_hyperedges):
        try:
            hyperedge = parse_hyperedge_line(lines[i], i + 1)
            hyperedges.append(hyperedge)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Output JSON structure
    output_data = {
        "nodes": nodes,
        "hyperedges": hyperedges
    }
    
    print(f"Writing parsed hypergraph to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print("Conversion completed successfully!")

if __name__ == "__main__":
    args = parse_arguments()
    convert_file(args.input_file, args.output_file)
