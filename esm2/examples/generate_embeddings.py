#!/usr/bin/env python3
"""
ESM2 Protein Embedding Generation
Author: Soumith Paritala
Purpose: Generate embeddings for protein sequences using ESM2-650M
Supports: FASTA and plain text formats
"""

import torch
import esm
import numpy as np
from pathlib import Path
import sys
import argparse

def detect_file_format(filepath):
    """
    Automatically detect if file is FASTA or plain text format
    
    Returns:
        'fasta' or 'plain'
    """
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # First non-empty line
                if line.startswith('>'):
                    return 'fasta'
                else:
                    return 'plain'
    return 'plain'

def read_fasta_file(filepath):
    """
    Parse FASTA format file (handles multi-line sequences)
    
    Returns:
        List of (name, sequence) tuples
    """
    sequences = []
    current_name = None
    current_seq = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Header line (starts with >)
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_name is not None and current_seq:
                    # Join all sequence lines and remove spaces
                    full_sequence = ''.join(current_seq).replace(' ', '')
                    sequences.append((current_name, full_sequence))
                
                # Start new sequence
                # Extract ID (everything between first | and second |)
                if '|' in line:
                    parts = line[1:].split('|')
                    if len(parts) >= 3:
                        current_name = parts[2].split()[0]  # e.g., "HBA_HUMAN"
                    else:
                        current_name = parts[0].split()[0]
                else:
                    # No pipes, just take first word after >
                    current_name = line[1:].split()[0]
                
                current_seq = []
            else:
                # Sequence line - just append (don't remove spaces yet)
                current_seq.append(line)
        
        # Save last sequence
        if current_name is not None and current_seq:
            full_sequence = ''.join(current_seq).replace(' ', '')
            sequences.append((current_name, full_sequence))
    
    return sequences

def read_plain_text_file(filepath):
    """
    Parse plain text file (one sequence per line or paragraph)
    
    Handles formats:
    1. name:sequence or name\tsequence
    2. Just sequences (one per line)
    3. Multi-line sequences separated by blank lines
    
    Returns:
        List of (name, sequence) tuples
    """
    sequences = []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by double newlines (paragraphs)
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    if not paragraphs:
        # Single paragraph, split by single newlines
        lines = [line.strip() for line in content.split('\n') if line.strip()]
    else:
        lines = paragraphs
    
    for i, line in enumerate(lines):
        # Remove all whitespace from sequence part
        line = ' '.join(line.split())  # Normalize whitespace first
        
        # Check for name:sequence or name\tsequence format
        if ':' in line:
            parts = line.split(':', 1)
            name = parts[0].strip()
            sequence = parts[1].strip().replace(' ', '')
        elif '\t' in line:
            parts = line.split('\t', 1)
            name = parts[0].strip()
            sequence = parts[1].strip().replace(' ', '')
        else:
            # Just sequence, no name
            name = f"protein{i+1}"
            sequence = line.replace(' ', '')
        
        # Only add if non-empty
        if sequence:
            sequences.append((name, sequence))
    
    return sequences

def validate_protein_sequence(sequence):
    """
    Check if a sequence is a valid protein sequence
    
    Returns:
        (is_valid, cleaned_sequence, error_message)
    """
    # Valid amino acid one-letter codes
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    
    # Convert to uppercase
    sequence = sequence.upper()
    
    # Remove any remaining whitespace
    sequence = ''.join(sequence.split())
    
    # Check for invalid characters
    invalid_chars = set(sequence) - valid_aa
    
    if invalid_chars:
        return False, sequence, f"Invalid characters: {sorted(invalid_chars)}"
    
    if len(sequence) < 10:
        return False, sequence, f"Too short ({len(sequence)} aa, minimum 10)"
    
    if len(sequence) > 50000:
        return False, sequence, f"Too long ({len(sequence)} aa, maximum 50000)"
    
    return True, sequence, None

def smart_read_sequences(filepath):
    """
    Intelligently read protein sequences from any file format
    
    Args:
        filepath: Path to input file
    
    Returns:
        List of (name, sequence) tuples
    """
    print("=" * 60)
    print("SMART FILE PARSER")
    print("=" * 60)
    
    # Detect format
    file_format = detect_file_format(filepath)
    print(f"Detected format: {file_format.upper()}")
    print("")
    
    # Parse accordingly
    if file_format == 'fasta':
        print("Using FASTA parser...")
        sequences = read_fasta_file(filepath)
    else:
        print("Using plain text parser...")
        sequences = read_plain_text_file(filepath)
    
    print(f"Found {len(sequences)} sequences in file")
    print("")
    
    # Validate each sequence
    print("Validating sequences...")
    print("-" * 60)
    
    validated_sequences = []
    
    for name, seq in sequences:
        is_valid, cleaned_seq, error_msg = validate_protein_sequence(seq)
        
        if is_valid:
            validated_sequences.append((name, cleaned_seq))
            print(f"  ✓ {name}: {len(cleaned_seq)} aa")
        else:
            print(f"  ✗ {name}: {error_msg}")
    
    print("-" * 60)
    print(f"\n✓ Successfully validated {len(validated_sequences)} sequences")
    
    if len(validated_sequences) == 0:
        raise ValueError("No valid sequences found in input file")
    
    return validated_sequences

def check_gpu():
    """Verify GPU availability"""
    print("=" * 60)
    print("GPU AVAILABILITY CHECK")
    print("=" * 60)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("⚠ WARNING: No GPU detected, using CPU (slow)")
    
    print("")
    return device

def load_model(device):
    """Load ESM2 model"""
    print("=" * 60)
    print("LOADING ESM2 MODEL")
    print("=" * 60)
    print("Model: ESM2-650M (33 layers, 650M parameters)")
    print("Loading...")
    print("")
    
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    print(f"  Embedding dimension: 1280")
    print("")
    
    return model, alphabet

def generate_embeddings(model, alphabet, sequences, device, output_dir="results"):
    """Generate ESM2 embeddings"""
    print("=" * 60)
    print("GENERATING EMBEDDINGS")
    print("=" * 60)
    print(f"Processing {len(sequences)} proteins...")
    print("")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare batch
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
    batch_tokens = batch_tokens.to(device)
    
    # Generate embeddings
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    
    embeddings = results["representations"][33].cpu().numpy()
    
    print(f"✓ Embeddings generated!")
    print(f"  Shape: {embeddings.shape}")
    print("")
    
    # Save results
    print("=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    for i, (name, seq) in enumerate(sequences):
        clean_name = name.replace("|", "_").replace("/", "_").replace(":", "_")
        
        # Save embedding
        np.save(output_path / f"{clean_name}_embedding.npy", embeddings[i])
        
        # Save metadata
        with open(output_path / f"{clean_name}_info.txt", "w") as f:
            f.write(f"Protein: {name}\n")
            f.write(f"Sequence length: {len(seq)} amino acids\n")
            f.write(f"\nSequence:\n")
            for j in range(0, len(seq), 60):
                f.write(seq[j:j+60] + "\n")
            f.write(f"\nEmbedding shape: {embeddings[i].shape}\n")
            f.write(f"Statistics:\n")
            f.write(f"  Mean: {embeddings[i].mean():.6f}\n")
            f.write(f"  Std:  {embeddings[i].std():.6f}\n")
        
        print(f"  ✓ {name}")
    
    print(f"\n✓ All results saved to: {output_path.absolute()}")
    return embeddings

def main():
    parser = argparse.ArgumentParser(
        description="Generate ESM2 protein embeddings from FASTA or text file"
    )
    parser.add_argument("input_file", help="Input FASTA or text file")
    parser.add_argument("-o", "--output", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("ESM2 PROTEIN EMBEDDING GENERATION")
    print("=" * 60)
    print("")
    
    # Check GPU
    device = check_gpu()
    
    # Load model
    model, alphabet = load_model(device)
    
    # Read sequences
    sequences = smart_read_sequences(args.input_file)
    
    # Generate embeddings
    embeddings = generate_embeddings(model, alphabet, sequences, device, args.output)
    
    print("\n" + "=" * 60)
    print("✓ COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()

