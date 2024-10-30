import gemmi
import numpy as np
import sys
from pathlib import Path
import os
from datetime import datetime
import shutil

def parse_atom_specification(spec):
    """
    Parse an atom specification string of format "CHAIN1 RES1 NUM1 ATOM1 to CHAIN2 RES2 NUM2 ATOM2"
    Returns tuple of ((chain1, res1, num1, atom1), (chain2, res2, num2, atom2))
    """
    try:
        first_part, second_part = spec.split(" to ")
        chain1, res1, num1, atom1 = first_part.split()
        chain2, res2, num2, atom2 = second_part.split()
        return ((chain1, res1, int(num1), atom1), (chain2, res2, int(num2), atom2))
    except ValueError:
        raise ValueError("Invalid atom specification format. Expected format: 'CHAIN1 RES1 NUM1 ATOM1 to CHAIN2 RES2 NUM2 ATOM2'")

def get_atom_position(structure, chain_name, residue_name, residue_number, atom_name):
    """
    Get the position of a specific atom in the structure
    Returns the position as a numpy array [x, y, z] or None if atom not found
    """
    for model in structure:
        for chain in model:
            if chain.name == chain_name:
                for residue in chain:
                    if residue.seqid.num == residue_number and residue.name == residue_name:
                        for atom in residue:
                            if atom.name == atom_name:
                                pos = atom.pos
                                return np.array([pos.x, pos.y, pos.z])
    return None

def calculate_distance(structure, atom_specification):
    """
    Calculate the distance between two atoms specified in the atom_specification
    Returns the distance in Angstroms or None if either atom is not found
    """
    # Parse the atom specification
    (chain1, res1, num1, atom1), (chain2, res2, num2, atom2) = parse_atom_specification(atom_specification)
    
    # Get positions of both atoms
    pos1 = get_atom_position(structure, chain1, res1, num1, atom1)
    pos2 = get_atom_position(structure, chain2, res2, num2, atom2)
    
    # Check if both atoms were found
    if pos1 is None or pos2 is None:
        missing_atoms = []
        if pos1 is None:
            missing_atoms.append(f"Chain {chain1}, {res1} {num1} {atom1}")
        if pos2 is None:
            missing_atoms.append(f"Chain {chain2}, {res2} {num2} {atom2}")
        return None, missing_atoms
    
    # Calculate Euclidean distance
    return np.linalg.norm(pos1 - pos2), None

def calculate_combined_distance(structure, specifications):
    """
    Calculate the combined distance for multiple atom specifications connected by '+'
    Returns the total distance in Angstroms or None if any atom is not found
    """
    total_distance = 0.0
    missing_atoms = []

    for spec in specifications:
        distance_result = calculate_distance(structure, spec)
        if distance_result[0] is not None:
            total_distance += distance_result[0]
        else:
            missing_atoms.extend(distance_result[1])
    
    if missing_atoms:
        return None, missing_atoms

    return total_distance, None

def find_cif_files(directory):
    """
    Recursively find all .cif files in the given directory and its subdirectories
    Returns a list of Path objects
    """
    directory = Path(directory)
    cif_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.cif'):
                cif_files.append(Path(root) / file)
    
    return sorted(cif_files)

def process_directory(directory, specification_file):
    """
    Process all AlphaFold prediction folders in the given directory
    Args:
        directory: Path to directory containing AlphaFold prediction folders
        specification_file: Path to file containing atom specifications
    """
    try:
        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(f"distance_results_{timestamp}.txt")
        
        # Create directory for copying cif files with lowest distance
        lowest_distance_dir = Path(f"lowest_distances_{timestamp}")
        lowest_distance_dir.mkdir(exist_ok=True)

        # Read specifications from file
        with open(specification_file, 'r') as f:
            specifications = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        if not specifications:
            raise ValueError("No valid specifications found in specification file")
        
        # Find all .cif files
        cif_files = find_cif_files(directory)
        if not cif_files:
            raise ValueError(f"No .cif files found in {directory}")
        
        print(f"\nFound {len(cif_files)} .cif files to process")
        print(f"Results will be written to: {output_file}")
        print("\nProcessing structures...\n")
        
        # Dictionary to hold the lowest distance information
        lowest_distances = {}

        # Process each structure and write results to file
        with open(output_file, 'w') as out_f:
            # Write header with timestamp
            out_f.write(f"# Distance calculations performed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            out_f.write("# Format: Structure_name | Specification | Distance(Å)\n")
            out_f.write("-" * 80 + "\n")
            
            for cif_file in cif_files:
                structure_name = f"{cif_file.parent.name}/{cif_file.name}"
                print(f"Processing: {structure_name}")
                
                try:
                    structure = gemmi.read_structure(str(cif_file))
                    
                    # Process each specification
                    for spec in specifications:
                        individual_specs = [s.strip() for s in spec.split('+')]
                        distance_result = calculate_combined_distance(structure, individual_specs)
                        
                        if distance_result[0] is not None:
                            # Write successful calculations to file
                            distance = distance_result[0]
                            out_f.write(f"{structure_name} | {spec} | {distance:.2f}\n")
                            
                            # Track the lowest distance for each specification
                            if spec not in lowest_distances or distance < lowest_distances[spec][0]:
                                lowest_distances[spec] = (distance, structure_name, cif_file)
                        else:
                            # Print missing atoms to terminal
                            missing_atoms = distance_result[1]
                            print(f"  Skipping: {spec}")
                            print(f"  Missing atoms in {structure_name}: {', '.join(missing_atoms)}")
                    
                except Exception as e:
                    print(f"  Error processing {structure_name}: {str(e)}", file=sys.stderr)
                
                print()  # Add blank line between structures

        # Copy the .cif files with the lowest distances to the new directory
        for spec, (distance, structure_name, cif_file) in lowest_distances.items():
            new_file_name = f"{cif_file.stem}_{spec.replace(' ', '_')}.cif"
            shutil.copy(cif_file, lowest_distance_dir / new_file_name)

        print(f"\nProcessing complete! Results written to {output_file}")
        print(f"Copied .cif files with lowest distances to: {lowest_distance_dir}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)

def main():
    """
    Example usage of the batch directory processor
    """
    # Example usage
    directory = ''
    spec_file = "input.txt"
    
    try:
        process_directory(directory, spec_file)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)

if __name__ == "__main__":
    main()
