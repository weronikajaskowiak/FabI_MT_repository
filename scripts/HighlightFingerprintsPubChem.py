import os
import json
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from skfp.fingerprints import PubChemFingerprint as FPFingerprint


def highlight_based_on_bits(
    molecules=None,
    smarts_patterns=None,
    output_dir="Results",
    list_of_active_bits=None,
    list_of_inactive_bits=None,
    active_color=(1.0, 0.0, 0.0, 0.5),
    inactive_color=(0.0, 0.0, 0.8, 0.5),
    overlap_color=(1.0, 1.0, 0.0, 0.5),
    coloring_mode="combine",
):
    """
    Highlights active and inactive molecular substructures based on SMARTS fingerprint bits.

    This function loads molecules and SMARTS patterns, computes substructure matches using RDKit,
    and generates 2D molecular depictions with specific color highlights for active, inactive,
    and overlapping fragments. Colouring is based on SHAP values: negative contributions
    (associated with active features) are shown in red, while positive contributions are shown in blue.
    It also saves textual descriptions of highlighted atoms and bonds.

    Args:
        molecules (str | list, optional):
            Input molecules, which can be:
              - Path to an SDF or SMI/SMILES file, or
              - List of SMILES strings, or
              - List of RDKit Mol objects.
        smarts_patterns (str | list | dict, optional):
            SMARTS patterns used for substructure matching. Can be:
              - Path to a JSON file containing SMARTS definitions,
              - A list of SMARTS strings, or
              - A dictionary mapping labels to SMARTS.
            Defaults to the PubChem fingerprints patterns.
        output_dir (str, optional):
            Directory where generated images and descriptions will be saved.
            Defaults to "Results".
        list_of_active_bits (list[int], optional):
            List of indices corresponding to active SMARTS patterns.
            If None, active bits are assigned automatically based on fingerprints matches.
        list_of_inactive_bits (list[int], optional):
            List of indices corresponding to inactive SMARTS patterns.
            If None, inactive bits are not highlighted, as only detected active bits are shown.
        active_color (tuple[float, float, float, float], optional):
            RGBA color used to highlight active fragments. Defaults to red (1.0, 0.0, 0.0, 0.5).
        inactive_color (tuple[float, float, float, float], optional):
            RGBA color used to highlight inactive fragments. Defaults to blue (1.0, 1.0, 0.0, 0.5).
        overlap_color (tuple[float, float, float, float], optional):
            RGBA color used for overlapping atoms or bonds that are both active and inactive.
            Defaults to yellow (1.0, 1.0, 0.0, 0.5).
        coloring_mode (str, optional):
            Visualization mode.
              - "combine": Draw active and inactive fragments together (default).
              - "separate": Generate separate images for active and inactive fragments.

    Returns:
        None:
            The function does not return any value. It saves:
              - Highlighted molecule images in <output_dir>/images/
              - Atom and bond descriptions in <output_dir>/descriptions/

    Notes:
        - Uses RDKit for molecule handling and SMARTS matching.

    Examples:
        highlight_based_on_bits(
            molecules=["CC(O)C(=O)O"],
            smarts_patterns={
                "Hydroxyl": "[OX2H]",
                "Alcohol_fragment": "[CX4][OX2H]",
                "Carbonyl": "C=O"
            },
            list_of_active_bits=[0],
            list_of_inactive_bits=[1],
            output_dir="test_output",
            active_color=(1.0, 0.41, 0.71, 0.5),
            inactive_color=(0.12, 0.56, 1.0, 0.5),
            coloring_mode="separate"
        )

    """

    fp_dict = {}
    default_features = FPFingerprint(count=True).get_feature_names_out()
    for idx, feature in enumerate(default_features):
        fp_dict[idx] = feature

    if isinstance(smarts_patterns, str):
        if smarts_patterns.endswith(".json"):
            with open(smarts_patterns, "r") as f:
                fp_dict = json.load(f)
            print("Loaded fingerprint dictionary " f"from JSON file: {smarts_patterns}")
    elif isinstance(smarts_patterns, list):
        fp_dict = {}
        for idx, feature in enumerate(smarts_patterns):
            fp_dict[idx] = feature
    elif isinstance(smarts_patterns, dict):
        fp_dict = smarts_patterns

    smarts_list = list(fp_dict.values())

    mol_list = []
    if isinstance(molecules, str):
        if molecules.endswith(".sdf"):
            suppl = Chem.SDMolSupplier(molecules)
            for mol in suppl:
                if mol is not None:
                    mol_list.append(mol)
        elif molecules.endswith(".smi") or molecules.endswith(".smiles"):
            suppl = Chem.SmilesMolSupplier(molecules, delimiter=" ", titleLine=False)
            for mol in suppl:
                if mol is not None:
                    mol_list.append(mol)
    elif isinstance(molecules, list):
        if isinstance(molecules[0], str):
            for mol in molecules:
                mol_list.append(Chem.MolFromSmiles(mol))
        elif isinstance(molecules[0], Chem.Mol):
            for mol in molecules:
                mol_list.append(mol)

    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    description_dir = os.path.join(output_dir, "descriptions")
    os.makedirs(description_dir, exist_ok=True)

    def compute_fingerprints(mol_list, smarts_list):
        results = []
        for mol in mol_list:
            fp = []
            for smarts in smarts_list:
                patt = Chem.MolFromSmarts(smarts)
                if patt:
                    matches = mol.GetSubstructMatches(patt)
                    if matches:
                        fp.append(1)
                    else:
                        fp.append(0)
                else:
                    fp.append(0)
            results.append(fp)

        return results

    def build_active_map(mol, smarts_list, active_bits):
        active_map = {}
        for bit in active_bits:
            patt = Chem.MolFromSmarts(smarts_list[bit])
            if not patt:
                active_map[bit] = {"atoms": [], "bonds": []}
                continue

            matches = mol.GetSubstructMatches(patt)
            atom_matches = []
            bond_matches = []

            for match in matches:
                atom_list = list(match)
                atom_matches.append(atom_list)
                for bond in patt.GetBonds():
                    aid1 = atom_list[bond.GetBeginAtomIdx()]
                    aid2 = atom_list[bond.GetEndAtomIdx()]
                    b = mol.GetBondBetweenAtoms(aid1, aid2)
                    if b:
                        bond_matches.append(b.GetIdx())

            active_map[bit] = {"atoms": atom_matches, "bonds": bond_matches}

        return active_map

    def build_inactive_map(mol, smarts_list, inactive_bits, active_map):
        inactive_map = {}
        for bit in inactive_bits:
            patt = Chem.MolFromSmarts(smarts_list[bit])
            if not patt:
                inactive_map[bit] = {"atoms": [], "bonds": []}
                continue
            matches = mol.GetSubstructMatches(patt)
            atom_matches = []
            bond_matches = []
            for match in matches:
                atom_list = list(match)
                atom_matches.append(atom_list)
                for bond in patt.GetBonds():
                    aid1 = atom_list[bond.GetBeginAtomIdx()]
                    aid2 = atom_list[bond.GetEndAtomIdx()]
                    b = mol.GetBondBetweenAtoms(aid1, aid2)
                    if b:
                        bond_matches.append(b.GetIdx())
            inactive_map[bit] = {"atoms": atom_matches, "bonds": bond_matches}

        all_atoms = set(range(mol.GetNumAtoms()))
        active_atoms = set()
        for frags in active_map.values():
            for frag in frags["atoms"]:
                for a in frag:
                    active_atoms.add(a)

        inactive_atoms = set()
        for frags in inactive_map.values():
            for frag in frags["atoms"]:
                for a in frag:
                    inactive_atoms.add(a)

        unmatched_atoms = set()
        for a in all_atoms:
            if a not in active_atoms and a not in inactive_atoms:
                unmatched_atoms.add(a)

        if unmatched_atoms:
            inactive_map[-1] = {"atoms": [list(unmatched_atoms)], "bonds": []}

        return inactive_map

    def visualize_fragments(
        mol,
        active_map,
        inactive_map,
        filename,
        active_color,
        inactive_color,
        overlap_color,
        size=(1200, 1200),
    ):
        mol_copy = Chem.Mol(mol)
        rdDepictor.Compute2DCoords(mol_copy)

        atom_colors = {}
        bond_colors = {}
        highlighted_atoms = []
        highlighted_bonds = []

        for bit, fragments in active_map.items():
            for frag in fragments["atoms"]:
                for a in frag:
                    atom_colors[a] = active_color
                    highlighted_atoms.append(a)
            for b in fragments["bonds"]:
                bond_colors[b] = active_color
                highlighted_bonds.append(b)

        for bit, fragments in inactive_map.items():
            if bit == -1:
                continue

            for frag in fragments["atoms"]:
                for a in frag:
                    if a in atom_colors:
                        if (
                            atom_colors[a] == active_color
                            or atom_colors[a] == overlap_color
                        ):
                            atom_colors[a] = overlap_color
                        else:
                            atom_colors[a] = inactive_color
                    else:
                        atom_colors[a] = inactive_color
                    highlighted_atoms.append(a)

            for b in fragments["bonds"]:
                if b in bond_colors:
                    if (
                        bond_colors[b] == active_color
                        or bond_colors[b] == overlap_color
                    ):
                        bond_colors[b] = overlap_color
                    else:
                        bond_colors[b] = inactive_color
                else:
                    bond_colors[b] = inactive_color
                highlighted_bonds.append(b)

        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        drawer.drawOptions().useBWAtomPalette()
        drawer.drawOptions().minFontSize = 16
        drawer.drawOptions().maxFontSize = 40
        drawer.drawOptions().legendFontSize = 50

        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            mol_copy,
            highlightAtoms=highlighted_atoms,
            highlightBonds=highlighted_bonds,
            highlightAtomColors=atom_colors,
            highlightBondColors=bond_colors,
            legend=mol.GetProp("_Name") if mol.HasProp("_Name") else "",
        )
        drawer.FinishDrawing()
        with open(filename, "wb") as f:
            f.write(drawer.GetDrawingText())

    def visualize_separate(mol, active_map, inactive_map, base_filename):
        visualize_fragments(
            mol,
            active_map,
            {},
            base_filename + "_active.png",
            active_color,
            inactive_color,
            overlap_color,
        )
        visualize_fragments(
            mol,
            {},
            inactive_map,
            base_filename + "_inactive.png",
            active_color,
            inactive_color,
            overlap_color,
        )

    def make_description(mol, active_map, inactive_map, filepath):

        active_atoms = set()
        for v in active_map.values():
            for frag in v["atoms"]:
                for a in frag:
                    active_atoms.add(a)

        inactive_atoms = set()
        for v in inactive_map.values():
            for frag in v["atoms"]:
                for a in frag:
                    inactive_atoms.add(a)

        overlap_atoms = active_atoms & inactive_atoms

        with open(filepath, "w") as f:
            f.write(f"SMILES: {Chem.MolToSmiles(mol)}\n\n")
            f.write("ACTIVE ATOMS:\n")
            for bit, data in active_map.items():
                for frag in data["atoms"]:
                    symbols = []
                    for a in frag:
                        atom = mol.GetAtomWithIdx(a)
                        symbols.append(atom.GetSymbol())
                    f.write(f"Bit {bit}: atoms {frag} -> {symbols}\n")

            f.write("\nINACTIVE ATOMS:\n")
            for bit, data in inactive_map.items():
                for frag in data["atoms"]:
                    symbols = []
                    for a in frag:
                        atom = mol.GetAtomWithIdx(a)
                        symbols.append(atom.GetSymbol())
                    if bit == -1:
                        f.write(f"Unmatched atoms: {frag} -> {symbols}\n")
                    else:
                        f.write(f"Bit {bit}: atoms {frag} -> {symbols}\n")

            if overlap_atoms:
                f.write("\nOVERLAPPING ATOMS:\n")
                overlap_symbols = [
                    mol.GetAtomWithIdx(a).GetSymbol() for a in sorted(overlap_atoms)
                ]
                f.write(f"Atoms: {sorted(list(overlap_atoms))} -> {overlap_symbols}\n")

    name_counts = {}
    for i, mol in enumerate(mol_list, start=1):
        if mol is None:
            continue

        fp = compute_fingerprints([mol], smarts_list)[0]

        user_active = list_of_active_bits is not None
        user_inactive = list_of_inactive_bits is not None

        if user_active:
            print("Using provided list of active bits.")
            active_bits = list_of_active_bits
        else:
            print("No list of active bits provided. Calculating active bits...")
            active_bits = []
            for j, b in enumerate(fp):
                if b == 1:
                    active_bits.append(j)

        if user_inactive:
            print("Using provided list of inactive bits.")
            inactive_bits = list_of_inactive_bits
        else:
            print("No list of inactive bits provided. Calculating inactive bits...")
            inactive_bits = []
            for j, b in enumerate(fp):
                if b == 0:
                    inactive_bits.append(j)

        if user_inactive and not user_active:
            active_bits = []
        elif user_active and not user_inactive:
            inactive_bits = []

        filtered_active_bits = []
        for bit in active_bits:
            if bit not in inactive_bits:
                filtered_active_bits.append(bit)
        active_bits = filtered_active_bits

        active_map = build_active_map(mol, smarts_list, active_bits)
        inactive_map = build_inactive_map(mol, smarts_list, inactive_bits, active_map)

        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"molecule_{i}"
        safe_name = mol_name.replace("/", "_").replace(" ", "_")

        name_counts[safe_name] = name_counts.get(safe_name, 0) + 1
        if name_counts[safe_name] > 1:
            unique_name = f"{safe_name}_{name_counts[safe_name]}"
        else:
            unique_name = safe_name

        base_file = os.path.join(image_dir, unique_name)
        txt_file = os.path.join(description_dir, f"{unique_name}_description.txt")
        make_description(mol, active_map, inactive_map, txt_file)

        if coloring_mode == "separate":
            visualize_separate(mol, active_map, inactive_map, base_file)
        else:
            print("No coloring mode provided. Using default: combine.")
            visualize_fragments(
                mol,
                active_map,
                inactive_map,
                base_file + "_combine.png",
                active_color,
                inactive_color,
                overlap_color,
            )

    print(f"RESULTS SAVED IN: {output_dir}")


highlight_based_on_bits(
    "/Users/weronikajaskowiak/Desktop/HighlightFingerprints/4FS3_regression_best_model/molecules_with_best_docking_scores_4FS3.smiles",
    smarts_patterns="/Users/weronikajaskowiak/Desktop/HighlightFingerprints/dataset/smart_patterns/PubChem_fp_bit_variant_secure.json",
    list_of_active_bits=[11, 688, 556, 490, 599],
    list_of_inactive_bits=[287, 23, 145, 19, 24],
)
