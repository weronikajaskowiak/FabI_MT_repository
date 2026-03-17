import os
import json
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
# from skfp.fingerprints import KlekotaRothFingerprint as FPFingerprint
from skfp.fingerprints import PubChemFingerprint as FPFingerprint


def highlight_based_on_importance(
    molecules=None,
    smarts_patterns=None,
    output_dir="Results_importance",
    dict_of_actives=None,
    dict_of_inactives=None,
    active_color=(1.0, 0.0, 0.0),
    inactive_color=(0.0, 0.0, 0.8),
    coloring_mode="combine",
):
    """
    Highlight molecular substructures using fragment-importance weights (e.g. SHAP-derived).

    In the modeling interpretation used by the caller, negative SHAP values may be associated
    with lower docking scores (more “active”), while positive SHAP values may be associated
    with higher docking scores (more “inactive”). This function itself does not compute SHAP
    values or split them by sign; it only visualizes the weights provided in
    `dict_of_actives` and `dict_of_inactives`.

    This function takes molecules and fragment-importance values, identifies matching
    substructures using pattern strings interpreted as SMARTS, and applies color scaling to
    visualize relative importance. It renders 2D depictions with highlighted atoms and bonds,
    producing PNG images that show active, inactive, or combined fragment contributions.

    Args:
        molecules (str | list[str] | list[Chem.Mol] | None):
            Input molecules. Can be an SDF/SMI file path, a list of SMILES,
            or a list of RDKit Mol objects.
        smarts_patterns (dict | list | str | None):
            Fragment SMARTS definitions or a path to a JSON file containing them.
            If None, Klekota–Roth SMARTS patterns or PubChem SMARTS patterns are used.
            Disclaimer: PubChem fingerprint feature definitions do not fully consist of valid SMARTS patterns.
            Use provided .json file for PubChem Fingerprints (bit and count variants).
            Reference: https://ftp.ncbi.nlm.nih.gov/pubchem/specifications/pubchem_fingerprints.txt
        output_dir (str):
            Directory where the generated PNG images will be saved.
            Defaults to "Results_importance".
        dict_of_actives (dict | None):
            Mapping of fragment index: importance weight for active contributions.
        dict_of_inactives (dict | None):
            Mapping of fragment index: importance weight for inactive contributions.
        active_color (tuple[float, float, float]):
            RGB color used for active fragments.
            Defaults to red (1.0, 0.0, 0.0).
        inactive_color (tuple[float, float, float]):
            RGB color used for inactive fragments.
            Defaults to blue (0.0, 0.0, 0.8).
        coloring_mode (str):
            Either "combine" (single image with overlapping highlights)
            or "separate" (two images: active and inactive).

    Notes:
        - Importance values scale the color intensity applied to atoms and bonds.
        - Colors blend smoothly when active and inactive fragments overlap.
        - SMARTS patterns determine which molecular regions correspond to each fragment.
        - Images provide a simple visual explanation of fragment contributions.

    Returns:
        None:
            Output filenames:
                - <output_dir>/<idx>_mol_combine.png
                - <output_dir>/<idx>_mol_active.png
                - <output_dir>/<idx>_mol_inactive.png

    Example:
        highlight_based_on_importance(
            molecules=["O=C(O)c1ccccc1O"],
            smarts_patterns=[
                "C=O",
                "[CX3]=[OX1]",
                "c1ccccc1",
                "[OX2H]"
            ],
            dict_of_actives={0: 1.0},
            dict_of_inactives={1: 0.9, 2: 0.3, 3: 0.5},
            coloring_mode="separate"
        )
    """

    kr_smarts_dict = {}
    default_features = FPFingerprint(count=True).get_feature_names_out()
    for idx, feature in enumerate(default_features):
        kr_smarts_dict[idx] = feature

    if isinstance(smarts_patterns, str):
        if smarts_patterns.endswith(".json"):
            with open(smarts_patterns, "r") as f:
                kr_smarts_dict = json.load(f)
            print(f"Loaded fingerprint dictionary from JSON file: {smarts_patterns}")
    elif isinstance(smarts_patterns, list):
        kr_smarts_dict = {idx: feature for idx, feature in enumerate(smarts_patterns)}
    elif isinstance(smarts_patterns, dict):
        kr_smarts_dict = smarts_patterns

    smarts_list = list(kr_smarts_dict.values())

    mol_list = []
    if isinstance(molecules, str):
        if molecules.endswith(".sdf"):
            suppl = Chem.SDMolSupplier(molecules)
            mol_list = [m for m in suppl if m is not None]
        elif molecules.endswith(".smi") or molecules.endswith(".smiles"):
            suppl = Chem.SmilesMolSupplier(molecules, delimiter=" ", titleLine=False)
            mol_list = [m for m in suppl if m is not None]
    elif isinstance(molecules, list):
        if isinstance(molecules[0], str):
            mol_list = [Chem.MolFromSmiles(m) for m in molecules]
        elif isinstance(molecules[0], Chem.Mol):
            mol_list = molecules

    os.makedirs(output_dir, exist_ok=True)

    MAX_ACTIVE = max(dict_of_actives.values()) if dict_of_actives else 1.0
    MAX_INACTIVE = max(dict_of_inactives.values()) if dict_of_inactives else 1.0

    def scale_color(base_color, weight, max_weight):
        if weight <= 0 or max_weight <= 0:
            return (1.0, 1.0, 1.0)  # no highlight

        w = weight / max_weight  # linear

        r, g, b = base_color
        return (1 - (1 - r) * w,
                1 - (1 - g) * w,
                1 - (1 - b) * w)

    def blend_color_when_overlap(c1, c2):
        r = (c1[0] + c2[0]) / 2
        g = (c1[1] + c2[1]) / 2
        b = (c1[2] + c2[2]) / 2
        return (r, g, b)

    def build_fragment_map(mol, smarts_list, bit_dict):
        frag_map = {}
        for bit in bit_dict.keys():
            patt = Chem.MolFromSmarts(smarts_list[bit])
            if not patt:
                frag_map[bit] = {"atoms": [], "bonds": []}
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

            frag_map[bit] = {"atoms": atom_matches, "bonds": bond_matches}
        return frag_map

    def visualize_gradient(
        mol,
        active_map,
        inactive_map,
        filename,
        dict_of_actives,
        dict_of_inactives,
        coloring_mode,
        size=(1200, 1200),
    ):
        mol_copy = Chem.Mol(mol)
        rdDepictor.Compute2DCoords(mol_copy)

        atom_colors = {}
        bond_colors = {}
        highlighted_atoms = []
        highlighted_bonds = []

        for bit, fragments in active_map.items():
            weight = dict_of_actives.get(bit, 0) if dict_of_actives else 0
            scaled_color = scale_color(active_color, weight, MAX_ACTIVE)

            for frag in fragments["atoms"]:
                for a in frag:
                    atom_colors[a] = scaled_color
                    highlighted_atoms.append(a)

            for b in fragments["bonds"]:
                bond_colors[b] = scaled_color
                highlighted_bonds.append(b)

        for bit, fragments in inactive_map.items():
            weight = dict_of_inactives.get(bit, 0) if dict_of_inactives else 0
            scaled_color = scale_color(inactive_color, weight, MAX_INACTIVE)

            for frag in fragments["atoms"]:
                for a in frag:
                    if a in atom_colors:
                        atom_colors[a] = blend_color_when_overlap(
                            atom_colors[a], scaled_color
                        )
                    else:
                        atom_colors[a] = scaled_color
                    highlighted_atoms.append(a)

            for b in fragments["bonds"]:
                if b in bond_colors:
                    bond_colors[b] = blend_color_when_overlap(
                        bond_colors[b], scaled_color
                    )
                else:
                    bond_colors[b] = scaled_color

                highlighted_bonds.append(b)

        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        drawer.drawOptions().useBWAtomPalette()
        drawer.drawOptions().highlightBondWidthMultiplier = 10
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
            legend=mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        )

        drawer.FinishDrawing()
        with open(filename, "wb") as f:
            f.write(drawer.GetDrawingText())

    for i, mol in enumerate(mol_list, start=1):
        if mol is None:
            continue

        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "molecule"
        safe_name = mol_name.replace("/", "_").replace(" ", "_")

        active_map = build_fragment_map(mol, smarts_list, dict_of_actives or {})
        inactive_map = build_fragment_map(mol, smarts_list, dict_of_inactives or {})

        base_file = os.path.join(output_dir, safe_name)

        if coloring_mode == "separate":
            visualize_gradient(
                mol,
                active_map,
                {},
                base_file + "_active.png",
                dict_of_actives,
                {},
                coloring_mode,
            )
            visualize_gradient(
                mol,
                {},
                inactive_map,
                base_file + "_inactive.png",
                {},
                dict_of_inactives,
                coloring_mode,
            )
        else:
            visualize_gradient(
                mol,
                active_map,
                inactive_map,
                base_file + "_combine.png",
                dict_of_actives,
                dict_of_inactives,
                coloring_mode,
            )

    print(f"GRADIENT RESULTS SAVED IN: {output_dir}")


highlight_based_on_importance(
    "/Users/weronikajaskowiak/Desktop/HighlightFingerprints/4FS3_regression_best_model/molecules_with_best_docking_scores_4FS3.smiles",
    "/Users/weronikajaskowiak/Desktop/HighlightFingerprints/dataset/smart_patterns/PubChem_fp_bit_variant.json",
    dict_of_actives={
        11: 0.0424625773667839,
        688: 0.0306581184978996,
        556: 0.0301729976480757,
        490: 0.0282131719868635,
        599: 0.0259968012680514
    },
    dict_of_inactives={
        287: 0.0518213891211777,
        23: 0.0495690397334639,
        145: 0.0265935986613709,
        19: 0.0228981927252144,
        24: 0.0199273741296879

    }, output_dir="Results_4FS3_best_molecules_importance_final"
)
