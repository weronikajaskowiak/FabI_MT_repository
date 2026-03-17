from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageDraw, ImageFont
import math, re, io

features = [
    (11, "[#6].[#6].[#6].[#6].[#6].[#6].[#6].[#6]"),
    (688, "[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]-,:[#6]"),
    (556, "[#6]=,:[#6]-,:[#6]-,:[#6]"),
    (490, "[#6]-,:[#6]-,:[#6]=,:[#6]"),
    (599, "[#6H,#6H2,#6H3]-,:[#6]=,:[#6H,#6H2,#6H3]"),
    (287, "[#6]~[F]"),
    (23, "[F]"),
    (145, "[N;r]1[#6][#6][#6][#6]1"),
    (19, "[O].[O]"),
    (24, "[F].[F]"),
]

def fix_smarts(s):
    return re.sub(r"\s+", "", s.strip())

# ---- prioritize double bonds for visualization ----
def prioritize_double_bonds(sm):
    sm = sm.replace("=,:", "=")   # double preferred over aromatic
    sm = sm.replace("-,:", "-")   # single if no double
    return sm

items = []
for fid, smarts in features:
    sm = fix_smarts(smarts)

    sm_draw = prioritize_double_bonds(sm)

    mol = Chem.MolFromSmarts(sm_draw)

    if mol:
        AllChem.Compute2DCoords(mol)
        items.append((fid, sm, mol))
    else:
        print(f"Failed to parse: {fid}")

# ---- Layout ----
cols = 3
mol_w, mol_h = 360, 220
text_h = 70
pad = 20

rows = math.ceil(len(items) / cols)
cell_w = mol_w
cell_h = mol_h + text_h

canvas_w = cols * cell_w + (cols + 1) * pad
canvas_h = rows * cell_h + (rows + 1) * pad

canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
drawer = ImageDraw.Draw(canvas)

font = ImageFont.truetype(
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf", 22
)

for idx, (fid, sm, mol) in enumerate(items):

    r = idx // cols
    c = idx % cols

    x0 = pad + c * (cell_w + pad)
    y0 = pad + r * (cell_h + pad)

    d2d = rdMolDraw2D.MolDraw2DCairo(mol_w, mol_h)
    opts = d2d.drawOptions()

    for atom in mol.GetAtoms():
        opts.atomLabels[atom.GetIdx()] = atom.GetSymbol()

    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()

    mol_img = Image.open(io.BytesIO(d2d.GetDrawingText()))
    canvas.paste(mol_img, (x0, y0))

    y_text = y0 + mol_h + 8

    id_text = f"Bit_{fid}"
    id_bbox = drawer.textbbox((0, 0), id_text, font=font)

    id_x = x0 + (cell_w - (id_bbox[2] - id_bbox[0])) // 2
    drawer.text((id_x, y_text), id_text, fill="black", font=font)

    sm_bbox = drawer.textbbox((0, 0), sm, font=font)
    sm_x = x0 + (cell_w - (sm_bbox[2] - sm_bbox[0])) // 2

    drawer.text((sm_x, y_text + 20), sm, fill="black", font=font)

canvas.save("feature_smarts.png")

print("Saved as feature_smarts.png")
