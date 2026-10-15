"""Signet ring with a hidden microSD card slot -- parametric and 3D-printable.

    python src/sd_ring.py                    # microSD, inner diameter 17.3 mm
    python src/sd_ring.py --inner-d 18.2     # measure the inside of a ring that fits you
    python src/sd_ring.py --resin            # tighter slot clearances for resin (SLA/DLP) printers
    python src/sd_ring.py --card sd          # full-size SD: works, but the head is ~35 x 27 mm
    python src/sd_ring.py --set              # sizing set: US 4..13 in half sizes, one plate
    python src/sd_ring.py --set --us-range 6 10
    python src/sd_ring.py --set --out stl --no-preview   # regenerates the STLs published in stl/

Writes two STLs plus a preview into out/ (or --out DIR):
  *_signet.stl  symmetric signet; resin, or FDM lying flat with tree supports under the band
  *_flat.stl    one side planar so it prints on FDM lying flat with no supports at all
--set instead writes flat rings with their inner diameter engraved inside the band, one STL
per size in rings/ plus all of them laid out on a single plate, sizing_plate_US*.stl.

Model frame: finger axis = Z, head on +Y, the card slides in from the +X side of the
head. A pin hole in the -X end pushes it back out (SIM-tray pin), and a notch over the
mouth exposes the card's edge for a fingernail. Needs: pip install manifold3d numpy matplotlib
"""
import argparse
import math
import struct
from pathlib import Path

import numpy as np
from manifold3d import Manifold, Mesh

CARDS = {"microsd": (15.0, 11.0, 1.0), "sd": (32.0, 24.0, 2.1)}  # length (slide), width (along finger), thickness


def cross2(u, v):
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def circle(cx, cy, r, n):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]


def convex_hull(p):
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2 and cross2(h[-1] - h[-2], q - h[-2]) <= 0:
                h.pop()
            h.append(q)
        return h

    lo, hi = half(p), half(p[::-1])
    return np.array(lo[:-1] + hi[:-1])


def ray_radius(poly, phis):
    """Distance from the origin to a convex polygon (containing it) along each angle."""
    a = poly[None]
    e = (np.roll(poly, -1, 0) - poly)[None]
    d = np.c_[np.cos(phis), np.sin(phis)][:, None]
    den = cross2(d, e)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = cross2(a, e) / den
        s = cross2(a, d) / den
    ok = (s >= -1e-9) & (s <= 1 + 1e-9) & (t > 0)
    return np.where(ok, t, np.inf).min(axis=1)


def loft(phis, r_in, r_out, z_lo, z_hi, f_in, f_out, k=8):
    """Sweep a rounded-rectangle cross-section around the Z axis; every slice may differ."""
    n = len(phis)
    q = np.linspace(0, np.pi / 2, k + 1)
    ri, ro, zl, zh = (np.broadcast_to(np.asarray(v, float), (n,))[:, None] for v in (r_in, r_out, z_lo, z_hi))
    corners = [  # (centre r, centre z, radius, start angle), counter-clockwise in the (r, z) plane
        (ri + f_in, zl + f_in, f_in, np.pi),
        (ro - f_out, zl + f_out, f_out, 1.5 * np.pi),
        (ro - f_out, zh - f_out, f_out, 0.0),
        (ri + f_in, zh - f_in, f_in, 0.5 * np.pi),
    ]
    r = np.hstack([cr + f * np.cos(a0 + q) for cr, cz, f, a0 in corners])
    z = np.hstack([cz + f * np.sin(a0 + q) for cr, cz, f, a0 in corners])
    m = r.shape[1]
    verts = np.stack([r * np.cos(phis)[:, None], r * np.sin(phis)[:, None], z], -1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
    a, b = i * m + j, (i + 1) % n * m + j
    c, d = (i + 1) % n * m + (j + 1) % m, i * m + (j + 1) % m
    tris = np.r_[np.c_[a.ravel(), b.ravel(), c.ravel()], np.c_[a.ravel(), c.ravel(), d.ravel()]]
    v0, v1, v2 = verts[tris].transpose(1, 0, 2)
    if np.einsum("ij,ij->", v0, np.cross(v1, v2)) < 0:
        tris = tris[:, ::-1]
    return Manifold(Mesh(vert_properties=verts.astype(np.float32), tri_verts=tris.astype(np.uint32)))


SEVEN = {"0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg", "5": "acdfg",
         "6": "acdefg", "7": "abc", "8": "abcdefg", "9": "abcdfg"}


def glyph_rects(text, w=2.2, h=3.2, t=0.7, gap=0.6):  # strokes stay wider than an 0.4 mm nozzle
    """Seven-segment strokes for digits and '.', as (u0, u1, v0, v1) rectangles centred on the origin."""
    seg = {"a": (0, w, h - t, h), "b": (w - t, w, h / 2, h), "c": (w - t, w, 0, h / 2), "d": (0, w, 0, t),
           "e": (0, t, 0, h / 2), "f": (0, t, h / 2, h), "g": (0, w, (h - t) / 2, (h + t) / 2)}
    rects, u = [], 0.0
    for ch in text:
        if ch in ".1":  # narrow glyphs, so "21.0" doesn't read as "2 1.0"
            rects.append((u, u + t, 0, t if ch == "." else h))
            u += t + gap
        else:
            rects += [(u + u0, u + u1, v0, v1) for u0, u1, v0, v1 in (seg[s] for s in SEVEN[ch])]
            u += w + gap
    u -= gap
    return [(u0 - u / 2, u1 - u / 2, v0 - h / 2, v1 - h / 2) for u0, u1, v0, v1 in rects]


def engrave_band(text, r, z_mid, depth=0.5):
    """Cutter for text inside the bore opposite the head, upright along the finger axis when read from inside."""
    cut = Manifold()
    for u0, u1, v0, v1 in glyph_rects(text):
        phi = 1.5 * math.pi - (u0 + u1) / 2 / r  # seen from inside, left-to-right runs toward -phi
        box = Manifold.cube([u1 - u0, depth + 1.0, v1 - v0]).translate([-(u1 - u0) / 2, r - 1.0, z_mid + v0])
        cut += box.rotate([0, 0, math.degrees(phi) - 90])
    return cut


def us_to_diameter(us):
    return 11.63 + 0.8128 * us


def build(inner_d=17.3, card="microsd", flat=False, resin=False, band_t=1.8, band_w=5.0,
          floor=1.0, roof=1.2, side=1.0, end=1.2, detent=None, label=None, n_phi=360):
    L, W, T = CARDS[card]
    ct, cw, cl = (0.2, 0.25, 0.2) if resin else (0.35, 0.4, 0.3)  # slot clearance: thickness, width, length
    detent = (0.1 if resin else 0.25) if detent is None else detent
    sL, sW, sT = L + cl, W + cw, T + ct
    ri = inner_d / 2
    y0 = ri + floor  # slot spans y0..y1, flat, right above the finger
    y1 = y0 + sT
    ytop = y1 + roof
    hx, hz = sL / 2 + end, sW / 2 + side  # head half-extents across / along the finger

    # Outline seen down the finger: the band circle hulled with a rounded plate for the head.
    poly = convex_hull(np.vstack([
        circle(0, 0, ri + band_t, 256),
        *[circle(sx * (hx - 1.5), ytop - 1.5, 1.5, 48) for sx in (-1, 1)],
        *[circle(sx * (hx - 2.0), y0 - 0.2, 2.0, 64) for sx in (-1, 1)],
    ]))
    assert abs(poly[:, 1].max() - ytop) < 1e-6  # nothing pokes above the roof
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    r_out = ray_radius(poly, phis)

    # Width along the finger: full head width over the slot, easing down to the band underneath.
    alpha = np.abs((phis - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi)  # angle away from the head
    a0 = math.atan2(hx, y0) + math.radians(6)
    a1 = min(a0 + math.radians(75), math.radians(170))
    s = np.clip((alpha - a0) / (a1 - a0), 0, 1)
    width = band_w + (2 * hz - band_w) * 0.5 * (1 + np.cos(np.pi * s))
    z_lo = np.full(n_phi, -hz) if flat else -width / 2
    ring = loft(phis, ri, r_out, z_lo, z_lo + width, f_in=0.6, f_out=0.8)

    slot = Manifold.cube([4 * hx, sT, sW]).translate([-sL / 2, y0, -sW / 2])
    c = 0.4  # lead-in chamfer around the mouth
    lead_in = Manifold.hull_points(np.array([
        (x, y, z)
        for x, e in ((hx + 0.01, c), (hx - c, 0.0))
        for y in (y0 - e, y1 + e)
        for z in (-sW / 2 - e, sW / 2 + e)
    ]))
    rn = min(4.0, sW / 2 - 1.0)  # fingernail notch over the mouth
    notch = Manifold.cylinder(10, rn, circular_segments=64).rotate([-90, 0, 0]).translate([hx + 1.0, y1 - 0.01, 0])
    pin = Manifold.cylinder(end + 2, 0.6, circular_segments=24).rotate([0, 90, 0]).translate([-hx - 1, (y0 + y1) / 2, 0])
    ring = ring - slot - lead_in - notch - pin
    if detent > 0:  # two bumps on the floor press the card against the roof so it can't slide out
        for z in (-sW / 4, sW / 4):
            ring += Manifold.sphere(0.5, 24).translate([sL / 2 - 3.0, y0 + detent - 0.5, z])
    k = int(np.argmin(np.abs(phis - 1.5 * np.pi)))  # bottom of the band, opposite the head
    label_at = np.array([0.0, -ri, z_lo[k] + width[k] / 2])
    label_cut = None
    if label:
        cutter = engrave_band(label, ri, label_at[2])
        label_cut = ring ^ cutter  # the removed material, only for highlighting in previews
        ring -= cutter

    # Sanity checks: the card fits seated, and nothing but the bumps blocks its way out.
    seated = Manifold.cube([L, T, W]).translate([-sL / 2 + 0.01, y0 + detent + 0.001, -W / 2])
    path = Manifold.cube([L + 4 * hx, T, W]).translate([-sL / 2 + 0.01, y0 + detent + 0.001, -W / 2])
    info = dict(
        head=(2 * hx, 2 * hz, ytop - ri), slot=(sL, sW, sT), detent=detent,
        seated_overlap=(ring ^ seated).volume(), exit_overlap=(ring ^ path).volume(),
        volume=ring.volume(), genus=ring.genus(), status=str(ring.status()),
        y0=y0, y1=y1, L=L, W=W, T=T, sL=sL, label_at=label_at, label_cut=label_cut,
    )
    return ring, seated, info


def write_stl(m, path, name=b"sd ring"):
    mesh = m.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3].astype(np.float64)
    tri = v[np.asarray(mesh.tri_verts)]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as f:
        f.write(name.ljust(80, b" ") + struct.pack("<I", len(tri)) + rec.tobytes())


ROT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # model -> plot frame: head points up


def orbit(elev, azim):
    """Camera (direction toward the camera, up hint) from elevation/azimuth in degrees."""
    e, a = math.radians(elev), math.radians(azim)
    return (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)), (0, 0, 1)


def raster(parts, camera, zoom, rot=ROT, size=520, ss=2):
    """Orthographic z-buffer render of [(manifold, rgb)] -> (size, size, 3) image."""
    view, up_hint = (np.asarray(v, float) for v in camera)
    view /= np.linalg.norm(view)
    right = np.cross(up_hint, view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    light = view + 0.6 * up - 0.4 * right
    light /= np.linalg.norm(light)
    tris, cols = [], []
    for m, color in parts:
        mesh = m.to_mesh()
        t = (np.asarray(mesh.vert_properties)[:, :3] @ rot.T)[np.asarray(mesh.tri_verts)]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        keep = n @ view > 0  # closed meshes: back faces are never visible
        lam = np.clip(n[keep] @ light, 0, 1)
        spec = np.clip(n[keep] @ ((light + view) / np.linalg.norm(light + view)), 0, 1) ** 40
        cols.append(np.clip(np.asarray(color) * (0.28 + 0.72 * lam)[:, None] + 0.35 * spec[:, None], 0, 1))
        tris.append(t[keep])
    tris, cols = np.vstack(tris), np.vstack(cols)
    if zoom is None:
        pts = tris.reshape(-1, 3)
        ctr = (pts.max(0) + pts.min(0)) / 2
        sx, sy = pts @ right, pts @ up
        rad = max(np.ptp(sx), np.ptp(sy)) / 2 * 1.08
    else:
        ctr, rad = zoom
    px = size * ss
    scr = np.stack([((tris - ctr) @ right / rad + 1) * px / 2, (1 - (tris - ctr) @ up / rad) * px / 2, tris @ view], -1)
    zbuf = np.full((px, px), -np.inf)
    img = np.ones((px, px, 3))
    for (p0, p1, p2), c in zip(scr, cols):
        x0, x1 = max(int(min(p0[0], p1[0], p2[0])), 0), min(int(max(p0[0], p1[0], p2[0])) + 1, px)
        y0, y1 = max(int(min(p0[1], p1[1], p2[1])), 0), min(int(max(p0[1], p1[1], p2[1])) + 1, px)
        if x0 >= x1 or y0 >= y1:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        d = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
        if abs(d) < 1e-12:
            continue
        w1 = ((gx - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (gy - p0[1])) / d
        w2 = ((p1[0] - p0[0]) * (gy - p0[1]) - (gx - p0[0]) * (p1[1] - p0[1])) / d
        w0 = 1 - w1 - w2
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        z = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]
        sub = zbuf[y0:y1, x0:x1]
        hit = inside & (z > sub)
        sub[hit] = z[hit]
        img[y0:y1, x0:x1][hit] = c
    return img.reshape(size, ss, size, ss, 3).mean((1, 3))


def render(panels, path, shape=(2, 2)):
    """panels: (title, [(manifold, rgb)], camera, zoom, rot) per subplot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    fig, axes = plt.subplots(*shape, figsize=(5.5 * shape[1], 5.5 * shape[0]), dpi=110)
    for ax, (title, parts, camera, zoom, rot) in zip(np.ravel(axes), panels):
        ax.imshow(raster(parts, camera, zoom, rot))
        ax.set_axis_off()
        ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.2)


GOLD, BLUE, DARK = (0.93, 0.75, 0.38), (0.25, 0.42, 0.85), (0.35, 0.2, 0.08)


def make_set(args, out, cols=5, gap=4.0):
    """Flat rings for every half US size in range, diameter engraved, plus one plate with all of them."""
    set_dir = out / "rings"
    set_dir.mkdir(exist_ok=True)
    suffix = "_resin" if args.resin else ""
    made = []
    print(" US   inner d   circumference  HK~   volume")
    for us in np.arange(args.us_range[0], args.us_range[1] + 1e-9, 0.5):
        d = round(us_to_diameter(us), 1)  # rounded first, so the engraved number is the exact geometry
        ring, _, info = build(d, args.card, flat=True, resin=args.resin, detent=args.detent, label=f"{d:.1f}")
        assert info["status"] == "Error.NoError" and info["seated_overlap"] < 1e-6 and info["exit_overlap"] < 1e-6
        on_bed = ring.translate([0, 0, -ring.bounding_box()[2]])
        write_stl(on_bed, set_dir / f"sd_ring_{d:.1f}mm_US{us:g}{suffix}.stl")
        made.append((us, d, ring, on_bed, info))
        print(f"{us:4g}   {d:5.1f} mm   {math.pi * d:6.1f} mm   {math.pi * d - 40:4.0f}   {info['volume']:.0f} mm3")

    # Lay them out smallest-first in rows on the bed, all flat side down.
    placed, y, total = [], 0.0, 0.0
    for r0 in range(0, len(made), cols):
        x, row_h = 0.0, 0.0
        for _, _, _, m, info in made[r0:r0 + cols]:
            x0, y0, _, x1, y1, _ = m.bounding_box()
            placed.append(m.translate([x - x0, y - y0, 0]))
            x += x1 - x0 + gap
            row_h = max(row_h, y1 - y0)
            total += info["volume"]
        y += row_h + gap
    plate = Manifold.compose(placed)
    x0, y0, _, x1, y1, z1 = plate.bounding_box()
    name = f"sizing_plate_US{args.us_range[0]:g}-{args.us_range[1]:g}{suffix}"
    write_stl(plate, out / f"{name}.stl")
    print(f"{len(made)} rings, plate {x1 - x0:.0f} x {y1 - y0:.0f} x {z1:.1f} mm, "
          f"{total / 1000:.1f} cm3 (PLA ~{total * 1.24e-3:.0f} g)")

    if not args.no_preview:
        us, d, ring, _, info = made[len(made) // 2]
        near_half_removed = ring.trim_by_plane([0, -1, 0], 0.0)  # so the camera can see inside the bore
        render([
            ("整版（平面朝下，从小到大排列）", [(plate, GOLD)], ((0, -0.55, 1), (0, 1, 0)), None, np.eye(3)),
            (f"戒圈内侧刻字：内径 {d:.1f} mm（美码 {us:g}，剖开示意，刻痕填色）",
             [(near_half_removed, GOLD), (info["label_cut"], DARK)], ((0.0, 1.0, 0.25), (0, 0, 1)),
             (info["label_at"], 6.5), np.eye(3)),
        ], out / f"{name}_preview.png", shape=(1, 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inner-d", type=float, default=17.3, help="ring inner diameter in mm (default 17.3 ~ US 7)")
    ap.add_argument("--card", choices=sorted(CARDS), default="microsd")
    ap.add_argument("--resin", action="store_true", help="tighter clearances for resin printers")
    ap.add_argument("--detent", type=float, default=None, help="bump height in mm (0 = none; default 0.25 FDM / 0.1 resin)")
    ap.add_argument("--set", action="store_true", help="sizing set: every half US size in --us-range, one plate")
    ap.add_argument("--us-range", type=float, nargs=2, default=(4, 13), metavar=("MIN", "MAX"))
    ap.add_argument("--out", type=Path, default=Path("out"), help="output directory (default: ./out)")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.set:
        make_set(args, out)
        return
    stem = f"sd_ring_{args.card}_{args.inner_d:g}mm{'_resin' if args.resin else ''}"
    rings = {}
    for flat in (False, True):
        ring, card, info = build(args.inner_d, args.card, flat=flat, resin=args.resin, detent=args.detent)
        tag = "flat" if flat else "signet"
        lo = ring.bounding_box()[2]
        write_stl(ring.translate([0, 0, -lo]), out / f"{stem}_{tag}.stl")
        rings[tag] = (ring, card, info)
        hx, hz, h = info["head"]
        print(f"[{tag}] status={info['status']} genus={info['genus']} volume={info['volume']:.0f} mm3 "
              f"(PLA ~{info['volume'] * 1.24e-3:.1f} g)  head {hx:.1f} x {hz:.1f} mm, {h:.2f} mm above the finger  "
              f"slot {info['slot'][0]:.2f} x {info['slot'][1]:.2f} x {info['slot'][2]:.2f}  "
              f"card overlap seated={info['seated_overlap']:.4f} exit-path={info['exit_overlap']:.4f} mm3")

    if not args.no_preview:
        ring, card, info = rings["signet"]
        roofless = ring.trim_by_plane([0, -1, 0], -(info["y1"] - 0.05))  # just under the slot ceiling
        head = (np.array([0.0, 0.0, info["y0"]]), max(info["head"][:2]) * 0.75)
        render([
            ("整体（对称款）", [(ring, GOLD)], orbit(22, -58), None, ROT),
            ("卡槽口 · 指甲槽 · 导入倒角", [(ring, GOLD)], orbit(12, 18), head, ROT),
            (f"去掉顶盖：{'microSD' if args.card == 'microsd' else 'SD'} 卡放进去的位置",
             [(roofless, GOLD), (card, BLUE)], orbit(58, -70), head, ROT),
            ("平底款：这一面朝下平放打印（另一端是顶针孔）", [(rings["flat"][0], GOLD)], orbit(8, 140), None, ROT),
        ], out / f"{stem}_preview.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
