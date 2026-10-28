"""Icosahedron geometry rebuilt from the 20 measured face-centre directions."""
import numpy as np
from scipy.spatial import ConvexHull


def direction(az_deg: float, el_deg: float):
    """Rig frame, y up: azimuth 0 = -z (the rig's reference direction), clockwise seen from above, so +x is 90 deg."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.sin(az) * np.cos(el), np.sin(el), -np.cos(az) * np.cos(el)])


def icosahedron(faces):
    """faces: [{'face': id, 'az': deg, 'el': deg}] x 20 -> {'vertices': [[x,y,z]]*12, 'triangles': {face: [i,j,k]}}.

    The face normals of an icosahedron are the vertices of a dodecahedron; the hull of those 20 points has 12
    pentagonal facets whose outward normals are the icosahedron's vertices. A face triangle is the three vertices
    whose pentagon contains that face's point.
    """
    if len(faces) != 20:
        return None
    pts = np.array([direction(f["az"], f["el"]) for f in faces])
    hull = ConvexHull(pts)
    groups = {}
    for simplex, eq in zip(hull.simplices, hull.equations):
        key = tuple(np.round(eq[:3], 3))
        groups.setdefault(key, set()).update(int(i) for i in simplex)
    if len(groups) != 12:
        return None
    verts = []
    members = []
    for key, idx in groups.items():
        n = np.array(key)
        verts.append((n / np.linalg.norm(n)).tolist())
        members.append(idx)
    tris = {}
    for i, f in enumerate(faces):
        owners = [v for v, m in enumerate(members) if i in m]
        if len(owners) != 3:
            return None
        tris[int(f["face"])] = owners
    return {"vertices": verts, "triangles": tris}
