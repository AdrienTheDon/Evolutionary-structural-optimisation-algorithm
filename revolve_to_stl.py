import argparse
import math
from PIL import Image
import numpy as np
from stl import mesh

def row_segments(row):
    segments=[]
    start=None
    for i,v in enumerate(row):
        if v and start is None:
            start=i
        elif not v and start is not None:
            segments.append((start,i))
            start=None
    if start is not None:
        segments.append((start,len(row)))
    return segments


def image_to_stl(image_path, out_path, segments=60):
    # Load image as grayscale and convert to binary mask
    img = Image.open(image_path).convert('L')
    arr = np.array(img)
    mask = arr > 128
    height = mask.shape[0]

    triangles = []
    two_pi = 2 * math.pi
    for y in range(height - 1):
        row = mask[y]
        segs_row = row_segments(row)
        for seg in segs_row:
            r1, r2 = seg
            for k in range(segments):
                a1 = two_pi * k / segments
                a2 = two_pi * (k + 1) / segments
                ca1, sa1 = math.cos(a1), math.sin(a1)
                ca2, sa2 = math.cos(a2), math.sin(a2)
                b0 = np.array([r1 * ca1, y, r1 * sa1])
                b1 = np.array([r2 * ca1, y, r2 * sa1])
                b2 = np.array([r1 * ca2, y, r1 * sa2])
                b3 = np.array([r2 * ca2, y, r2 * sa2])
                t0 = np.array([r1 * ca1, y + 1, r1 * sa1])
                t1 = np.array([r2 * ca1, y + 1, r2 * sa1])
                t2 = np.array([r1 * ca2, y + 1, r1 * sa2])
                t3 = np.array([r2 * ca2, y + 1, r2 * sa2])
                # side outer
                triangles.append([b1, b3, t3])
                triangles.append([b1, t3, t1])
                # side inner
                triangles.append([b0, t0, t2])
                triangles.append([b0, t2, b2])
                # top
                triangles.append([t0, t1, t3])
                triangles.append([t0, t3, t2])
                # bottom
                triangles.append([b0, b2, b3])
                triangles.append([b0, b3, b1])

    data = np.zeros(len(triangles), dtype=mesh.Mesh.dtype)
    m = mesh.Mesh(data)
    for i, tri in enumerate(triangles):
        m.vectors[i] = np.array(tri)
    m.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Revolve a binary image around its vertical axis and export as STL")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("output", help="Output STL file")
    parser.add_argument("--segments", type=int, default=60, help="Number of angular segments for revolution")
    args = parser.parse_args()
    image_to_stl(args.image, args.output, segments=args.segments)

if __name__ == "__main__":
    main()
