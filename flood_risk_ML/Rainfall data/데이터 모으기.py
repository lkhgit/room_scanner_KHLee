import csv
import glob
import os

folder = os.path.dirname(os.path.abspath(__file__))
output = os.path.join(folder, "강우량_2009_2026.csv")

files = sorted(
    f for f in glob.glob(os.path.join(folder, "*.csv"))
    if os.path.abspath(f) != output
)

header = None
rows = {}  # (지점, 일시) -> row, 겹치는 시간은 하나만 남김
total = 0
conflicts = 0

for f in files:
    with open(f, encoding="cp949", newline="") as fp:
        reader = csv.reader(fp)
        file_header = next(reader)
        if header is None:
            header = file_header
        for row in reader:
            if not row:
                continue
            total += 1
            key = (int(row[0]), row[2])
            if key in rows:
                if rows[key] != row:
                    conflicts += 1
                continue
            rows[key] = row

with open(output, "w", encoding="utf-8-sig", newline="") as fp:
    writer = csv.writer(fp)
    writer.writerow(header)
    for key in sorted(rows):
        writer.writerow(rows[key])

times = [k[1] for k in rows]
print(f"파일 {len(files)}개, 행 {total} -> 중복 제거 후 {len(rows)} (값이 다른 중복 {conflicts}건)")
print(f"기간: {min(times)} ~ {max(times)}")
print(f"지점 수: {len({k[0] for k in rows})}")
print(f"저장: {output}")
