import re
import csv

# Parse alignment.txt
data = []
current_tma = None
current_dx = None
current_dy = None
current_scale_x = None
current_scale_y = None

with open('alignment.txt', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    line = line.strip()
    
    # Match TMA name - look for it before "Image size"
    if 'Image size' in line:
        # Look backwards for TMA identifier
        for j in range(max(0, i-5), i+1):
            tma_match = re.search(r'(TMA\d+(?:_\d+)?)', lines[j])
            if tma_match:
                current_tma = tma_match.group(1)
                current_dx = None
                current_dy = None
                current_scale_x = None
                current_scale_y = None
                break
    
    # Match cross-correlation shift
    shift_match = re.search(r'Cross-correlation shift: dx=([-\d]+), dy=([-\d]+)', line)
    if shift_match:
        current_dx = int(shift_match.group(1))
        current_dy = int(shift_match.group(2))
    
    # Match INK_SCALE_X
    scale_x_match = re.search(r'INK_SCALE_X\s*=\s*([\d.]+)', line)
    if scale_x_match:
        current_scale_x = float(scale_x_match.group(1))
    
    # Match INK_SCALE_Y
    scale_y_match = re.search(r'INK_SCALE_Y\s*=\s*([\d.]+)', line)
    if scale_y_match:
        current_scale_y = float(scale_y_match.group(1))
        
        # When we have all values, add to data
        if current_tma and current_dx is not None and current_dy is not None and current_scale_x is not None:
            # Use average of X and Y scales as scale_factor
            scale_factor = (current_scale_x + current_scale_y) / 2
            data.append({
                'tma': current_tma,
                'x_offset': current_dx,
                'y_offset': current_dy,
                'scale_factor': scale_factor
            })

# Write to CSV
with open('alignment.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['tma', 'x_offset', 'y_offset', 'scale_factor'])
    writer.writeheader()
    writer.writerows(data)

print(f"Parsed {len(data)} TMAs")
for row in data:
    print(f"{row['tma']}: x_offset={row['x_offset']}, y_offset={row['y_offset']}, scale_factor={row['scale_factor']:.11f}")
