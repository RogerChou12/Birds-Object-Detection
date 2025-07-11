import json
import matplotlib.pyplot as plt
from collections import Counter

# Load the JSON file
with open('annotations.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

num_boxes_per_image = []
confidences = []
areas = []
category_all = []
for entry in data:
    objects = entry.get('objects', [])
    # If objects is a dict, convert to list
    if isinstance(objects, dict):
        areas_all, boxes, conf, categories = objects["area"], objects["bbox"], objects["confidence"], objects["category"]
    num_boxes = len(boxes)
    num_boxes_per_image.append(num_boxes)
    confidences.extend(conf)
    category_all.extend(categories)
    # Assume bbox is [x, y, width, height]
    for bbox in boxes:
        if bbox and len(bbox) == 4:
            area = bbox[2] * bbox[3]
            areas.append(area)

# Plot histograms
plt.figure(figsize=(18, 4))

plt.subplot(1, 4, 1)
plt.hist(num_boxes_per_image, bins=range(0, max(num_boxes_per_image)+2), color='skyblue', edgecolor='black')
plt.xlabel('Number of Boxes per Image')
plt.ylabel('Count')
plt.title('Boxes per Image')

plt.subplot(1, 4, 2)
plt.hist(confidences, bins=20, color='orange', edgecolor='black')
plt.xlabel('Confidence')
plt.ylabel('Count')
plt.title('Confidence Histogram')

plt.subplot(1, 4, 3)
plt.hist(areas, bins=30, color='green', edgecolor='black')
plt.xlabel('Box Area')
plt.ylabel('Count')
plt.title('Area Histogram')

# Pie chart for category distribution
plt.subplot(1, 4, 4)
category_counts = Counter(category_all)
labels = [str(cat) for cat in category_counts.keys()]
sizes = category_counts.values()
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Category Distribution')

plt.tight_layout()
plt.savefig('data_analysis.png')
plt.show()
