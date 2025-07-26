# Multi-Class Charpy Annotation Strategy

## Overview
You currently have 164 images with only `charpy_specimen` annotations (class 0). To enable multi-class detection, you need to add detailed annotations for the other 4 classes.

## Class Definitions

### 1. charpy_specimen (Class 0) ✅ Already Done
- **Purpose**: Overall specimen detection
- **What to annotate**: The entire Charpy specimen
- **Status**: Already annotated in all images

### 2. charpy_edge (Class 1) 🔴 Needs Annotation
- **Purpose**: Detect specimen edges/boundaries
- **What to annotate**: The outer edges of the specimen
- **Strategy**: Draw boxes around the visible edges of the specimen
- **Typical count**: 2-4 boxes per image (top, bottom, left, right edges)

### 3. charpy_corner (Class 2) 🔴 Needs Annotation  
- **Purpose**: Detect specimen corners
- **What to annotate**: Corner regions where edges meet
- **Strategy**: Small boxes at the corners of the specimen
- **Typical count**: 2-4 boxes per image (depending on visible corners)

### 4. fracture_surface (Class 3) 🔴 Needs Annotation
- **Purpose**: Detect the fracture/break area
- **What to annotate**: The area where the specimen broke during testing
- **Strategy**: Box around the fractured/broken surface
- **Typical count**: 1-2 boxes per image (main fracture area)

### 5. measurement_point (Class 4) 🔴 Needs Annotation
- **Purpose**: Detect areas where measurements should be taken
- **What to annotate**: Specific points for dimensional measurements
- **Strategy**: Small boxes at key measurement locations
- **Typical count**: 2-6 boxes per image (measurement points)

## Annotation Workflow

### Phase 1: Start with Training Set (93 images)
1. **Focus on charpy_edge first** - Most visible and consistent
2. **Then charpy_corner** - Usually clear corner regions
3. **Add fracture_surface** - Look for broken/fractured areas
4. **Finally measurement_point** - Key dimensional points

### Phase 2: Validation Set (45 images)
- Apply same strategy as training set
- Ensure consistency with training annotations

### Phase 3: Test Set (26 images)
- Complete the annotation process
- Maintain annotation quality

## Annotation Tips

### For charpy_edge:
- Look for clear boundaries between specimen and background
- Include both straight and curved edges
- Don't overlap too much with the overall specimen box

### For charpy_corner:
- Focus on actual geometric corners
- Make boxes small enough to be specific
- Usually at intersections of edges

### For fracture_surface:
- Look for rough, broken surfaces
- May have different texture/color than rest of specimen
- Often in the center or along a crack line

### For measurement_point:
- Think about where you'd place calipers or measuring tools
- Key dimensional points (width, height, thickness indicators)
- Often at edges or specific geometric features

## Quality Guidelines

### Good Annotations:
- ✅ Tight bounding boxes around features
- ✅ Consistent across similar images
- ✅ Clear class distinctions
- ✅ Multiple examples per class per image

### Avoid:
- ❌ Boxes that are too large/loose
- ❌ Overlapping boxes of different classes
- ❌ Inconsistent annotation styles
- ❌ Missing obvious features

## Recommended Annotation Order

1. **Start with 10-20 training images** to establish your annotation style
2. **Run analysis** to check class distribution
3. **Adjust strategy** based on initial results
4. **Complete training set**
5. **Annotate validation set**
6. **Complete test set**

## Time Estimation

- **Per image**: 2-5 minutes (depending on complexity)
- **Training set (93 images)**: 3-8 hours
- **Full dataset (164 images)**: 5-14 hours

## Tools Available

### multi_class_annotator.py
- Interactive annotation tool
- Mouse-based bounding box drawing
- Class switching with keyboard shortcuts
- Save/load existing annotations
- Navigate between images

### analyze_dataset.py
- Check annotation progress
- Verify class distributions
- Identify issues

## Success Metrics

After annotation, you should have:
- **Class 0 (charpy_specimen)**: ~164 annotations (1 per image)
- **Class 1 (charpy_edge)**: ~300-600 annotations (2-4 per image)
- **Class 2 (charpy_corner)**: ~200-400 annotations (1-3 per image)
- **Class 3 (fracture_surface)**: ~100-200 annotations (0-2 per image)
- **Class 4 (measurement_point)**: ~300-800 annotations (2-6 per image)

**Total target**: ~1000-2000+ annotations across all classes