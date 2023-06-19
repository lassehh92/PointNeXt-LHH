import argparse
import laspy

def print_las_classes(filename):
    # Open the .las file
    las_data = laspy.read(filename)

    # Get the classifications
    classifications = las_data.classification

    # Print unique classes in the las file
    unique_classes = set(classifications)
    print("Unique Classes in the LAS file: ", unique_classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process .las file.')
    parser.add_argument('lasfile', type=str, help='Path to .las file')

    args = parser.parse_args()
    
    print_las_classes(args.lasfile)
