import argparse
import os

from src.scale_detector.scale_detector import ScaleDetector
from src.visualization_utils import visualize_scale_detection_results


def get_image_name(path):
    file_name = os.path.basename(path)
    return file_name.split('.')[0]

def detect_scale_on_the_image(args, image_path):
    if image_path:
        candidate_segments, pixel_len, all_numbers, numbers_on_one_line = \
            ScaleDetector.detect_ruler_scale(image_path, args.grouping, args.model_type)

        if not pixel_len:
            print('Not able to detect ruler scale for this image')
        else:            
            print(f'For image: "{image_path}": 1 cm/in - {pixel_len} px')
            if args.show_figs and pixel_len:
                visualize_scale_detection_results(args.image_path, all_numbers, candidate_segments, pixel_len)

        return pixel_len

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pass -i and path to image to detect 1 cm/in to pixels ratio')
    parser.add_argument('-i', dest='image_path')
    parser.add_argument('-f', dest='image_paths_file_path')
    parser.add_argument('-s', dest='save_results_file_path')
    parser.add_argument('-v', dest='show_figs')
    parser.add_argument('-g', dest='grouping', default='cluster')
    parser.add_argument('-m', dest='model_type', default='v5')

    args = parser.parse_args()

    results = {}
    if args.image_paths_file_path:
        with open(args.image_paths_file_path) as file:
            for index, line in enumerate(file):
                image_name = get_image_name(line)
                pixel_len = detect_scale_on_the_image(args, line.strip())
                results[image_name] = pixel_len

    if args.image_path:
        image_name = get_image_name(args.image_path)
        pixel_len = detect_scale_on_the_image(args, args.image_path.strip())
        results[image_name] = pixel_len

    if args.save_results_file_path:
        with open(args.save_results_file_path, 'w') as file:
            file.write('%s,%s\n' % ('img_id', 'px_per_unit'))
            for key, value in results.items(): 
                file.write('%s,%s\n' % (key, value))
