import argparse

def gettestargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filter", help="Use file as filter")
    parser.add_argument("-p", "--plot", help="Visualizing the process of RAISR image upscaling",
     action="store_true")
    parser.add_argument("-e", "--extended", help="Use Extended Linear Mapping", action="store_true")
    parser.add_argument("-o", "--output", help="output folder name")
    parser.add_argument("-i", "--input", help="input folder name")
    parser.add_argument("-gt", "--groundTruth", help="Use test images as ground truth (down scale them first)",
    action="store_true")
    parser.add_argument("-li", "--linear", help="Use bilinear for init",
    action="store_true")
    parser.add_argument("-ex2", "--ex2", help="Use normalized features for ExLM",
    action="store_true")
    parser.add_argument("-cv2", "--cv2", help="Use cv2 interpolation",
    action="store_true")
    args = parser.parse_args()
    return args
