import argparse

def gettrainargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--extended", help="Use Extended Linear Mapping", action="store_true")
    parser.add_argument("-q", "--qmatrix", help="Use file as Q matrix")
    parser.add_argument("-v", "--vmatrix", help="Use file as V matrix")
    parser.add_argument("-i", "--input", help="Specify training set")
    parser.add_argument("-o", "--output", help="File to save filter")
    parser.add_argument("-p", "--plot", help="Plot the learned filters", action="store_true")
    parser.add_argument("-li", "--linear", help="Use bilinear for init",action="store_true")
    parser.add_argument("-ls", "--ls", help="Use normalized least square with normalization factor lambda -l",
        action="store_true")
    parser.add_argument("-l", "--l", help="Normalization factor lambda")
    parser.add_argument("-ex2", "--ex2", help="Use normalized features for ExLM",
        action="store_true")
    parser.add_argument("-cv2", "--cv2", help="Use cv2 interpolation",
        action="store_true")
    args = parser.parse_args()
    return args
