import argparse

def gettrainargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--extended", help="Use Extended Linear Mapping", action="store_true")
    parser.add_argument("-q", "--qmatrix", help="Use file as Q matrix")
    parser.add_argument("-v", "--vmatrix", help="Use file as V matrix")
    parser.add_argument("-i", "--input", help="Specify training set")
    parser.add_argument("-o", "--output", help="File to save filter")
    parser.add_argument("-p", "--plot", help="Plot the learned filters", action="store_true")
    parser.add_argument("-cu", "--cubic", help="Use bicubic for init",
    action="store_true")
    parser.add_argument("-li", "--linear", help="Use bilinear for init",
    action="store_true")
    parser.add_argument("-cgls", "--cgls", help="Use conjugate gradient least square",
    action="store_true")
    parser.add_argument("-ls", "--ls", help="Use normalized least square with normalization factor lambda -l",
    action="store_true")
    parser.add_argument("-l", "--l", help="Normalization factor lambda")
    args = parser.parse_args()
    return args
