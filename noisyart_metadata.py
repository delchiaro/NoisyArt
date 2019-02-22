import json
import os
from noisyart import DEFAULT_NOISYART_PATH

NOISYART_PATH = DEFAULT_NOISYART_PATH
JSON_RPATH = "metadata.json"
CLASSES_RPATH = "imgs/classes"


def uriToFolder(uri):
    return uri.replace("/", "^")

def folderToUri(folder):
    if folder[0:4].isdigit():
        folder=folder[5:]
    return folder.replace("^", "/")


import pprint
json_path = os.path.join(NOISYART_PATH, JSON_RPATH)
classes_path = os.path.join(NOISYART_PATH, CLASSES_RPATH)

def show_uri_metadata(uri, json_path=json_path, classes_path=classes_path):
    list_of_artworks = json.loads(open(json_path).read())


    classes = sorted(os.listdir(classes_path))
    classes = [cls[5:] for cls in classes]

    try:
        artworks_dict = {a['uri']: a for a in list_of_artworks}
        artwork = artworks_dict[uri]
        index = classes.index(uriToFolder(uri))
        print("Class/Index: {}".format(index))
        print("Artwork URI: {}".format(uri))
        pprint.pprint(artwork)
        return 0
    except KeyError:
        print("Can't find artwork with uri: '{}`".format(uri))
        return -1
    except ValueError:
        print("Can't find index of artwork with uri: '{}`".format(uri))
        return -2

import argparse, textwrap
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(
        prog='noisyart-metadata',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='This program will show metadata associated with classes of the NoisyArt dataset.',
        epilog=textwrap.dedent('''\
           NB: Some classes/folder-names could have special characters like apex ' .
           You should escape (e.g. \\' ) when you call this program from shell.''')
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-u", "--uri", action='store', default=None,
                        help="The uri of the artwork of which display metadata.")
    group.add_argument("-f", "--folder", action='store', default=None,
                        help="The folder-name/label of the artwork of which display metadata.")

    parser.add_argument("-p", "--path", action='store', default=NOISYART_PATH,
                        help="Specify the path of NoisyArt dataset folder.")

    parser.add_argument("-cp", "--classes-path", action='store', default=None,
                        help="Specify the path of the folder containing all the dataset class folders. "
                             "If not specified, it will be {path-to-NoisyArt}/imgs/classes")

    parser.add_argument("-jp", "--json-path",  action='store', default=None,
                        help="Specify the path of json file containing NoisyArt metadata. "
                             "If not specified, it will be {path-to-NoisyArt}/metadata.json")

    args = parser.parse_args()

    classes_path = os.path.join(args.path, CLASSES_RPATH) if args.classes_path is None else args.classes_path
    json_path = os.path.join(args.path, JSON_RPATH) if args.json_path is None else args.json_path

    if args.uri is not None:
        print(f"Showing info for uri: {args.uri}")
        ret = show_uri_metadata(args.uri, json_path, classes_path)
    elif args.folder is not None:
        print(f"Showing info for folder: {args.folder}")
        uri = folderToUri(args.folder)
        print(f"Folder converted to uri: {uri}")
        ret = show_uri_metadata(uri, json_path, classes_path)
    else:
        raise Exception()
    if ret < 0:
        print("NB: Be careful about special characters in folder-name or URI.\n    You should escape them with a backslash.")

    #input("Press Enter to continue...")
