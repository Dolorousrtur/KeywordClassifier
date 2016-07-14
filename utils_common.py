import os


def dictlist_append(dict, key, val):
    """
    appends element to a list-value in a dictionary (adds the list and its key to the dictionary in case it was not there)
    :param dict: dictionary
    :param key: key in the dictionary
    :param val: value to append
    """
    if key not in dict:
        dict[key] = []
    dict[key].append(val)


def contains(list, sublist):
    """
    checks whether a sublist is occured in a list
    :param list:
    :param sublist:
    :return: False if there is no occurece, bounds of the first occurence otherwise
    """
    if sublist[0] not in list:
        return False
    else:
        indices = [i for i, x in enumerate(list) if x == sublist[0]]
        for ind in indices:
            occ = True

            for i in range(1, len(sublist)):
                if ind+i >= len(list) or list[ind+i] != sublist[i]:
                    occ = False
                    break
            if occ:
                return (ind, ind+len(sublist))
    return False


def raw_filename(filename):
    """
    split file path into its directory name and raw filename (without extention)
    :param filename:
    :return: (directory name, raw filename)
    """
    filename = os.path.splitext(filename)[0]
    directory = ""
    if '/' in filename:
        ind = filename.rindex('/')
        raw_filename = filename[ind+1:]
        directory = filename[:ind]
    else:
        raw_filename = filename
    return directory, raw_filename